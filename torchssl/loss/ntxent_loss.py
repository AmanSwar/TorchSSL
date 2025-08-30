import triton
import triton.language as tl

from torch.autograd.function import Function
import torch
import torch.nn as nn


@triton.jit
def _ntxent_fwd_kernel(
    z_ptr,  # ptr to concatinated embedding -> (zi and zj)
    loss_ptr,  # ptr to output loss
    BS,  # batch size
    N,  # total number of embedding
    D,  # embedding dim
    temp,  # temp scaling factor
    BLOCK_SIZE: tl.constexpr,
):

    pid = tl.program_id(0)
    row_z_start_ptr = pid * D + z_ptr
    row_z_ptrs = row_z_start_ptr + tl.arange(0, BLOCK_SIZE)
    row_z = tl.load(row_z_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)

    sum_exp = tl.zeros((), dtype=tl.float32)

    # Calculate denominator: sum of exp(sim(zi, zk)/temp) for all k != i
    for col_idx in range(0, N):
        if pid != col_idx:  # exclude self-similarity
            col_z_ptrs = z_ptr + col_idx * D + tl.arange(0, BLOCK_SIZE)
            col_z = tl.load(col_z_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)
            dot_product = tl.sum(row_z * col_z)
            sum_exp += tl.exp(dot_product / temp)

    log_sum_exp = tl.log(sum_exp)

    # Find positive pair index
    if pid < BS:
        pos_idx = pid + BS
    else:
        pos_idx = pid - BS

    # Load the positive pair embedding
    pos_z_ptrs = z_ptr + pos_idx * D + tl.arange(0, BLOCK_SIZE)
    pos_z = tl.load(pos_z_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)

    # Calculate positive similarity
    pos_sim = tl.sum(row_z * pos_z) / temp

    # Calculate loss for this sample
    row_loss = -pos_sim + log_sum_exp

    # Atomic add with proper type casting
    tl.atomic_add(loss_ptr, row_loss.to(tl.float32))


@triton.jit
def _ntxent_bwd_kernel(z_ptr, grad_z_ptr, BS, N, D, temp, BLOCK_SIZE: tl.constexpr):

    row_idx = tl.program_id(axis=0)

    row_z_ptrs = z_ptr + row_idx * D + tl.arange(0, BLOCK_SIZE)
    row_z = tl.load(row_z_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)

    # Calculate denominator for softmax
    sum_exp = tl.zeros((), dtype=tl.float32)
    for j in range(N):
        if j != row_idx:
            col_z_ptrs = z_ptr + j * D + tl.arange(0, BLOCK_SIZE)
            col_z = tl.load(col_z_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)
            dot_product = tl.sum(row_z * col_z)
            sum_exp += tl.exp(dot_product / temp)

    grad_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    # Find positive pair
    if row_idx < BS:
        pos_idx = row_idx + BS
    else:
        pos_idx = row_idx - BS

    # Calculate gradient contributions
    for j in range(N):
        if j != row_idx:  # skip self
            z_j_ptrs = z_ptr + j * D + tl.arange(0, BLOCK_SIZE)
            z_j = tl.load(z_j_ptrs, mask=tl.arange(0, BLOCK_SIZE) < D)

            dot_product = tl.sum(row_z * z_j)
            softmax_prob = tl.exp(dot_product / temp) / sum_exp

            if j == pos_idx:
                # Positive pair: gradient is (p_ij - 1) * z_j / temp
                grad_row += (softmax_prob - 1.0) * z_j / temp
            else:
                # Negative pair: gradient is p_ij * z_j / temp
                grad_row += softmax_prob * z_j / temp

    # Store gradient
    grad_z_out_ptrs = grad_z_ptr + row_idx * D + tl.arange(0, BLOCK_SIZE)
    tl.store(grad_z_out_ptrs, grad_row, mask=tl.arange(0, BLOCK_SIZE) < D)


class _NtxentlossWrapper(Function):

    @staticmethod
    def forward(ctx, z_i, z_j, temp):
        batch_size, D = z_i.shape
        N = 2 * batch_size
        z = torch.cat([z_i, z_j], dim=0)
        z = nn.functional.normalize(z, p=2, dim=1)

        loss = torch.zeros(1, device=z.device, dtype=torch.float32)
        ctx.save_for_backward(z)
        ctx.temp = temp

        grid = (N,)
        BLOCK_SIZE = triton.next_power_of_2(D)

        _ntxent_fwd_kernel[grid](
            z_ptr=z,
            loss_ptr=loss,
            BS=batch_size,
            N=N,
            D=D,
            temp=temp,
            BLOCK_SIZE=BLOCK_SIZE,
        )

        return loss / N

    @staticmethod
    def backward(ctx, grad_output):

        (z,) = ctx.saved_tensors  # Fixed: use saved_tensors instead of save_tensors

        temp = ctx.temp

        N, D = z.shape
        BS = N // 2
        grad_z = torch.zeros_like(z)

        grid = (N,)
        BLOCK_DIM = triton.next_power_of_2(D)

        _ntxent_bwd_kernel[grid](
            z_ptr=z,
            grad_z_ptr=grad_z,
            BS=BS,
            N=N,
            D=D,
            temp=temp,
            BLOCK_SIZE=BLOCK_DIM,
        )

        grad_z *= grad_output.squeeze()
        grad_z_i = grad_z[:BS]
        grad_z_j = grad_z[BS:]

        return grad_z_i, grad_z_j, None


class NTXentLossTriton(nn.Module):
    """
    NT-Xent loss implementation using a custom Triton kernel.
    """

    def __init__(self, temp=0.5):
        super().__init__()
        self.temp = temp

    def forward(self, z_i, z_j):
        return _NtxentlossWrapper.apply(z_i, z_j, self.temp)


if __name__ == "__main__":
    # Uncomment and adjust import as needed
    from torchssl.loss.python.ntxent import NTXentLoss
    import time

    batch_size = 1024
    feature_dim = 128

    z_i = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)
    z_j = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)

    # Comment out original_loss if import is not available
    original_loss = NTXentLoss(temp=0.5, device="cuda")

    triton_loss = NTXentLossTriton(temp=0.5)

    # warmup
    for _ in range(10):
        _ = original_loss(z_i, z_j)
        _ = triton_loss(z_i, z_j)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        loss1 = original_loss(z_i, z_j)
        loss1.backward()
        pass
    torch.cuda.synchronize()
    original_time = time.time() - start

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        loss2 = triton_loss(z_i, z_j)
        loss2.backward()
    torch.cuda.synchronize()
    cuda_time = time.time() - start

    print(f"Original implementation: {original_time:.4f}s")
    print(f"Triton implementation: {cuda_time:.4f}s")
    print(f"Speedup: {original_time / cuda_time:.2f}x")
