import triton
import triton.language as tl

from torch.autograd.function import Function
import torch
import torch.nn as nn

@triton.jit
def _ntxent_fwd_kernel(
    z_ptr, # ptr to concatinated embedding -> (zi and zj)
    loss_ptr, # ptr to output loss
    BS, #batch size
    N, # total number of embedding
    D, # embedding dim
    temp, # temp scaling factor
    BLOCK_SIZE : tl.constexpr
):

    pid = tl.program_id(0)
    row_z_start_ptr = pid * D + z_ptr
    row_z_ptrs = row_z_start_ptr + tl.arange(0 , BLOCK_SIZE)
    row_z = tl.load(row_z_ptrs)

    similarity_row = tl.zeros((BLOCK_SIZE,) , dtype=tl.float32)

    # not for similarity we need to multiple row with columns
    for col_idx in range(0 , N):

        col_z = tl.load(z_ptr + col_idx * D + tl.arange(0 , BLOCK_SIZE))
        dot_product = tl.sum(row_z * col_z)

        # excluding self similarity term -> i == j for denom calculations
        if pid != col_idx:
            similarity_row += tl.exp(dot_product / temp)

    log_sum_exp = tl.log(similarity_row)

    pos_idx = None
    if pid < BS:
        pos_idx = pid + BS
    else:
        pos_idx = pid - BS


    #load the positive pair embedding
    pos_z = tl.load(z_ptr + pos_idx * D + tl.arange(0, BLOCK_SIZE))

    pos_sim = tl.sum(row_z * pos_z) / temp

    row_loss = -pos_sim + log_sum_exp

    tl.atomic_add(loss_ptr , row_loss)


@triton.jit
def _ntxent_bwd_kernel(
    z_ptr,
    grad_z_ptr,
    N, D,
    temp,
    BLOCK_SIZE : tl.constexpr
):

    row_index = tl.program_id(axis=0)

    row_z = tl.load(z_ptr + row_index * D + tl.arange(0, BLOCK_SIZE))

    sum_exp = 0.0
    for j in range(N):
        if j != row_index:
            col_z = tl.load(z_ptr + j * D + tl.arange(0, BLOCK_SIZE))
            dot_product = tl.sum(row_z * col_z)
            sum_exp += tl.exp(dot_product / temp)

    grad_row = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    pos_idx = None
    if row_idx < batch_size:
        pos_idx = row_idx + batch_size
    else:
        pos_idx = row_idx - batch_size

    for j in range(N):
        # load the jth embedding
        z_j = tl.load(z_ptr + j * D + tl.arange(0, BLOCK_SIZE))

        # from -ve pairs
        if j != row_idx and j != pos_idx:
            dot_product = tl.sum(row_z * z_j)
            softmax_prob = tl.exp(dot_product / temp) / sum_exp
            grad_row += softmax_prob * z_j

    # from +ve pairs
    z_pos = tl.load(z_ptr + pos_idx * D + tl.arange(0, BLOCK_SIZE))
    dot_product_pos = tl.sum(row_z * z_pos)
    softmax_prob_pos = tl.exp(dot_product_pos / temp) / sum_exp

    grad_row += (softmax_prob_pos - 1) * z_pos

    final_grad = grad_row / (temp * N)
    tl.store(grad_z_ptr + row_idx * D + tl.arange(0, BLOCK_SIZE), final_grad)


class _NtxentlossWrapper(Function):

    @staticmethod
    def forward(ctx , z_i , z_j , temp):
        batch_size, N , D = z_i.shape
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
            BLOCK_SIZE=BLOCK_SIZE
        )

        return loss / N

    @staticmethod
    def backward(ctx , grad_output):

        z, = ctx.save_tensors

        temp = ctx.temp

        BS , N , D = z.shape

        grad_z = torch.zeros_like(z)

        grid = (N,)
        BLOCK_DIM = triton.next_power_of_2(D)

        _ntxent_bwd_kernel[grid](
            z_ptr=z,
            grad_z_ptr=grad_z,
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
    from torchssl.loss.python.ntxent import NTXentLoss
    import time

    batch_size = 1024
    feature_dim = 128

    z_i = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)
    z_j = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)
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
