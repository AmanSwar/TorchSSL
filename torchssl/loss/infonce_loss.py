import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.autograd.function import Function


@triton.jit
def _infonce_fwd_kernel(
    q_ptr,  # query embeddings [batch_size, dim]
    k_ptr,  # key embeddings [batch_size, dim]
    queue_ptr,  # negative queue [queue_size, dim]
    logits_ptr,  # output logits [batch_size, queue_size + 1]
    loss_ptr,  # output loss scalar
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):
    # Each program handles one sample in the batch
    batch_idx = tl.program_id(0)

    # Load query vector for this batch item
    q_offset = batch_idx * dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
    q_mask = tl.arange(0, BLOCK_DIM) < dim
    q_vec = tl.load(q_ptrs, mask=q_mask, other=0.0)

    # Load corresponding key vector
    k_offset = batch_idx * dim
    k_ptrs = k_ptr + k_offset + tl.arange(0, BLOCK_DIM)
    k_vec = tl.load(k_ptrs, mask=q_mask, other=0.0)

    # Compute positive similarity (q · k)
    l_pos = tl.sum(q_vec * k_vec) / temperature

    # Store positive logit at position 0
    logits_offset = batch_idx * (queue_size + 1)
    tl.store(logits_ptr + logits_offset, l_pos)

    # Compute negative similarities with queue
    for neg_idx in range(queue_size):
        neg_offset = neg_idx * dim
        neg_ptrs = queue_ptr + neg_offset + tl.arange(0, BLOCK_DIM)
        neg_vec = tl.load(neg_ptrs, mask=q_mask, other=0.0)

        l_neg = tl.sum(q_vec * neg_vec) / temperature
        tl.store(logits_ptr + logits_offset + neg_idx + 1, l_neg)

    # Compute softmax and loss for this sample
    max_logit = l_pos
    for neg_idx in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + neg_idx + 1)
        max_logit = tl.maximum(max_logit, neg_logit)

    # Compute log-sum-exp
    sum_exp = tl.exp(l_pos - max_logit)
    for neg_idx in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + neg_idx + 1)
        sum_exp += tl.exp(neg_logit - max_logit)

    log_sum_exp = tl.log(sum_exp) + max_logit
    sample_loss = -l_pos + log_sum_exp

    # Atomic add to total loss
    tl.atomic_add(loss_ptr, sample_loss)


@triton.jit
def _infonce_bwd_q_kernel(
    q_ptr,  # query embeddings
    k_ptr,  # key embeddings
    queue_ptr,  # negative queue
    grad_q_ptr,  # gradient w.r.t. queries
    logits_ptr,  # precomputed logits
    grad_output,  # gradient from upstream
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Load logits for this sample and compute softmax
    logits_offset = batch_idx * (queue_size + 1)
    l_pos = tl.load(logits_ptr + logits_offset)

    # Find max logit for numerical stability
    max_logit = l_pos
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        max_logit = tl.maximum(max_logit, neg_logit)

    # Compute softmax denominator
    sum_exp = tl.exp(l_pos - max_logit)
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        sum_exp += tl.exp(neg_logit - max_logit)

    # Compute positive probability
    prob_pos = tl.exp(l_pos - max_logit) / sum_exp

    # Initialize gradient
    grad_q = tl.zeros((BLOCK_DIM,), dtype=tl.float32)
    dim_mask = tl.arange(0, BLOCK_DIM) < dim

    # Gradient from positive pair: (p_0 - 1) * k / temperature
    k_offset = batch_idx * dim
    k_ptrs = k_ptr + k_offset + tl.arange(0, BLOCK_DIM)
    k_vec = tl.load(k_ptrs, mask=dim_mask, other=0.0)

    grad_q += (prob_pos - 1.0) * k_vec / temperature

    # Gradient from negative pairs: p_i * queue_i / temperature
    for neg_idx in range(queue_size):
        neg_offset = neg_idx * dim
        neg_ptrs = queue_ptr + neg_offset + tl.arange(0, BLOCK_DIM)
        neg_vec = tl.load(neg_ptrs, mask=dim_mask, other=0.0)

        # Compute probability for this negative
        neg_logit = tl.load(logits_ptr + logits_offset + neg_idx + 1)
        prob_neg = tl.exp(neg_logit - max_logit) / sum_exp

        grad_q += prob_neg * neg_vec / temperature

    # Scale by upstream gradient and store
    grad_q *= grad_output
    grad_q_offset = batch_idx * dim
    grad_q_ptrs = grad_q_ptr + grad_q_offset + tl.arange(0, BLOCK_DIM)
    tl.store(grad_q_ptrs, grad_q, mask=dim_mask)


@triton.jit
def _infonce_bwd_k_kernel(
    q_ptr,  # query embeddings
    k_ptr,  # key embeddings
    grad_k_ptr,  # gradient w.r.t. keys
    logits_ptr,  # precomputed logits
    grad_output,  # gradient from upstream
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    # Load query vector
    q_offset = batch_idx * dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
    dim_mask = tl.arange(0, BLOCK_DIM) < dim
    q_vec = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    # Load logits for softmax computation
    logits_offset = batch_idx * (queue_size + 1)
    l_pos = tl.load(logits_ptr + logits_offset)

    # Compute max for numerical stability
    max_logit = l_pos
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        max_logit = tl.maximum(max_logit, neg_logit)

    # Compute softmax denominator
    sum_exp = tl.exp(l_pos - max_logit)
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        sum_exp += tl.exp(neg_logit - max_logit)

    # Gradient for key: (p_0 - 1) * q / temperature
    prob_pos = tl.exp(l_pos - max_logit) / sum_exp
    grad_k = (prob_pos - 1.0) * q_vec / temperature

    # Scale by upstream gradient and store
    grad_k *= grad_output
    grad_k_offset = batch_idx * dim
    grad_k_ptrs = grad_k_ptr + grad_k_offset + tl.arange(0, BLOCK_DIM)
    tl.store(grad_k_ptrs, grad_k, mask=dim_mask)


@triton.jit
def _infonce_bwd_queue_kernel(
    q_ptr,  # query embeddings
    queue_ptr,  # negative queue
    grad_queue_ptr,  # gradient w.r.t. queue
    logits_ptr,  # precomputed logits
    grad_output,  # gradient from upstream
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):
    queue_idx = tl.program_id(0)

    # Load queue vector
    queue_offset = queue_idx * dim
    dim_mask = tl.arange(0, BLOCK_DIM) < dim

    grad_queue_vec = tl.zeros((BLOCK_DIM,), dtype=tl.float32)

    # Accumulate gradients from all batch samples
    for batch_idx in range(batch_size):
        # Load query vector
        q_offset = batch_idx * dim
        q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
        q_vec = tl.load(q_ptrs, mask=dim_mask, other=0.0)

        # Load logits for this batch sample
        logits_offset = batch_idx * (queue_size + 1)
        l_pos = tl.load(logits_ptr + logits_offset)

        # Compute softmax probability for this queue item
        max_logit = l_pos
        for i in range(queue_size):
            neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
            max_logit = tl.maximum(max_logit, neg_logit)

        sum_exp = tl.exp(l_pos - max_logit)
        for i in range(queue_size):
            neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
            sum_exp += tl.exp(neg_logit - max_logit)

        # Get probability for this specific queue item
        this_neg_logit = tl.load(logits_ptr + logits_offset + queue_idx + 1)
        prob_neg = tl.exp(this_neg_logit - max_logit) / sum_exp

        # Accumulate gradient: p_i * q / temperature
        grad_queue_vec += prob_neg * q_vec / temperature

    # Scale by upstream gradient and store
    grad_queue_vec *= grad_output
    grad_queue_ptrs = grad_queue_ptr + queue_offset + tl.arange(0, BLOCK_DIM)
    tl.store(grad_queue_ptrs, grad_queue_vec, mask=dim_mask)


class _InfoNCELossWrapper(Function):

    @staticmethod
    def forward(ctx, q, k, queue, temperature):
        batch_size, dim = q.shape
        queue_size, _ = queue.shape

        # Normalize inputs
        q = nn.functional.normalize(q, p=2, dim=1)
        k = nn.functional.normalize(k, p=2, dim=1)
        queue = nn.functional.normalize(queue, p=2, dim=1)

        # Prepare output tensors
        logits = torch.zeros(
            (batch_size, queue_size + 1), device=q.device, dtype=torch.float32
        )
        loss = torch.zeros(1, device=q.device, dtype=torch.float32)

        # Save for backward
        ctx.save_for_backward(q, k, queue, logits)
        ctx.temperature = temperature

        # Launch forward kernel
        BLOCK_DIM = triton.next_power_of_2(dim)
        grid = (batch_size,)

        _infonce_fwd_kernel[grid](
            q_ptr=q,
            k_ptr=k,
            queue_ptr=queue,
            logits_ptr=logits,
            loss_ptr=loss,
            batch_size=batch_size,
            dim=dim,
            queue_size=queue_size,
            temperature=temperature,
            BLOCK_DIM=BLOCK_DIM,
        )

        return loss / batch_size

    @staticmethod
    def backward(ctx, grad_output):
        q, k, queue, logits = ctx.saved_tensors
        temperature = ctx.temperature

        batch_size, dim = q.shape
        queue_size, _ = queue.shape

        # Initialize gradients
        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_queue = torch.zeros_like(queue)

        BLOCK_DIM = triton.next_power_of_2(dim)
        grad_output_scalar = grad_output.item() / batch_size

        # Launch backward kernels
        grid_batch = (batch_size,)
        grid_queue = (queue_size,)

        _infonce_bwd_q_kernel[grid_batch](
            q_ptr=q,
            k_ptr=k,
            queue_ptr=queue,
            grad_q_ptr=grad_q,
            logits_ptr=logits,
            grad_output=grad_output_scalar,
            batch_size=batch_size,
            dim=dim,
            queue_size=queue_size,
            temperature=temperature,
            BLOCK_DIM=BLOCK_DIM,
        )

        _infonce_bwd_k_kernel[grid_batch](
            q_ptr=q,
            k_ptr=k,
            grad_k_ptr=grad_k,
            logits_ptr=logits,
            grad_output=grad_output_scalar,
            batch_size=batch_size,
            dim=dim,
            queue_size=queue_size,
            temperature=temperature,
            BLOCK_DIM=BLOCK_DIM,
        )

        _infonce_bwd_queue_kernel[grid_queue](
            q_ptr=q,
            queue_ptr=queue,
            grad_queue_ptr=grad_queue,
            logits_ptr=logits,
            grad_output=grad_output_scalar,
            batch_size=batch_size,
            dim=dim,
            queue_size=queue_size,
            temperature=temperature,
            BLOCK_DIM=BLOCK_DIM,
        )

        return grad_q, grad_k, grad_queue, None


class InfoNCELossTriton(nn.Module):
    """
    InfoNCE loss implementation using custom Triton kernels.
    """

    def __init__(self, temperature=0.2):
        super().__init__()
        self.temperature = temperature

    def forward(self, q, k, queue):
        return _InfoNCELossWrapper.apply(q, k, queue, self.temperature)


def infonce_loss_pytorch(q, k, queue, temperature=0.2):
    """Reference PyTorch implementation for comparison"""
    l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
    l_neg = torch.mm(q, queue.T)

    logits = torch.cat([l_pos, l_neg], dim=1)
    logits /= temperature

    labels = torch.zeros(logits.shape[0], dtype=torch.long, device=logits.device)
    loss = nn.CrossEntropyLoss()(logits, labels)

    return loss


if __name__ == "__main__":
    import time

    # Test parameters
    batch_size = 512
    dim = 128
    queue_size = 4096
    temperature = 0.2

    # Create test data
    q = torch.randn(batch_size, dim, device="cuda", requires_grad=True)
    k = torch.randn(batch_size, dim, device="cuda", requires_grad=True)
    queue = torch.randn(queue_size, dim, device="cuda", requires_grad=True)

    # Initialize losses
    pytorch_loss_fn = lambda q, k, queue: infonce_loss_pytorch(q, k, queue, temperature)
    triton_loss_fn = InfoNCELossTriton(temperature=temperature)

    # Warmup
    for _ in range(10):
        _ = pytorch_loss_fn(q, k, queue)
        _ = triton_loss_fn(q, k, queue)

    # Benchmark PyTorch
    torch.cuda.synchronize()
    start = time.time()
    print("lmao")
    for _ in range(1):
        loss1 = pytorch_loss_fn(q, k, queue)
        loss1.backward()
        q.grad.zero_()
        k.grad.zero_()
        queue.grad.zero_()
    torch.cuda.synchronize()
    pytorch_time = time.time() - start

    # Benchmark Triton
    torch.cuda.synchronize()
    start = time.time()
    print("lmao")
    for _ in range(1):
        loss2 = triton_loss_fn(q, k, queue)
        loss2.backward()
        q.grad.zero_()
        k.grad.zero_()
        queue.grad.zero_()
        # print("lmfao")
    torch.cuda.synchronize()
    triton_time = time.time() - start

    print(f"PyTorch implementation: {pytorch_time:.4f}s")
    print(f"Triton implementation: {triton_time:.4f}s")
    print(f"Speedup: {pytorch_time / triton_time:.2f}x")

    # Verify correctness
    loss_pytorch = pytorch_loss_fn(q, k, queue)
    loss_triton = triton_loss_fn(q, k, queue)

    print(f"PyTorch loss: {loss_pytorch.item():.6f}")
    print(f"Triton loss: {loss_triton.item():.6f}")
    print(
        f"Relative difference: {abs(loss_pytorch.item() - loss_triton.item()) / loss_pytorch.item():.6f}"
    )
