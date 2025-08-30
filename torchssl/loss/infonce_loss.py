import triton
import triton.language as tl
import torch
import torch.nn as nn
from torch.autograd.function import Function


@triton.jit
def _infonce_fwd_kernel(
    q_ptr, # query embeddings ptr -> [batch_size, dim]
    k_ptr, # key embeddings  ptr -> [batch_size, dim] 
    queue_ptr, # negative queue ptr -> [queue_size, dim]
    logits_ptr, # output logits ptr ->  [batch_size, queue_size + 1]
    loss_ptr, # loss ptr
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):

    # in this kernel -> each program handles once sample in the batch
    batch_idx = tl.program_id(0)

    q_offset = batch_idx * dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
    q_mask = tl.arange(0, BLOCK_DIM) < dim

    q_vec = tl.load(q_ptrs, mask=q_mask, other=0.0)

    k_offset = batch_idx * dim
    k_ptrs = k_ptr + k_offset + tl.arange(0, BLOCK_DIM)

    k_vec = tl.load(k_ptrs, mask=q_mask, other=0.0)

    l_pos = tl.sum(q_vec * k_vec) / temperature

    logits_offset = batch_idx * (queue_size + 1)

    tl.store(logits_ptr + logits_offset, l_pos)

    for neg_idx in range(queue_size):
        neg_offset = neg_idx * dim
        neg_ptrs = queue_ptr + neg_offset + tl.arange(0, BLOCK_DIM)
        neg_vec = tl.load(neg_ptrs, mask=q_mask, other=0.0)

        l_neg = tl.sum(q_vec * neg_vec) / temperature
        tl.store(logits_ptr + logits_offset + neg_idx + 1, l_neg)

    max_logit = l_pos
    for neg_idx in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + neg_idx + 1)
        max_logit = tl.maximum(max_logit, neg_logit)

    sum_exp = tl.exp(l_pos - max_logit)
    for neg_idx in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + neg_idx + 1)
        sum_exp += tl.exp(neg_logit - max_logit)

    log_sum_exp = tl.log(sum_exp) + max_logit
    sample_loss = -l_pos + log_sum_exp

    tl.atomic_add(loss_ptr, sample_loss)


@triton.jit
def _infonce_bwd_q_kernel(
    q_ptr, 
    k_ptr,
    queue_ptr,
    grad_q_ptr,   
    logits_ptr,
    grad_output,    
    batch_size: tl.constexpr,
    dim: tl.constexpr,
    queue_size: tl.constexpr,
    temperature,
    BLOCK_DIM: tl.constexpr,
):
    batch_idx = tl.program_id(0)

    logits_offset = batch_idx * (queue_size + 1)
    logits = tl.zeros(queue_size + 1, dtype=tl.float32)
    for i in range(queue_size + 1):
        logits += tl.load(logits_ptr + logits_offset + i) * (
            i == tl.arange(0, queue_size + 1)
        )

    max_logit = tl.max(logits)
    exp_logits = tl.exp(logits - max_logit)
    sum_exp = tl.sum(exp_logits)
    probs = exp_logits / sum_exp

    grad_q = tl.zeros(BLOCK_DIM, dtype=tl.float32)
    dim_mask = tl.arange(0, BLOCK_DIM) < dim

    k_offset = batch_idx * dim
    k_ptrs = k_ptr + k_offset + tl.arange(0, BLOCK_DIM)
    k_vec = tl.load(k_ptrs, mask=dim_mask, other=0.0)

    prob_pos = tl.load(
        logits_ptr + logits_offset
    ) 
    grad_q += (probs[0] - 1.0) * k_vec / temperature

    for neg_idx in range(queue_size):
        neg_offset = neg_idx * dim
        neg_ptrs = queue_ptr + neg_offset + tl.arange(0, BLOCK_DIM)
        neg_vec = tl.load(neg_ptrs, mask=dim_mask, other=0.0)

        grad_q += probs[neg_idx + 1] * neg_vec / temperature

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

    q_offset = batch_idx * dim
    q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
    dim_mask = tl.arange(0, BLOCK_DIM) < dim
    q_vec = tl.load(q_ptrs, mask=dim_mask, other=0.0)

    logits_offset = batch_idx * (queue_size + 1)
    l_pos = tl.load(logits_ptr + logits_offset)

    max_logit = l_pos
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        max_logit = tl.maximum(max_logit, neg_logit)

    sum_exp = tl.exp(l_pos - max_logit)
    for i in range(queue_size):
        neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
        sum_exp += tl.exp(neg_logit - max_logit)

    prob_pos = tl.exp(l_pos - max_logit) / sum_exp
    grad_k = (prob_pos - 1.0) * q_vec / temperature

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

    queue_offset = queue_idx * dim
    queue_ptrs = queue_ptr + queue_offset + tl.arange(0, BLOCK_DIM)
    dim_mask = tl.arange(0, BLOCK_DIM) < dim

    grad_queue_vec = tl.zeros(BLOCK_DIM, dtype=tl.float32)

    for batch_idx in range(batch_size):
        q_offset = batch_idx * dim
        q_ptrs = q_ptr + q_offset + tl.arange(0, BLOCK_DIM)
        q_vec = tl.load(q_ptrs, mask=dim_mask, other=0.0)

        logits_offset = batch_idx * (queue_size + 1)
        l_pos = tl.load(logits_ptr + logits_offset)

        max_logit = l_pos
        for i in range(queue_size):
            neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
            max_logit = tl.maximum(max_logit, neg_logit)

        sum_exp = tl.exp(l_pos - max_logit)
        for i in range(queue_size):
            neg_logit = tl.load(logits_ptr + logits_offset + i + 1)
            sum_exp += tl.exp(neg_logit - max_logit)

        this_neg_logit = tl.load(logits_ptr + logits_offset + queue_idx + 1)
        prob_neg = tl.exp(this_neg_logit - max_logit) / sum_exp

        grad_queue_vec += prob_neg * q_vec / temperature

    grad_queue_vec *= grad_output
    grad_queue_ptrs = grad_queue_ptr + queue_offset + tl.arange(0, BLOCK_DIM)
    tl.store(grad_queue_ptrs, grad_queue_vec, mask=dim_mask)


class _InfoNCELossWrapper(Function):

    @staticmethod
    def forward(ctx, q, k, queue, temperature):
        batch_size, dim = q.shape
        queue_size, _ = queue.shape

        q = nn.functional.normalize(q, p=2, dim=1)
        k = nn.functional.normalize(k, p=2, dim=1)
        queue = nn.functional.normalize(queue, p=2, dim=1)

        logits = torch.zeros(
            (batch_size, queue_size + 1), device=q.device, dtype=torch.float32
        )
        loss = torch.zeros(1, device=q.device, dtype=torch.float32)

        ctx.save_for_backward(q, k, queue, logits)
        ctx.temperature = temperature

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

        grad_q = torch.zeros_like(q)
        grad_k = torch.zeros_like(k)
        grad_queue = torch.zeros_like(queue)

        BLOCK_DIM = triton.next_power_of_2(dim)
        grad_output_scalar = grad_output.item() / batch_size

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
