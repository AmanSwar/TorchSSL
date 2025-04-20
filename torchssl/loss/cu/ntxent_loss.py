import os
import torch
from torch import nn
from torch.utils.cpp_extension import load
from torchssl.loss.python.ntxent import NTXentLoss
current_dir = os.path.dirname(os.path.abspath(__file__))
ntxent_cuda = load(
    name="ntxent_cuda",
    sources=["torchssl/loss/cu/ntxent_loss.cu"],
    verbose=True
)


class FusedNTXentLoss(nn.Module):
    def __init__(self, temp=0.5, device="cuda"):
        super().__init__()
        self.temp = temp
        self.device = device
        
    def forward(self, z_i, z_j):
        
        # Make sure inputs are on CUDA
        z_i = z_i.to(self.device)
        z_j = z_j.to(self.device)
        
        # Call the CUDA implementation
        loss = ntxent_cuda.ntxent_loss(z_i, z_j, self.temp)
        
        return loss

if __name__ == "__main__":
    # Example usage
    batch_size = 128
    feature_dim = 128


    z_i = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)
    z_j = torch.randn(batch_size, feature_dim, device="cuda", requires_grad=True)

    original_loss = NTXentLoss(temp=0.5, device="cuda")

    cuda_loss = FusedNTXentLoss(temp=0.5, device="cuda")
    
    # Compare performance
    import time
    
    # Warmup
    for _ in range(10):
        _ = original_loss(z_i, z_j)
        _ = cuda_loss(z_i, z_j)
    

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
        loss2 = cuda_loss(z_i, z_j)
        loss2.backward()
    torch.cuda.synchronize()
    cuda_time = time.time() - start
    
    print(f"Original implementation: {original_time:.4f}s")
    print(f"CUDA implementation: {cuda_time:.4f}s")
    print(f"Speedup: {original_time / cuda_time:.2f}x")

    print(f"Original loss: {original_loss(z_i, z_j).item():.6f}")
    print(f"CUDA loss: {cuda_loss(z_i, z_j).item():.6f}")