import torch
import torch.nn as nn
import torch.nn.functional as F


class IJEPALoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, predicted_features, target_features):
        predicted_features = F.normalize(predicted_features, dim=-1)
        target_features = F.normalize(target_features, dim=-1)
        sim = torch.einsum('bnc, bnc -> bn', predicted_features, target_features)
        loss = -sim.mean()
        return loss
    

