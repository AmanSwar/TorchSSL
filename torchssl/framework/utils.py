import torch
import torch.nn as nn


class NTXentLoss(nn.Module):

    def __init__(self , batch_size ,temperature=0.5 , device="cuda"):

        super(NTXentLoss , self).__init__()
        self.batch_size = batch_size
        self.temperature = temperature
        self.device = device