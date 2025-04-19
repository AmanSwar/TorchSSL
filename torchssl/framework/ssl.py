import torch
import torch.nn as nn
from tqdm import tqdm
import logging


class SSL:

    def __init__(
            self,
            device,
            wandb_run,
    ):
        self.device = device
        self.wandb_run = wandb_run

    def train_one_epoch(
            self
    ):
        pass

    def validate(self):
        pass

    def fit(self):
        pass
    
    def extract_features(
            dataloader
    ):
        pass
    
    

