from torchssl.dataset.ssldataloader import SSLDataloader
from torchssl.dataset.sslaug import DinoAug
from torchssl.framework.Dino import Dino
from torchssl.model.backbones import Backbone
import torch
import torch.nn as nn
path_dir = "tests/test_data/mnist/mnist_png/train/9"
ssl_dataloader = SSLDataloader(data_dir=path_dir,augmentation=DinoAug(),batch_size=8,num_workers=3)
train_dl , valid_dl = ssl_dataloader()
device = torch.device("cuda")
model = Backbone("convnext_tiny" , in_channels=1).to(device)
dino = Dino(backbone_model=model , projection_dim=128 , hidden_dim=512 , bottleneck_dim=1024 , teacher_temp=0.04 , student_temp=0.1 , ncrops=8, device=device , wandb_run=None)
lr = 1e-5
optim = torch.optim.Adam(dino.model.parameters() , lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=len(train_dl), eta_min=0,last_epoch=-1)
dino.fit(train_loader=train_dl,valid_loader=valid_dl,num_epochs=10,optimizer=optim,scheduler=scheduler,lr=lr)


