from torchssl.dataset.ssldataloader import SSLDataloader
from torchssl.dataset.sslaug import SimclrAug
from torchssl.framework.SimCLR import SimCLR
from torchssl.model.backbones import Backbone
import torch
import torch.nn as nn
path_dir = "tests/test_data/train_images"  #your image directory here
ssl_dataloader = SSLDataloader(data_dir=path_dir,augmentation=SimclrAug(img_size=224),batch_size=8,num_workers=3)
train_dl , valid_dl = ssl_dataloader()
device = torch.device("cuda")
model = Backbone("convnext_tiny" , in_channels=1).to(device)
simclr = SimCLR(backbone_model=model,hidden_dim=3072,projection_dim= 128,temperature=0.5,)
lr = 1e-5
optim = torch.optim.Adam(simclr.model.parameters() , lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=len(train_dl), eta_min=0,last_epoch=-1)
simclr.fit(train_dataloader=train_dl,valid_dataloader=valid_dl,num_epoch=10,optimizer=optim,scheduler=scheduler,lr=lr)


