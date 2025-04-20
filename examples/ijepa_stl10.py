from torchssl.dataset.ssldataloader import SSLDataloader
from torchssl.dataset.sslaug import IjepaAug
from torchssl.framework.Ijepa import IJEPA
from torchssl.model.backbones import Backbone
import torch
import torch.nn as nn
path_dir = "tests/test_data/mnist/mnist_png/train/9"
ssl_dataloader = SSLDataloader(data_dir=path_dir,augmentation=IjepaAug(img_size=224),batch_size=8,num_workers=3)
train_dl , valid_dl = ssl_dataloader()
device = torch.device("cuda")
model = Backbone("convnext_tiny" , in_channels=1).to(device)
ijepa = IJEPA(context_encoder=model , target_encoder=model , img_size=224 , num_box=6 , device=device , wandb_run=None)
lr = 1e-5
optim = torch.optim.Adam(ijepa.model.parameters() , lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=len(train_dl), eta_min=0,last_epoch=-1)
ijepa.fit(train_loader=train_dl,valid_loader=valid_dl,num_epoch=10,optimizer=optim,scheduler=scheduler,lr=lr)


