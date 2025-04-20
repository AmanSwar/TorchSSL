from torchssl.dataset.ssldataloader import SSLDataloader
from torchssl.dataset.sslaug import MocoAug
from torchssl.framework.Moco import MoCO
from torchssl.model.backbones import Backbone
import torch
import torch.nn as nn
path_dir = "tests/test_data/mnist/mnist_png/train/9"
ssl_dataloader = SSLDataloader(data_dir=path_dir,augmentation=MocoAug(img_size=224),batch_size=8,num_workers=3)
train_dl , valid_dl = ssl_dataloader()
device = torch.device("cuda")
model = Backbone("convnext_tiny" , in_channels=1).to(device)
moco = MoCO(encoder_model=model , projection_dim=128 , hidden_dim=512 , queue_size=1024 , momentum=0.99 , device=device , wandb_run=None)
lr = 1e-5

optim = torch.optim.Adam(moco.model.parameters() , lr=lr)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optim, T_max=len(train_dl), eta_min=0,last_epoch=-1)
moco.fit(train_dataloader=train_dl,valid_dataloader=valid_dl,num_epoch=10,optimizer=optim,scheduler=scheduler,lr=lr)


