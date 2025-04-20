import torch
import torch.nn as nn
from torchssl.framework.ssl import SSL
from torchssl.loss.python.ntxent import NTXentLoss
import logging
from torchssl.framework.utils import save_checkpoint

class SimclrMLP(nn.Module):
    def __init__(
            self,
            backbone_feat,
            hidden_dim,
            projection_dim
    ):
        
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=backbone_feat , out_features=hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_dim , out_features=projection_dim)
        )

    def forward(self , x):

        out = self.mlp(x)

        return out



class SimclrModel(nn.Module):

    def __init__(
            self,
            backbone_model,
            projector_head
    ):
        super().__init__()
        self.backbone_model = backbone_model
        backbone_feat = self.backbone_model.get_features()
        self.projection_dim = projector_head

        self.projector_head = projector_head
        
    def forward(self , x):
        out = self.projector_head(self.backbone_model(x))
        return out
    


class SimCLR(SSL):

    def __init__(
            self,
            backbone_model,
            hidden_dim,
            projection_dim,
            temperature,
            device,
            wandb_run= None

    ):
        super().__init__(device=device , wandb_run=wandb_run)
        self.wandb_run = wandb_run
        self.temp = temperature
        self.device = device

        self.loss_fn = NTXentLoss(
            temp=self.temp,
            device=self.device
        )

        self.backbone_model = backbone_model
        backbone_feat = self.backbone_model.get_features()

        self.projector_head = SimclrMLP(
            backbone_feat=backbone_feat,
            hidden_dim=hidden_dim,
            projection_dim=projection_dim
        )

        self.model = SimclrModel(
            backbone_model=backbone_model,
            projector_head=self.projector_head
        )



    def train_one_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
            epoch,
            ):
        
        self.model.train()
        running_loss = 0.0

        for i , (x1 , x2) in enumerate(dataloader):
            x1 = x1.to(self.device)
            x2 = x2.to(self.device)
            
            optimizer.zero_grad()

            #x1 pass
            # z1 = self.mlp(self.backbone_model(x1))
            z1 = self.model(x1)
            #x2 pass
            # z2 = self.mlp(self.backbone_model(x2))
            z2 = self.model(x2)
            loss = self.loss_fn(z1 , z2)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if i % 10 == 0:
                current_lr = optimizer.param_groups[0]['lr']
                logging.info(f"Epoch [{epoch+1}] Step [{i}/{len(dataloader)}] Loss: {loss.item():.4f} LR: {current_lr:.6f}")

                if self.wandb_run:
                    self.wandb_run.log({
                        "train_loss": loss.item(), 
                        "epoch": epoch+1,
                        "learning_rate": current_lr
                    })
        scheduler.step()

        avg_loss = running_loss / len(dataloader)

        return avg_loss
    

    def validate(self,
                dataloader,
                epoch,
                ):
        
        self.model.eval()

        running_loss = 0

        with torch.no_grad():

            for i , (x1 , x2) in enumerate(dataloader):
                x1 = x1.to(self.device)
                x2 = x2.to(self.device)
                #x1 pass
                # z1 = self.mlp(self.backbone_model(x1))
                z1 = self.model(x1)
                #x2 pass
                # z2 = self.mlp(self.backbone_model(x2))
                z2 = self.model(x2)

                loss = self.loss_fn(z1 , z2)
                running_loss += loss.item()

        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")

        if self.wandb_run:
            self.wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
            
        return avg_loss

    def fit(
            self,
            train_dataloader,
            valid_dataloader,
            num_epoch,
            optimizer,
            lr,
            scheduler,
            evaluation_epoch = 5,
            save_checkpoint_epoch: int = 0,
            checkpoint_dir : str = None,
            mixed_precision=False,
            warmup_scheduler_epoch=0,
            lr_min = 0,

    ):
        
        train_loss = 0
        valid_loss = 0
        
        if warmup_scheduler_epoch > 0:
            initial_lr = lr
            for param_group in optimizer.param_groups:
                param_group["lr"] = lr_min if lr_min != 0 else lr

        for epoch in range(num_epoch):
            logging.info(f"--- Epoch {epoch+1}/{num_epoch} ---")

            # wanrmup scheduler logic
            if warmup_scheduler_epoch > 0:
                if epoch < warmup_scheduler_epoch:
                    progress = (epoch + 1) / warmup_scheduler_epoch
                    lr = lr + progress * (initial_lr - lr_min)

                    for param_grp in optimizer.param_groups:
                        param_grp["lr"] = lr

                    logging.info(f"Warm-up phase: LR set to {lr:.6f}")
                    if self.wandb_run:
                        self.wandb_run.log({"learning_rate": lr, "epoch": epoch+1})


            train_loss = self.train_one_epoch(
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

            valid_loss = self.validate(
                dataloader=valid_dataloader,
                epoch=epoch
            )

            if save_checkpoint_epoch > 0:
                assert checkpoint_dir != None , "checkpoint directory is not given"

                if (epoch + 1) % save_checkpoint_epoch == 0:

                    checkpoint_state = {
                        'epoch': epoch + 1,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'scheduler_state_dict': scheduler.state_dict(),  
                        'train_loss': train_loss,
                        'val_loss': valid_loss,
                    }
                    epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
                    save_checkpoint(checkpoint_state, checkpoint_dir, epoch_ckpt)


            if (epoch + 1) % evaluation_epoch == 0:
                self.linear_probe_evaluation()
                self.knn_evaluation()
