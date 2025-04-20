import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torchssl.framework.ssl import SSL
from tqdm import tqdm
from torchssl.loss.python.ijepaloss import IJEPALoss
from einops import rearrange
import wandb

from torchssl.framework.utils import save_checkpoint
class IjepaModel(nn.Module):

    def __init__(
            self,
            context_encoder,
            target_encoder,
            img_size,
            num_box,
            
    ):
        super().__init__()


        self.context_encoder = context_encoder
        self.target_encoder = target_encoder

        self.encoder_out_dim = context_encoder.get_features()

        self.predictor = nn.Sequential(
            nn.Conv2d(self.encoder_out_dim, self.encoder_out_dim, kernel_size=1),
            nn.BatchNorm2d(self.encoder_out_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.encoder_out_dim, self.encoder_out_dim, kernel_size=1)
        )

        self.n_box = num_box
        self.img_size = img_size

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)


    @torch.no_grad()
    def momentum_update(self, momentum=0.999):
        for t_param, c_param in zip(
                self.target_encoder.parameters(), 
                self.context_encoder.parameters()):
            t_param.data.mul_(momentum).add_(c_param.data, alpha=1-momentum)

    def get_random_boxes(
            self,
            batch_size
    ):
      
        boxes = []
        for _ in range(batch_size):
            batch_boxes = []
            for _ in range(self.n_box):
                x1 = torch.randint(0, 14, (1,)).item()
                y1 = torch.randint(0, 14, (1,)).item()
                w = torch.randint(2, 6, (1,)).item()
                h = torch.randint(2, 6, (1,)).item()
                batch_boxes.append([x1, y1, x1 + w, y1 + h])
            boxes.append(batch_boxes)
        return torch.tensor(boxes)

    def extract_target(self, feature, boxes):
    
        B, C, H, W = feature.shape
        target_features = []
        
        #batch loop

        for b in range(B):
            
            batch_targets = []
            
            for box in boxes[b]:
                x1, y1, x2, y2 = box
                target = feature[b:b+1, :, y1:y2, x1:x2]
                target = F.adaptive_avg_pool2d(target, (1, 1))
                batch_targets.append(target)
            batch_targets = torch.cat(batch_targets, dim=0)
            batch_targets = batch_targets.view(-1, C)
            target_features.append(batch_targets)
        target_features = torch.stack(target_features)
        return target_features
    
    def forward(self, images, boxes=None):
        B = images.shape[0]
        
      
        context_feats = self.context_encoder.forward_features(images)  # (B, C, H, W)
        H = context_feats.shape[2]

        if boxes is None:
            boxes = self.get_random_boxes(B, H)

        #context_feats -> prediction head
        pred_feats = self.predictor(context_feats)
        pred_feats = self.extract_target(pred_feats, boxes)

        # Compute target features without gradient
        with torch.no_grad():
            target_feats = self.target_encoder.forward_features(images)
            target_feats = self.extract_target(target_feats, boxes)

        return pred_feats, target_feats


class IJEPA(SSL):

    def __init__(
            self, 
            context_encoder,
            target_encoder,
            img_size,
            num_box,
            device, 
            wandb_run
            ):
        super().__init__(device, wandb_run)
        self.wandb_run = wandb_run
        self.device = device


        self.model = IjepaModel(
            context_encoder=context_encoder,
            target_encoder=target_encoder,
            img_size=img_size,
            num_box=num_box
        ).to(self.device)

        self.loss_fn = IJEPALoss()

        

    def train_one_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
            epoch
            ):
        
        self.model.train()

        n_batch = len(dataloader)

        total_loss = 0
        pbar = tqdm(total=n_batch, desc=f"Epoch: {epoch}")

        for batch_idx, batch in enumerate(dataloader):
            img = batch.to(self.device)
            optimizer.zero_grad()

            pred_feat, target_feat = self.model(img)
            loss = self.loss_fn(pred_feat, target_feat)
            loss.backward()
            optimizer.step()

            self.model.momentum_update()

            total_loss += loss.item()

            if self.wandb_run:
                self.wandb.log({
                    'batch_loss': loss.item(),
                    'epoch': epoch,
                    'batch': batch_idx
                })

            logging.info(f"BATCH LOSS : {loss}")
            pbar.update()

        scheduler.step()

        pbar.close()
        avg_loss = total_loss / n_batch
        return avg_loss


    def validate(
            self,
            dataloader,
            epoch
    ):
        self.model.eval()
        n_batch = len(dataloader)

        total_loss = 0
        pbar = tqdm(total=n_batch, desc=f"Epoch: {epoch}")

        for batch_idx, batch in enumerate(dataloader):
            img = batch.to(self.device)
  
            pred_feat, target_feat = self.model(img)
            loss = self.loss_fn(pred_feat, target_feat)
            total_loss += loss.item()
            pbar.update()


        pbar.close()
        avg_val_loss = total_loss / n_batch
        if self.wandb_run:
            wandb.log({'val_loss': avg_val_loss, 'epoch': epoch})
        logging.info(f"Validation Epoch {epoch}: Loss = {avg_val_loss:.4f}")
        return avg_val_loss
    

    def fit(
            self,
            train_loader,
            valid_loader,
            num_epoch,
            optimizer,
            scheduler,
            lr,
            evaluation_epoch = 5,
            save_checkpoint_epoch: int = 0,
            checkpoint_dir : str = None,

    ):
        train_loss = 0
        valid_loss = 0

        for epoch in range(num_epoch):
            logging.info(f"--- Epoch {epoch+1}/{num_epoch} ---")

            train_loss = self.train_one_epoch(
                dataloader=train_loader,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch
            )

            valid_loss = self.validate(
                dataloader=valid_loader,
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