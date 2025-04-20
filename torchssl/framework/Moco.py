import torch
import torch.nn as nn
import logging
from torchssl.framework.ssl import SSL
from tqdm import tqdm
from torchssl.loss.python.infonce import infonce_loss
from copy import deepcopy

from torchssl.framework.utils import save_checkpoint


class MocoModel(nn.Module):

    def __init__(
            self,
            encoder_model,
            projection_dim,
            hidden_dim,
            queue_size,
            momentum,

    ):
        super().__init__()

        self.m = momentum

        #query define ->
        self.query_encoder = encoder_model
        encoder_feature_dim = encoder_model.get_features()

        self.query_projector = nn.Sequential(
            nn.Linear(encoder_feature_dim , hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim , projection_dim)
        )


        #key define
        self.key_encoder = deepcopy(encoder_model)
        self.key_projector = nn.Sequential(
            nn.Linear(encoder_feature_dim , hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim , projection_dim)
        )

        #initialize key weights-> same as query + stop gradient in key
        for q_param , k_param in zip(self.query_encoder.parameters() , self.key_encoder.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad_ = False

        for q_param,  k_param in zip(self.query_projector.parameters() , self.key_projector.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad_ = False
        

        self.queue_size = queue_size
        self.register_buffer("queue" , torch.randn(queue_size , projection_dim))
        self.queue = nn.functional.normalize(self.queue, dim=1)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))



    def forward(self , img_q , img_k):

        q_encoder_out = self.query_encoder(img_q)
        q_projector_out = self.query_projector(q_encoder_out)
        #normalize across channels
        q_out = nn.functional.normalize(q_projector_out , dim=1)


        with torch.no_grad():

            k_encoder_out = self.key_encoder(img_k)
            k_projector_out = self.key_projector(k_encoder_out)
            k_out = nn.functional.normalize(k_projector_out , dim=1)


        return q_out , k_out
    
    @torch.no_grad()
    def updated_key(self):

        for q_param , k_param in zip(self.query_encoder.parameters() , self.key_encoder.parameters()):
            k_param.data = k_param.data * self.m + q_param.data * (1 - self.m)

        for q_param , k_param in zip(self.query_projector.parameters() , self.key_projector.parameters()):
            k_param.data = k_param.data * self.m + q_param.data * (1 - self.m)

    @torch.no_grad()
    def dequeue_enqueue(self , keys):

        bs = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + bs > self.queue_size:
            overflow = (ptr + bs) - self.queue_size
            self.queue[ptr : self.queue_size] = keys[:(bs - overflow)]
            self.queue[0 : overflow] = keys[(bs - overflow):]
            self.queue_ptr[0] = overflow

        else:
            self.queue[ptr : ptr + bs] = keys
            self.queue_ptr[0] = (ptr + bs) % self.queue_size


class MoCO(SSL):
    def __init__(
            self,
            encoder_model,
            projection_dim,
            hidden_dim,
            queue_size,
            momentum,
            device,
            wandb_run,
    ):
        super().__init__(device=device , wandb_run=wandb_run)
        self.model = MocoModel(
            encoder_model=encoder_model,
            projection_dim=projection_dim,
            hidden_dim=hidden_dim,
            queue_size=queue_size,
            momentum=momentum
        )

    
    def train_one_epoch(
            self,
            dataloader,
            optimizer,
            scheduler,
            temperature,
            epoch,
    ):
        
        self.model.train()
        running_loss = 0.0

        for i , (img_q , img_k) in tqdm(enumerate(dataloader)):

            img_q = img_q.to(self.device)
            img_k = img_k.to(self.device)

            optimizer.zero_grad()


            q_out , k_out = self.model(img_q , img_k)
            loss = infonce_loss(
                q=q_out,
                k=k_out,
                temperature=temperature
            )
            loss.backward()
            optimizer.step()


            self.model.updated_key()

            with torch.no_grad():
                self.model.dequeue_enqueue(k_out)

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
                

    def validate(
            self,
            dataloader,
            temperature,
            epoch
    ):
        
        self.model.eval()
        running_loss = 0.0
        with torch.no_grad():
            for i, (im_q, im_k) in enumerate(dataloader):
                img_q = im_q.to(self.device)
                img_k = im_k.to(self.device)
                q_out , k_out = self.model(img_q , img_k)
                loss = infonce_loss(
                    q=q_out,
                    k=k_out,
                    temperature=temperature
                )
                running_loss += loss.item()
        avg_loss = running_loss / len(dataloader)
        logging.info(f"Epoch [{epoch+1}] Validation Loss: {avg_loss:.4f}")

        if self.wandb_run:
            self.wandb_run.log({"val_loss": avg_loss, "epoch": epoch+1})
        return avg_loss
        

    def get_temperature(self , epoch, max_epochs, initial_temp=0.5, final_temp=0.1):
        progress = epoch / max_epochs
        return initial_temp - progress * (initial_temp - final_temp)
    

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


        for epoch in tqdm(range(num_epoch)):
            logging.info(f"--- Epoch {epoch+1}/{num_epoch} ---")

            if warmup_scheduler_epoch > 0:
                if epoch < warmup_scheduler_epoch:
                    progress = (epoch + 1) / warmup_scheduler_epoch
                    lr = lr + progress * (initial_lr - lr_min)

                    for param_grp in optimizer.param_groups:
                        param_grp["lr"] = lr

                    logging.info(f"Warm-up phase: LR set to {lr:.6f}")
                    if self.wandb_run:
                        self.wandb_run.log({"learning_rate": lr, "epoch": epoch+1})


            temp = self.get_temperature(epoch=epoch , max_epochs=num_epoch)
            train_loss = self.train_one_epoch(
                dataloader=train_dataloader,
                optimizer=optimizer,
                scheduler=scheduler,
                temperature=temp,
                epoch=epoch
            )

            valid_loss = self.validate(
                dataloader=valid_dataloader,
                temperature=temp,
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
                        'queue': self.model.queue.clone(),
                        'queue_ptr': self.model.queue_ptr.clone(), 
                        'train_loss': train_loss,
                        'val_loss': valid_loss,
                    }
                    epoch_ckpt = f"checkpoint_epoch_{epoch+1}.pth"
                    save_checkpoint(checkpoint_state, checkpoint_dir, epoch_ckpt)


            if (epoch + 1) % evaluation_epoch == 0:
                self.linear_probe_evaluation()
                self.knn_evaluation()


