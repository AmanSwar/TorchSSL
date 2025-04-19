import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import numpy as np
from collections import Counter

class EvaluateSSL:

    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            device

    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader =valid_loader
        self.device = device


    def extract_features(self, dataloader):

        self.model.eval()

        all_features = []
        all_labels = []
        logging.info("Extracting features ...")
        for _ , img , label in tqdm(enumerate(dataloader)):

            img = img.to(self.device)
            label = label.to(self.device)

            feats = self.model(img)
            all_features.append(feats.cpu())
            all_labels.append(label)

        all_features = torch.cat(all_features , dim=0)
        all_labels = torch.can_cast(all_labels , dim=0)

        return all_features , all_labels

    def linear_probe_evaluation(
            self,
            probe_epoch=5,
            
    ):
        logging.info("\nStarting Linear proble evaluation ")
        train_feats ,train_labels = self.extract_features(self.train_loader)
        valid_feats ,valid_labels = self.extract_features(self.valid_loader)

        embed_dim = train_feats.shape[1]
        batch_size = train_feats.shape[0]
        num_classes = len(train_labels.unique())

        linear_probe = nn.Sequential(
            nn.Linear(embed_dim , num_classes).to(self.device)
        )
        optimizer = torch.optim.Adam(linear_probe.parameters(), lr=1e-3)
        criterion = nn.CrossEntropyLoss()
        
        logging.info(f"Training Linear Probe ....")
        for _ , epoch in tqdm(enumerate(range(probe_epoch))):
            linear_probe.train()
            
            perm = torch.randperm(train_feats.size(0))
            train_feats_shuffle = train_feats[perm].to(self.device)
            train_labels_shuffle = train_labels[perm].to(self.device)

            for i in range(0 , train_feats_shuffle.size(0) , batch_size):
                end = i + batch_size
                batch_feats = train_feats_shuffle[i : end]
                batch_labels = train_labels_shuffle[i : end]

                optimizer.zero_grad()
                out = linear_probe(batch_feats)
                loss = criterion(out , batch_labels)

                loss.backward()
                optimizer.step()

                logging.info(f"Epoch : {i} | Loss : {loss.item():.4f}")


        linear_probe.eval()

        valid_feats = valid_feats.to(self.device)

        with torch.no_grad():
            out = linear_probe(valid_feats)
            pred = torch.argmax(out , dim=1).cpu()

            acc = (pred == valid_labels).float().mean().item() * 100

        logging.info(f"[Linear Probe] Validation Accuracy: {acc:.2f}%")
        if self.wandb_run:
            self.wandb_run.log({"linear_probe_accuracy": acc})
        return acc


    def knn_evaluation(
            self,
            k : int
    ):
        """
        K-nearest neighbor evaluation
        Args:
            k -> should be equals to number of classes
        """
        logging.info("\nStarting KNN evaluation ")
        train_feats ,train_labels = self.extract_features(self.train_loader)
        valid_feats ,valid_labels = self.extract_features(self.valid_loader)

        train_feats_np = train_feats.numpy()
        train_labels_np = train_labels.numpy()
        val_feats_np = valid_feats.numpy()
        val_labels_np = valid_labels.numpy()

        correct = 0

        for i  in tqdm(range(len(val_feats_np))):
            diff = train_feats_np - val_feats_np[i]
            dist = np.sum(diff * 2 , axis=1)
            idx = np.argsort(dist)[:k]              
            neighbors = train_labels_np[idx] 

            majority = Counter(neighbors).most_common(1)[0][0]
            if majority == val_feats_np[i]:
                correct += 1


        acc = 100 * correct / len(val_feats_np)

        logging.info(f"[k-NN (k={k})] Validation Accuracy: {acc:.2f}%")
        if self.wandb_run is not None:
            self.wandb_run.log({"knn_accuracy": acc})
        return acc
            
 