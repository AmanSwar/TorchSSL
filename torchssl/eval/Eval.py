import torch
import torch.nn as nn
import logging
from tqdm import tqdm
import numpy as np
from collections import Counter
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from wandb import Image

class EvaluateSSL:

    def __init__(
            self,
            model,
            train_loader,
            valid_loader,
            device,
            wandb_run

    ):
        self.model = model
        self.train_loader = train_loader
        self.valid_loader =valid_loader
        self.device = device
        self.wandb_run = wandb_run


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
            k : int = 10
    ):
        """
        K-nearest neighbor evaluation
        Args:
            k -> should be equals to number of classes (default = 10 (stl10))
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
    
    def _plot_pca_visual(
            self,
            features,
            labels,
            class_labels=None
    ):
        pca = PCA(n_components=2)
        pca_out = pca.fit_transform(features)


        fig ,ax = plt.subplot(figsize=(10,8))

        unique_labels = np.unique(labels)
    
        for i, label in enumerate(unique_labels):
            indices = labels == label
            scatter = ax.scatter(pca_out[indices, 0], pca_out[indices, 1], 
                    alpha=0.6, s=20, label=class_labels[i] if class_labels else f"Class {label}")
            
        ax.legend()
        ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)')
        ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)')
        ax.set_title('PCA visualization of features')
        
        plt.tight_layout()
        return fig



    def _plot_tsne_visual(
            self,
            features,
            labels,
            class_labels=None
    ):
        tsne = TSNE(
            n_components=2 ,
            random_state=6969,
            perplexity=min(30 , len(features)-1)
        )

        tsne_result = tsne.fit_transform(features)

        fig, ax = plt.subplots(figsize=(10, 8))
    
        unique_labels = np.unique(labels)
        
        for i, label in enumerate(unique_labels):
            indices = labels == label
            scatter = ax.scatter(tsne_result[indices, 0], tsne_result[indices, 1], 
                    alpha=0.6, s=20, label=class_labels[i] if class_labels else f"Class {label}")
        
        ax.legend()
        ax.set_xlabel('t-SNE dimension 1')
        ax.set_ylabel('t-SNE dimension 2')
        ax.set_title('t-SNE visualization of features')
        
        plt.tight_layout()
        return fig


    def _compute_feature_sim(
            self,
            features,
            sample_size=1000
    ):
        features_norm = features / np.linalg.norm(features, axis=1, keepdims=True)
    
        n = features.shape[0]
        if n > sample_size:
            indices = np.random.choice(n, size=sample_size, replace=False)
            features_sample = features_norm[indices]
        else:
            features_sample = features_norm
        
        similarity_matrix = np.matmul(features_sample, features_sample.T)
        
        mask = np.ones_like(similarity_matrix, dtype=bool)
        np.fill_diagonal(mask, False)
        
        return similarity_matrix[mask]
    


    def monitor_feature_representation(
            self,
            epoch,
            num_samples=1000,
            plot_pca=True,
            plot_tsne=True,
            class_labels=None
    ):
        
        self.model.eval()

        all_features = []
        all_labels = []
        sample_count = 0

        with torch.no_grad():
            for images, labels in self.valid_loader:
                if sample_count >= num_samples:
                    break
            
            current_batch_size = images.size(0)
            if sample_count + current_batch_size > num_samples:
                images = images[:num_samples - sample_count]
                labels = labels[:num_samples - sample_count]

            images = images.to(self.device)


            features = self.model(images)
            all_features.append(features.cpu().numpy())
            all_labels.append(labels.numpy())
            
            sample_count += images.size(0)

        features = np.vstack(all_features)
        labels = np.concatenate(all_labels)


        if plot_pca:
            pca_fig = self._plot_pca_visual(features, labels, class_labels)
            if self.wandb_run:
                self.wandb_run.log({"feature_diversity/pca": Image(pca_fig), "epoch": epoch})
            plt.close(pca_fig)
        
        if plot_tsne:
            tsne_fig = self._plot_tsne_visual(features, labels, class_labels)
            if self.wandb_run:
                self.wandb_run.log({"feature_diversity/tsne": Image(tsne_fig), "epoch": epoch})
            plt.close(tsne_fig)


        feature_norm = np.linalg.norm(features, axis=1)
        feature_similarity = self._compute_feature_sim(features)
        
        if self.wandb_run:
            self.wandb_run.log({
                "feature_diversity/mean_norm": np.mean(feature_norm),
                "feature_diversity/std_norm": np.std(feature_norm),
                "feature_diversity/mean_similarity": np.mean(feature_similarity),
                "feature_diversity/std_similarity": np.std(feature_similarity),
                "epoch": epoch
            })
