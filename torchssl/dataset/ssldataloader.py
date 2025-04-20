import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset , DataLoader
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import random_split


class SSLDataset(Dataset):
    
    def __init__(self , data_dir , augmentation):
        super().__init__()
        
        self.aug = augmentation
        self.tensor_aug = transforms.ToTensor()
        self.IMAGE_PATH = []    
        for img in tqdm(os.listdir(data_dir)):
            self.IMAGE_PATH.append(os.path.join(data_dir , img))

        

    def __len__(self):
        return len(self.IMAGE_PATH)


    def __getitem__(self, index):
        img_path = self.IMAGE_PATH[index]
        img = Image.open(img_path)

        if self.aug:
            return self.aug(img)
        
        return self.tensor_aug(img)
    
class SSLDataloader(object):
    def __init__(self, 
                 data_dir,
                 augmentation,
                 batch_size,
                 num_workers,
                 val_split=0.2,  
                 val_dir=None,   
                 val_augmentation=None 
                ):
        
        #check if the directory exists
        assert os.path.exists(data_dir), "Path doesn't exist"
        
        val_aug = val_augmentation if val_augmentation is not None else transforms.ToTensor()
        
        if val_dir is not None:
            assert os.path.exists(val_dir), "Validation path doesn't exist"
            __ssl_train_ds = SSLDataset(data_dir=data_dir, augmentation=augmentation)
            __ssl_val_ds = SSLDataset(data_dir=val_dir, augmentation=val_aug)
        else:

            full_dataset = SSLDataset(data_dir=data_dir, augmentation=augmentation)
            
            total_size = len(full_dataset)
            
            val_size = int(val_split * total_size)
            
            train_size = total_size - val_size
            
            
            train_indices, val_indices = random_split(
                range(total_size), [train_size, val_size]
            )
            
            #another dataset function for creating 2 splits
            class SubsetDataset(Dataset):
                def __init__(self, dataset, indices, transform):
                    self.dataset = dataset
                    self.indices = indices
                    self.transform = transform
                
                def __len__(self):
                    return len(self.indices)
                
                def __getitem__(self, idx):
                    img = self.dataset[self.indices[idx]]
                    return img
            
            __ssl_train_ds = SubsetDataset(full_dataset, train_indices, augmentation)
            __ssl_val_ds = SubsetDataset(full_dataset, val_indices, val_aug)
        
        self.train_dataloader = DataLoader(
            dataset=__ssl_train_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=True,
            pin_memory=True
        )
        
        self.val_dataloader = DataLoader(
            dataset=__ssl_val_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=False,  # No need to shuffle validation data
            pin_memory=True
        )

    def __call__(self):
        """
        SSL Dataloader function
        Returns:
            trainloader, validloader
        """
        return self.train_dataloader, self.val_dataloader
        




        


