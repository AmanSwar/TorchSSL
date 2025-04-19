import os
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset , DataLoader , WeightedRandomSampler
import numpy as np
from torchvision.transforms import transforms

class SSLDataset(Dataset):
    
    def __init__(self , data_dir , augmentation):
        super().__init__()
        
        self.aug = augmentation
        self.tensor_aug = transforms.ToTensor()
        self.IMAGE_PATH = []    
        #traverse along paths
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
    

class SSLDataloader(DataLoader):
    def __init__(self , 
                 data_dir,
                 augmentation,
                 batch_size,
                 num_workers,

                 ):
        """
        data_dir : directory containing all the images 
        """
        #check if the directory exist
        assert os.path.exists(data_dir) , "Path doesn't exist"

        __ssl_ds = SSLDataset(data_dir=data_dir , augmentation=augmentation)

        super(SSLDataloader , self).__init__(
            dataset=__ssl_ds,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=True
        )





        


