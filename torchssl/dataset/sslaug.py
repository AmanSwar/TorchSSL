import torch
from torchvision.transforms import transforms



#--------------------------------------------
#SIMCLR

class SimclrAug:
    def __init__(self , img_size):
        self.img_size = img_size
        self.base_trans = self.base_transforms()
        
    def base_transforms(self):
        _kernel_size = int(self.img_size * 0.1)
        _guas_blur_kernel = _kernel_size if _kernel_size % 2 != 0 else _kernel_size + 1
        base_trans = transforms.Compose(
            [
                
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomResizedCrop(self.img_size , scale=(0.5,1)),
                transforms.RandomApply(
                    [
                        transforms.ColorJitter(brightness=0.8 , contrast=0.8 , saturation=0.8 , hue=0.2)
                    ] , p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=_guas_blur_kernel , sigma=(0.1 , 2.0))] , p=0.5
                ),

                transforms.ToTensor()
                
            ]   
        )

        return base_trans

    def __call__(self , img):

        view1 = self.base_trans(img)
        view2 = self.base_trans(img)

        return view1 , view2







#---------------------------------------------
#MOCO





#---------------------------------------------
#DINO






#---------------------------------------------
#IJEPA