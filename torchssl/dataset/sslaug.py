import torch
from torchvision.transforms import transforms
from torchvision.transforms import InterpolationMode


#--------------------------------------------
#SIMCLR

class SimclrAug:
    def __init__(self , img_size):
        self.img_size = img_size
        self.base_trans = self.base_transforms()
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)
        
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

                transforms.ToTensor(),
                # transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
            ]   
        )

        return base_trans

    def __call__(self , img):

        view1 = self.base_trans(img)
        view2 = self.base_trans(img)

        return view1 , view2


#---------------------------------------------
#MOCO

class MocoAug:

    def __init__(
            self,
            img_size=224,
            s=1.0
    ):
        color_jitter = transforms.ColorJitter(
            brightness=0.4,
            contrast=0.4,
            saturation=0.4,
            hue=0.1
        )

        self.base_trans = transforms.Compose([
                transforms.RandomResizedCrop(size=img_size, scale=(0.2, 1.0)),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),
            ])
    def __call__(self, img):
        q = self.base_transform(img)
        k = self.base_transform(img)
        return q, k



#---------------------------------------------
#DINO

class DinoAug:

    def __init__(
            self,
            global_crop_size = 224,
            local_crop_size = 96,
            global_crop_scale = (0.4 , 1.0),
            local_crop_scale = (0.05, 0.4),
            local_crop_num = 8
    ):
        
        self.global_crops_scale = global_crop_scale
        self.local_crops_scale = local_crop_scale
        self.local_crops_number = local_crop_num
        self.global_crops_size = global_crop_size
        self.local_crops_size = local_crop_size

        #imagenet val
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        flip_and_color_jitter_trans = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean, std=self.std),
        ])

        self.global_transform1 = transforms.Compose([
            transforms.RandomResizedCrop(
                size=self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            flip_and_color_jitter_trans,
            transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23 , sigma=(0.1 , 2.0))] , p=0.5
                ),
            normalize_trans,
        ])


        self.global_transform2 = transforms.Compose([
            transforms.RandomResizedCrop(
                size=self.global_crops_size,
                scale=self.global_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            flip_and_color_jitter_trans,
            transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23 , sigma=(0.1 , 2.0))] , 
                    p=0.5
                    ),
            transforms.RandomSolarize(threshold=0.5),
            normalize_trans,
        ])
        
        self.local_transform = transforms.Compose([
            transforms.RandomResizedCrop(
                size=self.local_crops_size,
                scale=self.local_crops_scale,
                interpolation=transforms.InterpolationMode.BICUBIC,
            ),
            flip_and_color_jitter_trans,
            transforms.RandomApply(
                    [transforms.GaussianBlur(kernel_size=23 , sigma=(0.1 , 2.0))] , 
                    p=0.5
                    ),
            normalize_trans,
        ])

    def __call__(self , img):

        crops = []

        crops.append(self.global_transform1(img=img))
        crops.append(self.global_transform2(img=img))

        for _ in range(self.local_crops_number):
            crops.append(self.local_transform(img))

        return crops
        



#---------------------------------------------
#IJEPA

class IjepaAug:

    def __init__(
            self,
            img_size
    ):
        self.mean = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

        self.img_size = img_size

        self.base_trans = transforms.Compose(
            [
                transforms.Resize(self.img_size , interpolation=InterpolationMode.BICUBIC),
                transforms.CenterCrop(self.img_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean ,std=self.std)
            ]
        )


    def __call__(
            self,
            img
    ):
        return self.base_trans(img=img)
    
    
        