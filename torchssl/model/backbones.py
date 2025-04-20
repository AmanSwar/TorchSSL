import timm
import torch
import torch.nn as nn


class Backbone(nn.Module):

    def __init__(self, name , in_channels):
        super().__init__()
        self.name = name
        self.in_channels = in_channels
        self.backbone = timm.create_model(self.name ,in_chans=in_channels , pretrained=False , num_classes=0)
    

    def get_features(self) -> int:
        """
        Get backbone output dimension 
        """
        return self.backbone.num_features
    
    def forward(self , x):
        out = self.backbone(x)
        return out  

class ConvNextBackbone(Backbone):
    def __init__(self , in_channels):
        super().__init__(in_channels)
        self.name = "convnext_base"
        


# class ConvNextBackbone(nn.Module):
#     def __init__(self , num_channels):
#         self.backbone = timm.create_model("convnext_base" ,num_channels=num_channels , pretrained=False , num_classes=0)
        
#     def forward(self,x):
#         out = self.backbone(x)
#         return out
    

class ResnetBackbone(nn.Module):

    def __init__(self , num_channels):

        self.backbone = timm.create_model("resnet50" , num_channels=num_channels ,pretrained=False , num_classes=0)

    def forward(self , x):
        out = self.backbone(x)
        return out
    


