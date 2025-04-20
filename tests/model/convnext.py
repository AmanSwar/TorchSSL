import torch
from torchssl.model.backbones import Backbone


model = Backbone("convnext_small" , in_channels=3)

temp_input = torch.rand(size=(8,3,224,224))

out = model(temp_input)

print(out.shape)
print(model.get_features())
