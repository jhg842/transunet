import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models._utils import IntermediateLayerGetter

# class FrozenbBatchNorm2d(nn.Module):
#     def __init__(self):
#         super().__init__()
        
#     def forward(self,):
        
class BackboneBase(nn.Module):
    def __init__(self,backbone: nn.Module, num_channels: int, return_interm_layers: bool):
        super().__init__()
        for name, parameter in backbone.named_parameters():
            if 'layer2' not in name and 'layer3' not in name and 'layer4' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2'}
            
        else:
            return_layers = {'layer4': '0'}
        
        self.body = IntermediateLayerGetter(backbone, return_layers=return_layers)
        self.num_channels = num_channels
        
    def forward(self, xs):
        output = self.body(xs)
        out = {}
        for name, x in output.items():
            out[name] = x
            
        return out
    
class Backbone(BackboneBase):
    def __init__(self, name, return_interm_layers):
        backbone = getattr(models, name)(pretrained=True)
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels, return_interm_layers)
        

model = Backbone('resnet50', True)
img = torch.randn(1,3,256,256)
output = model(img)
print(output['2'].shape)
     
# def build_backbone(args):
#     backbone = Backbone(args.backbone, return_interm_layers)
    
#     return backbone         
        

# model = models.resnet50(pretrained=True)

# print(model)
# return_layers = {'layer1':'0', 'layer2': '1'}
# model = IntermediateLayerGetter(model, return_layers=return_layers)
# img = torch.randn(1,3,256,256)
# output = model(img)
# out = {}
# for name, x in output.items():
#     out[name] = x
    
# print(out['0'])

# output_val = list(output.values())
# print(output_val[1].shape)

