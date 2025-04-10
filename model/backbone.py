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
            if 'layer2' not in name and 'layer3' not in name:
                parameter.requires_grad_(False)
        if return_interm_layers:
            return_layers = {'layer1': '0', 'layer2': '1', 'layer3': '2', 'layer4': '3'}
            
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
        backbone = getattr(models, name)(pretrained=True,replace_stride_with_dilation=[False, False, False])
        backbone.maxpool = nn.Identity()
        backbone.conv1 = nn.Conv2d(1,64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        # backbone.layer3[5].conv3 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(1, 1), bias=False)
        # backbone.layer3[5].bn3 = nn.BatchNorm2d(512, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        # backbone.layer3[0].downsample =nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, stride=2, bias=False),nn.BatchNorm2d(512))
        num_channels = 512 if name in ('resnet18', 'resnet34') else 2048
        super().__init__(backbone, num_channels, return_interm_layers)
        


     
def build_backbone(args):
    return_interm_layers = args.layers
    backbone = Backbone(args.backbone, return_interm_layers)
    
    return backbone         
        

# model = models.resnet50(pretrained=True)
# print(model)

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

