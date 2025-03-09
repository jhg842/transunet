import torch
import torch.nn as nn

from .backbone import build_backbone

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm):
        super().__init__()
        
        self.conv1 = Conv2ReLU(in_channels, out_channels, kernel_size=3, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        
    
    def forward(self,src):

        

class DecoderCup(nn.Module):
    def __init__(self):
        
    def forward(self, src):
        bs, d, num_patches = src.shape
        src = src.view(bs, d, H,W) # H, W is original img size

class Conv2ReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
        use_batchnorm,
    )
    conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
    relu = nn.ReLU(inplace=True)
    bn = nn.BatchNorm2d(out_channels)
    
    super(Conv2dReLU,self).__init__(conv, bn, relu)