import torch
import torch.nn as nn

import numpy as np

from .backbone import build_backbone

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, use_batchnorm = True):
        super().__init__()
        
        self.conv1 = Conv2ReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = Conv2ReLU(out_channels, out_channels,kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        
    
    def forward(self,x, skip = None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
        

class DecoderCup(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        
        self.d_model = d_model
        self.conv1 = Conv2ReLU(2048,1024,kernel_size=3)
        self.decoder1 = DecoderBlock(2048, 512,)
        self.decoder2 = DecoderBlock(1024, 256)
        self.decoder3 = DecoderBlock(512, 128)
        self.decoder4 = DecoderBlock(128, 64)
        
        self.SegmentationHead = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        
    def forward(self, src, features):
        bs, d_model, n_patches = src.size()
        src = src.view(bs, d_model, int(n_patches ** 0.5), int(n_patches ** 0.5))
        src = self.conv1(src)
        src = self.decoder1(src, features['2'])
        src = self.decoder2(src, features['1'])
        src = self.decoder3(src, features['0'])
        src = self.decoder4(src)
        output = self.SegmentationHead(src)
        
        return output
        
        

class Conv2ReLU(nn.Sequential):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        padding=1,
        stride=1,
        use_batchnorm = True,
    ):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not (use_batchnorm))
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels)

        super().__init__(conv, bn, relu)
    
# class SegmentationHead(nn.Sequential):
#     def __init__(self, in_channels, n_classes, kernel_size=3, upsampling=1):
#         conv2d = nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=kernel_size//2)
#         upsampling = nnUpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        
#         super().__init__(conv2d, upsampling)
            
            
def upsampler_decoder(args):
    return DecoderCup(
        d_model = args.d_model,
        n_classes = args.n_classes
    )

