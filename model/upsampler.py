import torch
import torch.nn as nn

import numpy as np

from .backbone import build_backbone

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm):
        super().__init__()
        
        self.conv1 = Conv2ReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)
        self.conv2 = Conv2ReLU(out_channels, out_channels,kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
        
    
    def forward(self,x, skip=None):
        x = self.up(x)
        if skip in not None:
            # add a backbone concat
        x = self.conv1(x)
        x = self.conv2(x)
        
        return x
        

class DecoderCup(nn.Module):
    def __init__(self, d_model, n_classes):
        super().__init__()
        
        self.d_model = d_model
        self.conv1 = Conv2ReLU(self.d_model,512,kernel_size=3, use_batchnorm=True)
        self.seghead = SegmentationHead(in_channels, n_classes, kernel_size=kernel_size, upsampling=1)
        self.cov_repeat = Conv2ReLU()
        
        
    def forward(self, src):
        bs, self.d_model, n_patches = src.shape
        H, W = int(np.sqrt(n_patches)), int(np.sqrt(n_patches))
        src = src.view(bs, self.d_model, H/16 ,W/16) # H, W is original img size
        src = self.conv1(src)
        
        output = SegmentationHead(src)
        
        return output
        
        

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
    
    super().__init__(conv, bn, relu)
    
    class SegmentationHead(nn.Sequential):
        def __init__(self, in_channels, n_classes, kernel_size=3, upsampling=1):
            conv2d = nn.Conv2d(in_channels, n_classes, kernel_size=kernel_size, padding=kernel_size//2)
            upsampling = nnUpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identify()
            
            super().__init__(conv2d, upsampling)
            
            
def upsampler_decoder(args):
    return DecoderCup(
        d_model = args.d_model,
        n_classes = args.n_classes
    )