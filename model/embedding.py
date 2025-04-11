import torch
from torch import nn

from .backbone import build_backbone


class PositionEmbedding(nn.Module):
    def __init__(self,  patch_size, num_patches, d_model):
        super().__init__()
        
        self.patch_size = patch_size
        
        self.patch_embedding = nn.Conv2d(2048, d_model, kernel_size=patch_size, stride=patch_size)
        self.position = nn.Parameter(torch.randn(1, num_patches**2, d_model))
        
        
    def forward(self, features):
        
        x = self.patch_embedding(features['3'])
        x = x.flatten(2).transpose(1,2)
        x += self.position
    
        return x
        

def build_embedding(args):
    
    return PositionEmbedding( args.patch_size, args.num_patches, args.d_model)
