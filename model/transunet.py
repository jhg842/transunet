import torch
from torch import nn

import torchvision

from backbone import build_backbone
from embedding import build_embedding
from transformer import build_transformer
from upsampler import upsampler_decoder

class TransUNet(nn.Module):
    def __init__(self, embedding, transformer, num_classes):
        self.embedding = embedding
        self.transformer = transformer
        self.num_classes = num_classes
        
        
        
    def forward(self, x):
        x = self.embedding(x)
        x = transformer(x)
        x = upsampler(x)
        
        return x
    
class SetCriterion(nn.Module):
    
        