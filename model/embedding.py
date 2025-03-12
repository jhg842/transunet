import torch
from torch import nn

from backbone import build_backbone


class PositionEmbedding(nn.Module):
    def __init__(self, backbone,  patch_size, vec_dim, position:bool):
        super().__init__()
        
        self.backbone = backbone
        self.patch_size = patch_size
        
        # self.linear = nn.Linear()
        # self.patch_embedding = nn.Conv2d(in, out, )
        self.position = nn.Parameter(torch.randn(1, num_patches**2, vec_dim))
        
        
        
    def forward(self, x):
        
        x = self.backbone(x)['2']
        # x = self.patch_embedding(x)
        B, C, H, W = x.shape
        x = x.view(B,C, H*W)
        x = x.transpose(1,2)
        
        if position:
            x += self.position
    
        return x
        
x = torch.randn(1,3,224,224)
backbone = build_backbone('resnet50', True)

model = PositionEmbedding(backbone, 14, 1024)
print(model(x).shape)

def build_embedding(args):
    
    backbone = args.backbone
    return PositionEmbedding(backbone, args.patch_size, args.vec_dim, args.position)
