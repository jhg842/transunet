import torch
from torch import nn

from model.backbone import build_backbone


class PositionEmbedding(nn.Module):
    def __init__(self, backbone, patch_size, num_patches):
        super().__init__()
        
        self.backbone = backbone
        self.patch_size = patch_size
        # self.num_patches = num_patches
        
        # self.linear = nn.Linear()
        # self.patch_embedding = nn.Conv2d(in, out, )
        # self.position = nn.Parameter(torch.randn(1, num_patches, vec_dim))
        
        
        
    def forward(self, x):
        
        x = self.backbone(x)['2']
        B, C, H, W = x.shape
        num_patches = H // self.patch_size
        x = x.view(B, C, num_patches, self.patch_size, num_patches, self.patch_size)
        x = x.permute(0,2,4,1,3,5)
        x = x.contiguous().view(B, num_patches**2, C*self.patch_size*self.patch_size)
    
        return x
        
x = torch.randn(1,3,224,224)
backbone = build_backbone('resnet50', True)

model = PositionEmbedding(backbone, 2,None)
print(model(x).shape)