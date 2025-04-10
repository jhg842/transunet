import torch
from torch import nn

import torchvision

from .backbone import build_backbone
from .embedding import build_embedding
from .transformer import build_transformer
from .upsampler import upsampler_decoder

class TransUNet(nn.Module):
    def __init__(self, backbone, embedding, transformer, upsampler):
        super().__init__()
        self.backbone = backbone
        self.embedding = embedding
        self.transformer = transformer
        self.upsampler = upsampler
        
    def forward(self, x):
        features = self.backbone(x)
        embed_features = self.embedding(features)
        trans_features = self.transformer(embed_features)
        output = self.upsampler(trans_features, features)
        
        return output
    
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes
    
    def _one_hot_encoding(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i
            tensor_list.append(temp_prob.unsqueeze(1))
            output_tensor = torch.cat(tensor_list, dim=1)
            
        return output_tensor.float()
        
    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        
        return loss
    
    
    def forward(self, inputs, target, weight = None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoding(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), f'predict {inputs.size()} & target {target.size()} shape do not match'
        class_with_dice = []
        loss = 0.0
        for i in range(self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_with_dice.append(1 - dice.item())
            loss += dice * weight[i]
            
        return loss / self.n_classes


def build(args):
    
    backbone = build_backbone(args)
    embedding = build_embedding(args)
    transformer = build_transformer(args)
    upsampler = upsampler_decoder(args)
    
    model = TransUNet(backbone, embedding, transformer, upsampler)
    
    criterion = DiceLoss(args.n_classes)
    
    return model, criterion