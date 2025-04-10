import torch.utils.data
import torchvision

from .data_nii import build_nii

def build_dataset(image_set, args):
    
    return build_nii(image_set, args)