from pathlib import Path
import os
import random
import numpy as np
from scipy import ndimage
from scipy.ndimage.interpolation import zoom
import nibabel as nib

import os
import nibabel as nib
import numpy as np
from scipy import ndimage
import torch
import torchvision
import torchvision.transforms as T
from torch.utils.data import Dataset
from PIL import Image

class NiiSliceDataset(Dataset):
    def __init__(self, data_dir, label_dir, transform):
        self.data_slices = []  # (img_path, label_path, slice_index) 리스트 저장
        self.transform = transform
        
        # 이미지-라벨 파일 목록 정렬 (파일 이름이 일치한다고 가정)
        img_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.nii.gz')])
        label_files = sorted([f for f in os.listdir(label_dir) if f.endswith('.nii.gz')])

        # 각 파일에서 슬라이스별로 항목 추가
        for img_file, label_file in zip(img_files, label_files):
            img_path = os.path.join(data_dir, img_file)
            label_path = os.path.join(label_dir, label_file)

            img_data = nib.load(img_path).get_fdata()
            label_data = nib.load(label_path).get_fdata()

            num_slices = img_data.shape[2]

            for slice_idx in range(num_slices):
                self.data_slices.append((img_path, label_path, slice_idx))

    def __len__(self):
        return len(self.data_slices)

    def __getitem__(self, idx):
        img_path, label_path, slice_idx = self.data_slices[idx]

        # Load and slice
        img = nib.load(img_path).get_fdata()[:, :, slice_idx]
        label = nib.load(label_path).get_fdata()[:, :, slice_idx]

        # Normalize 이미지 (0~1 정규화)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)


        # Add channel dimension: (1, H, W)
        # label = np.expand_dims(label, axis=0)  # 선택: 필요 없을 수도 있음

        # Tensor 변환
        # img = torch.tensor(img, dtype=torch.float32)
        # label = torch.tensor(label, dtype=torch.long)

        if self.transform:
            img, label = self.transform(img, label)

        return img, label

def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class JointTransform:
    
    def __init__(self, image_set):
        self.image_set = image_set
        

        
    def __call__(self, img, label):
        
        if random.random() > 0.5:
            img, label = random_rot_flip(img, label)
        if random.random() > 0.5:
            img, label = random_rotate(img, label)
            
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label).long()
        
        return img, label
        
 

def build_nii(image_set, args):
    root = Path(args.NG_path)
    print(f"Looking for data at: {root}")
    assert root.exists(), f'provided NiiSliceDataset path {root} does not exist'
    
    PATHS = {
        "train": (root / "RawData"/ "Training"/"img", root / "RawData"/ "Training"/"label"),
        "val": (root / "RawData"/ "Val"/"img", root / "RawData"/ "Val"/"label"),
    }
    
    img_folder, label_folder = PATHS[image_set]
    dataset = NiiSliceDataset(img_folder, label_folder, transform = JointTransform(image_set))
    
    return dataset


# /mnt/d/Users/jhg84/Downloads