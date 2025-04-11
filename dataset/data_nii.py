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
    
class JointTransform:
    
    def __init__(self, image_set):
        self.image_set = image_set
        
        self.image_transform = T.Compose([
            
            T.Resize((256,256)),
        ])

        self.label_transform = T.Compose([
            T.Resize([256,256], interpolation = T.InterpolationMode.NEAREST),
          
        ])
        
    def __call__(self, img, label):
        
        if self.image_set == 'train':
        
            if random.random() > 0.5:
                k = np.random.randint(0, 4)
                img = np.rot90(img, k)
                label = np.rot90(label, k)
                axis = np.random.randint(0, 2)
                img = np.flip(img, axis=axis).copy()
                label = np.flip(label, axis=axis).copy()
            elif random.random() > 0.5:
                angle = np.random.randint(-20, 20)
                img = ndimage.rotate(img, angle, order=0, reshape=False)
                label = ndimage.rotate(label, angle, order=0, reshape=False)
                
        img = torch.from_numpy(img.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        
        img = self.image_transform(img)
        label = self.label_transform(label)
        
        return img, label
        
 

def build_nii(image_set, args):
    root = Path(args.NG_path)
    assert root.exists(), f'provided NiiSliceDataset path {root} does not exist'
    
    PATHS = {
        "train": (root / "RawData"/ "Training"/"img", root / "RawData"/ "Training"/"label"),
        "val": (root / "RawData"/ "Val"/"img", root / "RawData"/ "Val"/"label"),
    }
    
    img_folder, label_folder = PATHS[image_set]
    dataset = NiiSliceDataset(img_folder, label_folder, transform = JointTransform(image_set))
    
    return dataset