import os
from pathlib import Path
from typing import *

import cv2
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.transforms import functional as F, InterpolationMode

class MVTecLOCO(Dataset):
    def __init__(self, 
        data_root: str, 
        category: str, 
        input_res: int, 
        split: str, 
        is_mask=False, 
        color='rgb'
    ):
        """Dataset for MVTec LOCO.
        Args:
            data_root: Root directory of MVTecAD dataset. It should contain the data directories for each class under this directory.
            category: Class name. Ex. 'hazelnut'
            input_res: Input resolution of the model.
            split: 'train' or 'test'
            is_mask: If True, return the mask image as the target. Otherwise, return the label.
            color: rgb or grayscale
        """
        self.data_root = data_root
        self.category = category
        self.input_res = input_res
        self.split = split
        self.is_mask = is_mask
        self.color = color
        
        assert Path(self.data_root).exists(), f"Path {self.data_root} does not exist"
        assert self.split == 'train' or self.split == 'val' or self.split == 'test'
        
        # # load files from the dataset
        self.img_files = self.get_files()
        if self.split == 'test':
            self.mask_transform = transforms.Compose(
                [
                    transforms.Resize(input_res),
                    transforms.ToTensor(),
                ]
            )

            self.labels = []
            for file in self.img_files:
                status = str(file).split(os.path.sep)[-2]
                if status == 'good':
                    self.labels.append(0)
                else:
                    self.labels.append(1)
    
    def __getitem__(self, index):

        img_file = self.img_files[index]
        img = Image.open(img_file)
        
        inputs = {}

        if self.color == 'gray':
            img = img.convert('L')
        
        img = np.array(img)
        img = cv2.resize(img, (self.input_res, self.input_res)) 
        
        
        if self.split == 'train' or self.split == 'val':
            inputs["images"] = img
            return inputs
        else:
            if not self.is_mask:
                inputs["images"] = img
                inputs["labels"] = self.labels[index]
                return inputs
            else:
                raise NotImplementedError
            # # create mask for normal image
            # if os.path.dirname(img_file).endswith("good"):
            #     mask = torch.zeros([1, img.shape[-2], img.shape[-1]])

            # # read mask for anomaly image
            # else:
            #     sep = os.path.sep
            #     mask = Image.open(
            #         img_file.replace(f"{sep}test{sep}", f"{sep}ground_truth{sep}").replace(
            #             ".png", "_mask.png"
            #         )
            #     )
            #     mask = self.mask_transform(mask)
            # return inputs
    
    def __len__(self):
        return len(self.img_files)
    
    def get_files(self):
        if self.split == 'train':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'train', 'good')).glob('*.png'))
        elif self.split == 'val':
            files = sorted(Path(os.path.join(self.data_root, self.category, 'validation', 'good')).glob('*.png'))
        elif self.split == 'test':
            normal_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'good')).glob('*.png'))
            logical_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'logical_anomalies')).glob('*.png'))
            struct_img_files = sorted(Path(os.path.join(self.data_root, self.category, 'test', 'structural_anomalies')).glob('*.png'))
            files = normal_img_files + logical_img_files + struct_img_files
            
            # logical_mask_files = sorted(Path(os.path.join(self.data_root, self.category, 'ground_truth', 'logical_anomalies/')).glob('*/*.png'))
            # struct_mask_files = sorted(Path(os.path.join(self.data_root, self.category, 'ground_truth', 'structural_anomalies/')).glob('*/*.png'))
            
            # files = logical_mask_files + struct_mask_files
            
        return files
    
def build_loco_dataloader(
    args, 
    dataset,
    training
) -> DataLoader:
    
    # build dataloader
    if training:
        return DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=True,
        )
    else:
        return DataLoader(
            dataset, 
            batch_size=1,
            num_workers=args.num_workers,
            pin_memory=True,
            shuffle=False
        )
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="/home/sakai/projects/VQ-Reconstructor/data/MVTec-LOCO")
    parser.add_argument("--category", type=str, default="breakfast_box")
    parser.add_argument("--input_res", type=int, default=224)
    parser.add_argument("--is_mask", action="store_true")
    parser.add_argument("--color", type=str, default="rgb")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    
    args = parser.parse_args()
    
    dataset = MVTecLOCO(
        data_root="/home/sakai/projects/VQ-Reconstructor/data/MVTec-LOCO/mvtec_loco",
        category="breakfast_box",
        input_res=224,
        split="test",
        is_mask=False,
    )
    dataloader = build_loco_dataloader(
        args,
        dataset,
        training=False
    )
    
    for sample in dataloader:
        print(sample["images"].shape)
        print(sample["labels"])
        
        if sample["labels"] == 0:
            print("Normal")