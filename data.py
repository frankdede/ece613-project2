import os

import pandas as pd
import torch
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image
import cv2


class DAGMDataset(Dataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=False):
        meta_df = pd.read_csv(meta_file)
        if defect_only:
            self.meta_df = meta_df[meta_df["has_defect"] == 1]
        else:
            self.meta_df = meta_df
        self.transform = transform
        self.target_transform = target_transform
        self.defect_only = defect_only

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        img_path = self.meta_df["img_file"].iloc[idx]
        image = read_image(img_path)
        label = self.meta_df["class"].iloc[idx]
        has_defect = self.meta_df["has_defect"].iloc[idx] == 1
        defect_mask_path = str(self.meta_df["label_file"].iloc[idx])
        defect_mask = read_image(defect_mask_path)

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path, has_defect, defect_mask_path, defect_mask


class AEDataset(DAGMDataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=False):
        super(AEDataset, self).__init__(meta_file, transform, target_transform, defect_only)

    def __getitem__(self, idx):
        img_path = self.meta_df["img_file"].iloc[idx]
        image = read_image(img_path).float()
        defect_mask_path = str(self.meta_df["label_file"].iloc[idx])
        label = self.meta_df["class"].iloc[idx]

        trans_image = image
        if self.transform:
            trans_image = self.transform(image)

        defect_mask = read_image(defect_mask_path).flatten().float()
        defect_mask[defect_mask != 0.] = 1

        img_filename = os.path.basename(img_path)
        name, ext = img_filename.split('.')
        return trans_image.float(), image, defect_mask.float(), label, img_path, name, ext


class CnnDataset(DAGMDataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=True):
        super(CnnDataset, self).__init__(meta_file, transform, target_transform, defect_only)

    def __getitem__(self, idx):
        image, label, img_path, has_defect, defect_mask_path, defect_mask = super().__getitem__(idx)
        npimage = image.clone().detach().cpu().numpy().squeeze()
        npmask = defect_mask.clone().detach().cpu().numpy().squeeze()
        mask = (npmask > 10).astype("uint8")
        x, y, w, h = cv2.boundingRect(mask)
        crop_img = npimage[y: y + h, x: x + w]
        crop_img = cv2.resize(crop_img, [227, 227])
        return torch.from_numpy(crop_img.reshape(1, 227, 227)), label, img_path, has_defect, defect_mask_path, \
               defect_mask


class DetectionModuleDataset(Dataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=False):
        self.meta_df = pd.read_csv(meta_file)
        self.transform = transform
        self.target_transform = target_transform
        self.defect_only = defect_only

    def __len__(self):
        return len(self.meta_df)

    def __getitem__(self, idx):
        img_path = self.meta_df["path"].iloc[idx]
        image = read_image(img_path)
        label = self.meta_df["label"].iloc[idx]
        model_stack_id = self.meta_df["model_stack"].iloc[idx]

        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label, img_path, model_stack_id
