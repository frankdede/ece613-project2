import pandas as pd
from torch.utils.data.dataset import Dataset
from torchvision.io import read_image


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
