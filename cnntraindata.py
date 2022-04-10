import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sampler import sampler
from torch.utils.data import DataLoader
import sys
import cv2
from data import DAGMDataset
from settings import TRAINING_LABEL_FILE_PATH, TEST_LABEL_FILE_PATH, MER_OUTPUT
from pathlib import Path
import matplotlib.pyplot as plt
from sampler import sampler
class CnnDataset(DAGMDataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=True):
        super(CnnDataset, self).__init__(meta_file,transform,target_transform, defect_only)

    def __getitem__(self,idx):
        image, label, img_path, has_defect, defect_mask_path, defect_mask = super(CnnDataset, self).__getitem__(idx)
        npimage = image.clone().detach().cpu().numpy().squeeze()
        npmask =  defect_mask.clone().detach().cpu().numpy().squeeze()
        mask = (npmask > 10).astype("uint8")
        x,y,w,h = cv2.boundingRect(mask)
        crop_img = npimage[y: y + h, x: x + w]   
        crop_img = cv2.resize(crop_img,[227,227])     
        return crop_img.reshape(1,227,227), label


def run():
    training_set = CnnDataset(meta_file=TRAINING_LABEL_FILE_PATH)
    testing_set = CnnDataset(meta_file=TEST_LABEL_FILE_PATH)

    train_dataloader = DataLoader(training_set, batch_size=5)

    train_features, train_labels = next(iter(train_dataloader))
    img = train_features[0].squeeze()

    plt.imshow(img, cmap="gray")
    plt.show()


if __name__ == '__main__':
    run()