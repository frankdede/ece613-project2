import os

import cv2 as cv
import cv2
import pandas as pd

from ae.ae import autoencoder
from ae.casae import CASAE

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from data import DAGMDataset
from region_detect.threshold import to_numpy, binarize, mer
from settings import TEST_LABEL_FILE_PATH, MER_OUTPUT, DATASET_META_DIR
import matplotlib.pyplot as plt
from torchvision.io import read_image
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class AEDataset(DAGMDataset):
    def __init__(self, meta_file, transform=None, target_transform=None, defect_only=False):
        super(AEDataset, self).__init__(meta_file, transform, target_transform, defect_only)

    def __getitem__(self, idx):
        img_path = self.meta_df["img_file"].iloc[idx]
        image = read_image(img_path).float()
        defect_mask_path = str(self.meta_df["label_file"].iloc[idx])
        label = self.meta_df["class"].iloc[idx]

        if self.transform:
            trans_image = self.transform(image)

        defect_mask = read_image(defect_mask_path).flatten().float()
        defect_mask[defect_mask != 0.] = 1

        img_filename = os.path.basename(img_path)
        name, ext = img_filename.split('.')
        return trans_image.float(), image, defect_mask.float(), label, img_path, name, ext


# preprocessing
transform = transforms.Normalize((0,), (255,))
testing_set = AEDataset(meta_file=TEST_LABEL_FILE_PATH, transform=transform, defect_only=True)

# load test data
test_dataloader = DataLoader(testing_set, batch_size=1, shuffle=False)

# load CASAE
ae1 = autoencoder()
ae1.load_state_dict(torch.load("./models/ae1-11_class-default_net-default_loss.pt"))
ae1.eval()

ae2 = autoencoder()
ae2.load_state_dict(torch.load("./models/ae2-11_class-default_net-default_loss.pt"))
ae2.eval()

casae = CASAE(ae1, ae2).to(device)


def draw(img):
    plt.imshow(img.cpu().numpy().reshape((512, 512)), cmap='gray')
    plt.axis('off')


if __name__ == '__main__':

    if not os.path.exists(DATASET_META_DIR):
        os.mkdir(DATASET_META_DIR)

    output_cols = ["crop_img", "class", "original_img"]
    df = pd.DataFrame(columns=output_cols)

    iou_list = []
    for trans_image, image, defect_mask, label, img_path, name, ext in test_dataloader:
        with torch.no_grad():
            # ground truth
            defect_mask_np = to_numpy(defect_mask[0]).reshape((512, 512)).squeeze()
            trans_image = trans_image.to(device)
            # prediction
            output = casae(trans_image)

            img_np = to_numpy(output).reshape((512, 512)).squeeze()
            # set the new pixel value to 1 if the original one is greater than `threshold` else 0
            bin_img = binarize(img_np, 100/255)
            defect_mask_bool = np.array(defect_mask_np, dtype=bool)
            bin_img_bool = np.array(bin_img, dtype=bool)

            # IoU
            intersection = np.logical_and(defect_mask_bool, bin_img_bool)
            union = np.logical_or(defect_mask_bool, bin_img_bool)
            iou = np.sum(intersection) / np.sum(union)
            iou_list.append(iou)

            x, y, w, h = mer(bin_img)

            print(f"current IoU {iou}, mean {np.mean(iou_list)} MER f{(x, y, w, h)}")

            # crop the bounded region
            crop_img = to_numpy(image).reshape((512, 512)).squeeze()[y: y + h, x: x + w]
            if w == 0 and h == 0:
                crop_img = np.zeros((1, 1))
            # please the image writer by converting 0s and 1s to 0s and 255s
            cv2.imwrite(os.path.join(MER_OUTPUT, f'class_{label[0]}_crop_{name[0]}.{ext[0]}'), crop_img)

