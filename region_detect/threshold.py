import cv2
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader
import sys
sys.path.append('../ece613-project2')
from data import DAGMDataset
from main import ohe
from settings import TRAINING_LABEL_FILE_PATH, TEST_LABEL_FILE_PATH, MER_OUTPUT
from pathlib import Path

import os

RGB_WHITE = (255, 255, 255)

def to_numpy(img):
    return img.clone().detach().cpu().numpy()


def binarize(image, threshold):
    return (image > threshold).astype(image.dtype)


def mer(image):
    return cv.boundingRect(image)


if __name__ == '__main__':
    training_set = DAGMDataset(meta_file=TRAINING_LABEL_FILE_PATH, target_transform=ohe, defect_only=True)
    testing_set = DAGMDataset(meta_file=TEST_LABEL_FILE_PATH, target_transform=ohe, defect_only=True)

    train_dataloader = DataLoader(training_set, batch_size=5)
    test_dataloader = DataLoader(testing_set, batch_size=5)

    train_features, train_labels, img_paths, has_defects, defect_mask_paths, defect_masks = next(iter(train_dataloader))

    Path(MER_OUTPUT).mkdir(parents=True, exist_ok=True)

    # load some images
    for img, img_path in zip(train_features, img_paths):
        # flatten multi channel image into single channel image
        img_np = to_numpy(img).squeeze()

        # set the new pixel value to 1 if the original one is greater than `threshold` else 0
        bin_img = binarize(img_np, 200)
        x, y, w, h = mer(bin_img)

        # find the bounding box and draw it on the original image
        bounded_img = cv.rectangle(np.copy(img_np), (x, y), (x + w, y + h), RGB_WHITE, 2)

        # find the bounding box and draw it on the binarized image
        bounded_bin_img = cv.rectangle(np.copy(bin_img) * 255, (x, y), (x + w, y + h), RGB_WHITE, 2)

        # crop the bounded region
        crop_img = img_np[y: y + h, x: x + w]

        mask_img = np.zeros(img_np.shape)

        # please the image writer by converting 0s and 1s to 0s and 255s
        mask_img[y: y + h, x: x + w] = 255

        img_filename = os.path.basename(img_path)
        name, ext = img_filename.split('.')
        cv2.imwrite(os.path.join(MER_OUTPUT, f'{name}_bounded.{ext}'), bounded_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'{name}_bounded_bin.{ext}'), bounded_bin_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'{name}_mask.{ext}'), mask_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'{name}_crop.{ext}'), crop_img)
