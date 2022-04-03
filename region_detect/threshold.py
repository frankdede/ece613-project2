import cv2
import cv2 as cv
import numpy as np
from torch.utils.data import DataLoader

from data import DAGMDataset
from main import ohe
from settings import TRAINING_LABEL_FILE_PATH, TEST_LABEL_FILE_PATH, MER_OUTPUT
from pathlib import Path

import os


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

    for img, img_path in zip(train_features, img_paths):
        img_np = to_numpy(img).squeeze()
        bin_img = binarize(img_np, 200)
        x, y, w, h = mer(bin_img)

        bounded_img = cv.rectangle(np.copy(img_np), (x, y), (x + w, y + h), (0, 255, 0), 2)
        bounded_bin_img = cv.rectangle(np.copy(bin_img) * 255, (x, y), (x + w, y + h), (0, 255, 0), 2)

        crop_img = img_np[y: y + h, x: x + w]

        mask_img = np.zeros(img_np.shape)
        mask_img[y: y + h, x: x + w] = 255

        img_filename = os.path.basename(img_path)
        print(img_path)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'bounded_{img_filename}'), bounded_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'bounded_bin_{img_filename}'), bounded_bin_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'mask_{img_filename}'), mask_img)
        cv2.imwrite(os.path.join(MER_OUTPUT, f'crop_{img_filename}'), crop_img)
