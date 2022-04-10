import os
import sys

import cv2 as cv
import cv2
import pandas as pd

from ae.ae import autoencoder, autoencoder_wasp
from ae.casae import CASAE

from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torch
from data import DAGMDataset, AEDataset
from region_detect.threshold import to_numpy, binarize, mer
from settings import TEST_LABEL_FILE_PATH, MER_OUTPUT, DATASET_META_DIR
import matplotlib.pyplot as plt
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# preprocessing
transform = transforms.Normalize((0,), (255,))
testing_set = AEDataset(meta_file=TEST_LABEL_FILE_PATH, transform=transform, defect_only=True)

# load test data
test_dataloader = DataLoader(testing_set, batch_size=1, shuffle=False)

# load CASAE
def draw(img):
    plt.imshow(img.cpu().numpy().reshape((512, 512)), cmap='gray')
    plt.axis('off')


if __name__ == '__main__':
    if not sys.argv[1]:
        sys.exit("Missing model stack id argument: 1 - default, 2 - dynamic loss, 3 - WSAP net")

    if not sys.argv[2]:
        sys.exit("Missing threshold")

    stack_id = int(sys.argv[1])
    threshold = int(sys.argv[2])

    assert 0 <= threshold <= 255, "Invalid threshold"
    stack_list = [
        ("./models/ae1-11_class-default_net-default_loss.pt",
         "./models/ae2-11_class-default_net-default_loss.pt",),
        ("./models/ae1-11_class-default_net-dynamic_loss.pt",
         "./models/ae2-11_class-default_net-dynamic_loss.pt",),
        ("./models/ae1-11_class-wasp_net-default_loss.pt",
         "./models/ae2-11_class-default_net-dynamic_loss.pt",),
    ]

    assert 1 <= stack_id <= len(stack_list), "Unknown model stack id"

    ae1_path, ae2_path = stack_list[stack_id - 1]
    if stack_id == 3:
        ae1 = autoencoder_wasp()
    else:
        ae1 = autoencoder()
    ae1.load_state_dict(torch.load(ae1_path))
    ae1.eval()

    ae2 = autoencoder()
    ae2.load_state_dict(torch.load(ae2_path))
    ae2.eval()

    casae = CASAE(ae1, ae2).to(device)

    iou_list = []
    meta = []

    output_dir = os.path.join(MER_OUTPUT, str(stack_id))
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for trans_image, image, defect_mask, label, img_path, name, ext in test_dataloader:
        with torch.no_grad():
            # ground truth
            defect_mask_np = to_numpy(defect_mask[0]).reshape((512, 512)).squeeze()
            trans_image = trans_image.to(device)
            # prediction
            output = casae(trans_image)

            img_np = to_numpy(output).reshape((512, 512)).squeeze()
            # set the new pixel value to 1 if the original one is greater than `threshold` else 0
            bin_img = binarize(img_np, threshold/255)
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

            crop_img_name = f'class_{label[0]}-crop_{name[0]}-threshold_{threshold}.{ext[0]}'
            crop_img_path = os.path.join(output_dir, crop_img_name)

            cv2.imwrite(crop_img_path, crop_img)
            meta.append({"model_stack": stack_id, "label": label.item(), "path": crop_img_path, "iou": iou})

    df = pd.DataFrame(meta)
    df.to_csv(os.path.join(output_dir, f"meta-threshold_{threshold}.csv"), index=False)
