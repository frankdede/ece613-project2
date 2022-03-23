import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from data import DAGMDataset
from settings import TRAINING_LABEL_FILE_PATH, TEST_LABEL_FILE_PATH
import matplotlib.pyplot as plt


def run():
    num_classes = 10
    ohe = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))

    training_set = DAGMDataset(meta_file=TRAINING_LABEL_FILE_PATH, target_transform=ohe, defect_only=True)
    testing_set = DAGMDataset(meta_file=TEST_LABEL_FILE_PATH, target_transform=ohe, defect_only=True)

    train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_set, batch_size=64, shuffle=True)

    train_features, train_labels, img_paths, has_defects, defect_mask_paths = next(iter(train_dataloader))

    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    image_path = img_paths[0]
    has_defect = has_defects[0]
    defect_mask_path = defect_mask_paths[0]
    plt.imshow(img, cmap="gray")
    plt.show()

    print(f"Label: {label}")
    print(f"Image path: {image_path}")
    print(f"Has defect: {has_defect}")
    print(f"Defect image path: {defect_mask_path}")


if __name__ == '__main__':
    run()

