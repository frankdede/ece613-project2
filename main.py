import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from data import DAGMDataset
from settings import TRAINING_LABEL_FILE_PATH, TEST_LABEL_FILE_PATH
import matplotlib.pyplot as plt


def run():
    num_classes = 10
    ohe = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))
    training_set = DAGMDataset(meta_file=TRAINING_LABEL_FILE_PATH, target_transform=ohe)
    testing_set = DAGMDataset(meta_file=TEST_LABEL_FILE_PATH, target_transform=ohe)
    train_dataloader = DataLoader(training_set, batch_size=64, shuffle=True)
    test_dataloader = DataLoader(testing_set, batch_size=64, shuffle=True)

    train_features, train_labels = next(iter(train_dataloader))
    print(type(train_features))
    print(type(train_labels))
    print(f"Feature batch shape: {train_features.size()}")
    print(f"Labels batch shape: {train_labels.size()}")
    img = train_features[0].squeeze()
    label = train_labels[0]
    plt.imshow(img, cmap="gray")
    plt.show()
    print(f"Label: {label}")


if __name__ == '__main__':
    run()

