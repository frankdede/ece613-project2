import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Lambda

from data import CnnDataset
from settings import TRAINING_LABEL_FILE_PATH, CNN_DICT
from ae.cmp_cnn import train

num_classes = 10
ohe = Lambda(lambda y: torch.zeros(num_classes, dtype=torch.float).scatter_(dim=0, index=torch.tensor(y), value=1))


def run():
    training_set = CnnDataset(meta_file=TRAINING_LABEL_FILE_PATH)
    train_dataloader = DataLoader(training_set, batch_size=30, shuffle=True)

    net = train(train_dataloader, 30)
    torch.save(net.state_dict(), CNN_DICT)


if __name__ == '__main__':
    run()

