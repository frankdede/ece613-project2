import os.path
import sys

import torch
from torch.utils.data import DataLoader

from ae.cmp_cnn import Net, test
from data import CnnDataset, DetectionModuleDataset
from settings import CNN_DICT, TEST_LABEL_FILE_PATH, MER_OUTPUT

if __name__ == '__main__':

    net = Net()
    net.load_state_dict(torch.load(CNN_DICT))
    net.eval()

    if sys.argv[1] == "dagm":
        test_set = CnnDataset(meta_file=TEST_LABEL_FILE_PATH)
    elif sys.argv[1] == "mer":
        assert len(sys.argv) == 4, "expect 3 arguments"
        model_stack_id = int(sys.argv[2])
        threshold = int(sys.argv[3])
        metafile_path = os.path.join(MER_OUTPUT, str(model_stack_id), f"meta-threshold_{threshold}.csv")
        test_set = DetectionModuleDataset(meta_file=metafile_path)
    else:
        raise ValueError("Unknown test dataset")

    test_dataloader = DataLoader(test_set, batch_size=1)
    test(net, test_dataloader)
