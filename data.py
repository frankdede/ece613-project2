from torch.utils.data.dataset import Dataset


class DataLoader(Dataset):
    def __init__(self, annotation_file, img_dir, transform=None, target_transform=None):
        self.img_labels = {0: "crazing", 1: "inclusion", 2: "patches",
                           3: "pitted_surface", 4: "rolled-in_scale", 5: "scratches"}
        self.img_dir = pd.
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path =
