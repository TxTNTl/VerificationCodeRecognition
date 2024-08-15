import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import config


def label_transform_tensor(label: str):
    num_classes = len(config.MAIN_SET['characters'])
    index = [config.MAIN_SET['characters'].index(c) for c in label]
    one_hot = torch.nn.functional.one_hot(torch.tensor(index), num_classes)
    one_hot = one_hot.view(len(index) * num_classes).float()
    return one_hot


def tensor_transform_label(one_hot: torch.Tensor):
    num_classes = len(config.MAIN_SET['characters'])
    one_hot = one_hot.view(-1, num_classes)
    _, predicted = torch.max(one_hot, dim=1)
    label = str(''.join([config.MAIN_SET['characters'][i] for i in predicted]))
    return label


class CustomDataset(Dataset):
    def __init__(self, image_dir, label_file, transform=None):
        self.image_dir = image_dir
        self.label_file = label_file
        self.transform = transform

        with open(label_file, 'r') as f:
            self.labels = [label_transform_tensor(line.strip()) for line in f.readlines()]

        self.image_paths = [os.path.join(self.image_dir, f"{i:05}.png") for i in range(len(self.labels))]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        image = Image.open(img_path).convert('RGB')

        if self.transform:
            image = self.transform(image)

        return image, label
