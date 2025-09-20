import os
import torchvision.datasets as datasets
from PIL import Image
import numpy as np


class CIFAR100C(datasets.VisionDataset):
    def __init__(self, root: str, name: str,
                 transform=None, target_transform=None, level=1):
        # assert name in corruptions
        super(CIFAR100C, self).__init__(
            root, transform=transform,
            target_transform=target_transform)
        data_path = os.path.join(root, name + '.npy')
        target_path = os.path.join(root, 'labels.npy')

        data = np.load(data_path)
        targets = np.load(target_path)

        if level == 0:
            self.data = data
            self.targets = targets
        else:
            start = (level - 1) * 10000
            self.data = data[start:start + 10000]
            self.targets = targets[start:start + 10000]

    def __getitem__(self, index):
        img, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            img = Image.fromarray(img)
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)
