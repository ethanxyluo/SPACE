import os
import torch
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import numpy as np


class CIFAR10C(datasets.VisionDataset):
    def __init__(self, root: str, name: str,
                 transform=None, target_transform=None, level=1):
        # assert name in corruptions
        super(CIFAR10C, self).__init__(
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
        # img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            targets = self.target_transform(targets)

        return img, targets

    def __len__(self):
        return len(self.data)


test_transform = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                         0.557, 0.549, 0.5534])
                                     ])


def load_data(args):
    batch_size = args.batch_size
    dataset_root_dir = args.data
    if args.domain == 'cifar10':
        test_set = torchvision.datasets.CIFAR10(root=dataset_root_dir,
                                                train=False,
                                                transform=torchvision.transforms.Compose([
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                                        0.557, 0.549, 0.5534])
                                                ]),
                                                download=True)
    else:
        adapt_set = CIFAR10C(
            root='/path/to/cifar10/CIFAR-10-C', name=args.domain, level=5)
        test_set = CIFAR10C(root='/path/to/cifar10/CIFAR-10-C', name=args.domain,
                            transform=torchvision.transforms.Compose([
                                transforms.ToTensor(),
                                transforms.Normalize(mean=[0.4914, 0.4822, 0.4465], std=[
                                    0.557, 0.549, 0.5534])
                            ]), level=5)

    if args.phase == 'test':
        test_data_loader = torch.utils.data.DataLoader(
            dataset=test_set, batch_size=batch_size, shuffle=False, pin_memory=True, drop_last=False, num_workers=args.workers)

        adapt_set = None

    else:
        test_data_loader = None

    return test_data_loader, adapt_set
