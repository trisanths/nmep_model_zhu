from __future__ import annotations

import h5py
import torch
from PIL import Image, ImageFile
from torch.utils.data import Dataset, Subset
from torchvision import transforms
from torchvision.datasets import CIFAR10, ImageFolder

ImageFile.LOAD_TRUNCATED_IMAGES = True


class MediumImagenetHDF5Dataset(Dataset):
    def __init__(
        self,
        img_size,
        split: str = "train",
        filepath: str = "/data/nmep/cv/medium-imagenet-96.hdf5",
        augment: bool = True,
    ):
        assert split in ["train", "val", "test"]
        self.split = split
        self.augment = augment
        self.input_size = img_size
        self.transform = self._get_transforms()
        self.file = h5py.File(filepath, "r")

    def __getitem__(self, index):
        image = self.file[f"images-{self.split}"][index]
        if self.split != "test":
            label = self.file[f"labels-{self.split}"][index]
        else:
            label = -1
        image = self.transform(image)
        label = torch.tensor(label, dtype=torch.long)
        return image, label

    def __len__(self):
        return len(self.file[f"images-{self.split}"])

    def _get_transforms(self):
        transform = []
        transform.append(lambda x: torch.tensor(x / 256).to(torch.float))
        normalization = torch.Tensor([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
        transform.append(transforms.Normalize(normalization[0], normalization[1]))
        transform.append(transforms.Resize([self.input_size] * 2))
        if self.split == "train" and self.augment:
            transform.extend(
                [
                    transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                ]
            )
        return transforms.Compose(transform)


class CIFAR10Dataset(Dataset):
    MEAN, STD = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)

    def __init__(self, img_size=32, train=True, root="/data/nmep/cv/cifar10", scheme="default"):
        self.train = train
        self.img_size = img_size
        self.scheme = scheme

        self.transform = self._get_transforms()
        self.dataset = CIFAR10(root=root, train=self.train, download=True)

    def __getitem__(self, index):
        image, label = self.dataset[index]
        image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.dataset)

    def _get_transforms(self):
        to_tensor = [
            transforms.ToTensor(),
            transforms.Normalize(self.MEAN, self.STD),
            transforms.Resize([self.img_size] * 2),
        ]
        if not self.train or self.scheme == "none":
            return transforms.Compose(to_tensor)
        if self.scheme == "default":
            aug = [
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
                transforms.RandomHorizontalFlip(),
            ]
        elif self.scheme == "crop_flip":
            aug = [transforms.RandomCrop(32, padding=4, padding_mode="reflect"), transforms.RandomHorizontalFlip()]
        elif self.scheme == "trivial_erase":
            aug = [transforms.TrivialAugmentWide(), transforms.RandomHorizontalFlip()]
            return transforms.Compose(aug + to_tensor + [transforms.RandomErasing(p=0.25)])
        else:
            raise ValueError(f"unknown augmentation scheme {self.scheme}")
        return transforms.Compose(aug + to_tensor)
