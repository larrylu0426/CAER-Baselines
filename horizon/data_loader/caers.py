import os

import cv2
import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import Dataset

from horizon.base.data_loader import BaseDataLoader

class BaseDataset(Dataset):

    def __init__(self, root: str, phase: str) -> None:
        self.root = root
        self.phase = phase
        self._load_data()
        self.transform = self._transform()

    def _load_data(self):
        self.face = np.load(
            os.path.join(self.root, self.phase + "_face_arr" + ".npy"))
        self.context = np.load(
            os.path.join(self.root, self.phase + "_context_arr" + ".npy"))
        self.image = np.load(
            os.path.join(self.root, self.phase + "_img_arr" + ".npy"))
        self.bbox = np.load(
            os.path.join(self.root, self.phase + "_bbox_arr" + ".npy"))
        self.label = np.load(
            os.path.join(self.root, self.phase + "_label_arr" + ".npy"))

    def _transform(self):
        if self.phase == "train":
            return transforms.Compose([
                self.ToPILImage(),
                self.Crop(112, self.phase),
                self.Augmentation(),
                self.Normalize()
            ])
        else:
            return transforms.Compose([
                self.ToPILImage(),
                self.Crop(112, self.phase),
                self.Normalize()
            ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        if idx >= len(self.label):
            raise IndexError('Index out of bound')
        data = {
            'face': self.face[idx],
            'context': self.context[idx],
            'image': self.image[idx],
            'bbox': self.bbox[idx]
        }
        label = torch.tensor(self.label[idx], dtype=torch.int64)
        if self.transform:
            data = self.transform(data)
        return data, label

    class ToPILImage(object):

        def __init__(self):
            self.transforms = transforms.ToPILImage()

        def __call__(self, sample):
            face, context, image, bbox = sample['face'], sample[
                'context'], sample['image'], sample['bbox']
            return {
                'face':
                self.transforms(cv2.cvtColor(face, cv2.COLOR_BGR2RGB)),
                'context':
                self.transforms(cv2.cvtColor(context, cv2.COLOR_BGR2RGB)),
                'image':
                self.transforms(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                'bbox':
                bbox
            }

    class Crop(object):
        """
        (Randomly) crop context region

        Args:
        size (int): context region size
        mode (string): takes value "train" or "test". If "train", use random crop.
                        If "test", use center crop.
        """

        def __init__(self, size, mode="train"):
            self.size = size
            self.mode = mode

        def __call__(self, sample):
            context = sample['context']
            if self.mode == "train":
                context = transforms.RandomCrop(self.size)(context)
            else:
                context = transforms.CenterCrop(self.size)(context)

            return {
                'face': sample['face'],
                'context': context,
                'image': sample['image'],
                'bbox': sample['bbox']
            }

    class Augmentation(object):

        def __init__(self):
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.0,
                                       contrast=0.5,
                                       saturation=0.0,
                                       hue=0.5),
            ])

        def __call__(self, sample):
            face, context, image, bbox = sample['face'], sample[
                'context'], sample['image'], sample['bbox']
            return {
                'face': face,
                'context': self.transforms(context),
                'image': image,
                'bbox': bbox
            }

    class Normalize(object):

        def __call__(self, sample):
            face = sample['face'].convert("RGB")
            context = sample['context'].convert("RGB")
            image = sample['image'].convert("RGB")
            toTensor = transforms.ToTensor()
            normalize = transforms.Normalize([0.485, 0.456, 0.406],
                                             [0.229, 0.224, 0.225])
            return {
                'face': normalize(toTensor(face)),
                'context': normalize(toTensor(context)),
                'image': normalize(toTensor(image)),
                'bbox': sample['bbox']
            }


class CAERS(BaseDataLoader):

    def __init__(self,
                 root,
                 batch_size,
                 phase,
                 shuffle=False,
                 val_split=0.0,
                 test_split=0.0,
                 num_workers=1):
        self.dataset = BaseDataset(root, phase)
        super().__init__(self.dataset, batch_size, shuffle, val_split,
                         test_split, num_workers)
