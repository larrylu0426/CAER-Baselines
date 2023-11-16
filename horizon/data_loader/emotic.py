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
        self.body = np.load(
            os.path.join(self.root, self.phase + "_body_arr" + ".npy"))
        self.context = np.load(
            os.path.join(self.root, self.phase + "_context_arr" + ".npy"))
        self.image = np.load(
            os.path.join(self.root, self.phase + "_img_arr" + ".npy"))
        self.bbox = np.load(
            os.path.join(self.root, self.phase + "_bbox_arr" + ".npy"))
        self.cat_label = np.load(
            os.path.join(self.root, self.phase + "_cat_label_arr" + ".npy"))
        self.cont_label = np.load(
            os.path.join(self.root, self.phase + "_cont_label_arr" + ".npy"))

    def _transform(self):
        if self.phase == "train":
            return transforms.Compose(
                [self.ToPILImage(),
                 self.Augmentation(),
                 self.Normalize()])
        else:
            return transforms.Compose([self.ToPILImage(), self.Normalize()])

    def __len__(self):
        return len(self.image)

    def __getitem__(self, idx):
        if idx >= len(self.image):
            raise IndexError('Index out of bound')
        data = {
            'body': self.body[idx],
            'context': self.context[idx],
            'image': self.image[idx],
            'bbox': self.bbox[idx]
        }
        label = {
            "cat": torch.tensor(self.cat_label[idx], dtype=torch.int64),
            "cont": torch.tensor(self.cont_label[idx], dtype=torch.float32)
        }
        if self.transform:
            data = self.transform(data)
        return data, label

    class ToPILImage(object):

        def __init__(self):
            self.transforms = transforms.ToPILImage()

        def __call__(self, sample):
            body, context, image, bbox = sample['body'], sample[
                'context'], sample['image'], sample['bbox']
            return {
                'body':
                self.transforms(cv2.cvtColor(body, cv2.COLOR_BGR2RGB)),
                'context':
                self.transforms(cv2.cvtColor(context, cv2.COLOR_BGR2RGB)),
                'image':
                self.transforms(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)),
                'bbox':
                bbox
            }

    class Augmentation(object):

        def __init__(self):
            self.transforms = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.4,
                                       contrast=0.4,
                                       saturation=0.4)
            ])

        def __call__(self, sample):
            body, context, image, bbox = sample['body'], sample[
                'context'], sample['image'], sample['bbox']
            return {
                'body': self.transforms(body),
                'context': self.transforms(context),
                'image': self.transforms(image),
                'bbox': bbox
            }

    class Normalize(object):

        def __call__(self, sample):
            body = sample['body'].convert("RGB")
            context = sample['context'].convert("RGB")
            image = sample['image'].convert("RGB")
            to_tensor = transforms.ToTensor()
            img_norm = transforms.Normalize(
                [0.4690646, 0.4407227, 0.40508908],
                [0.2514227, 0.24312855, 0.24266963])
            body_norm = transforms.Normalize(
                [0.43832874, 0.3964344, 0.3706214],
                [0.24784276, 0.23621225, 0.2323653])
            return {
                'body': body_norm(to_tensor(body)),
                'context': img_norm(to_tensor(context)),
                'image': img_norm(to_tensor(image)),
                'bbox': sample['bbox']
            }


class EMOTIC(BaseDataLoader):

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
