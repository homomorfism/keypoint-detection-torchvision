from pathlib import Path

import numpy as np
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def xyxyn2xyxy(original_size: torch.Size, coords: torch.tensor):
    channels, height, width = original_size

    new_coords = torch.empty_like(coords, dtype=torch.int)

    new_coords[:, 0] = (coords[:, 0] * height).type(torch.int)
    new_coords[:, 1] = (coords[:, 1] * width).type(torch.int)

    return new_coords


def xyxy2torchvision(coords: torch.tensor):
    new_coords = torch.empty(size=[coords.size()[0], 3])

    new_coords[:, 0] = coords[:, 0]
    new_coords[:, 1] = coords[:, 1]
    new_coords[:, 2] = 1

    return new_coords


class ChessDataset(Dataset):
    def __init__(self, root_folder: Path, flag: str = "train", transform=None):
        assert flag in ['train', 'test']

        super().__init__()
        self.transform = transform
        self.flag = flag

        if self.flag == 'train':
            self.images = np.load(file=str(root_folder / "xtrain.npy"))
            self.labels = np.load(file=str(root_folder / 'ytrain.npy'))

        else:
            self.images = np.load(file=str(root_folder / 'xtest.npy'))

    def __getitem__(self, item):
        x = self.images[item]

        x = self.transform(x) if self.transform else transforms.ToTensor()(x)

        if self.flag == 'train':
            # Format of y is (num_instances, num_keypoints_in_instance, 3)
            # 3 means that each keypoint should be represented in this form: [x, y, visibility],
            # where visibility = 0 means that keypoint is not visible (visible - 1)
            y = self.labels[item]
            y = torch.as_tensor(y).reshape(-1, 2)
            y = xyxyn2xyxy(x.size(), y)

            boxes = torch.as_tensor([torch.min(y[:, 0]),
                                     torch.min(y[:, 1]),
                                     torch.max(y[:, 0]),
                                     torch.max(y[:, 1])])

            points = xyxy2torchvision(y)
            output = []

            for ii in range(len(boxes)):
                output.append({
                    "boxes": boxes[ii],
                    "labels": torch.as_tensor(0),
                    "points": points[ii]
                })
            return x, output

        else:
            return x

    def __len__(self):
        return len(self.images)


if __name__ == '__main__':
    train_dataset = ChessDataset(root_folder=Path("../data"), flag='train')
    image, labels = train_dataset[0]

    image = image.permute(1, 2, 0).numpy()
    labels = labels['boxes'].squeeze()[:, :2]

    plt.imshow(image)
    plt.scatter(labels[:, 0], labels[:, 1], c='r')

    plt.show()
