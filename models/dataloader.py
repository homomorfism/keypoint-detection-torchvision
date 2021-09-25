from pathlib import Path

import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader, random_split

from models.dataset import ChessDataset


def collate_fn(batch):
    data_list, label_list = [], []

    for data, label in batch:
        data_list.append(data)
        label_list.append(label)

    return data_list, label_list



class ChessDataloader(pl.LightningDataModule):

    def __init__(self, folder: str, batch_size: int, train_size: float):
        super().__init__()
        folder = Path(folder)
        assert folder.is_dir(), f"{str(folder.resolve())} is not dir!"
        self.batch_size = batch_size
        self.train_size = train_size

        self.train_dataset = ChessDataset(root_folder=folder, flag='train')
        self.test_dataset = ChessDataset(root_folder=folder, flag='test')

    def train_dataloader(self):
        dataset_length = len(self.train_dataset)
        num_training = int(dataset_length * self.train_size)

        train_dataset, _ = random_split(self.train_dataset,
                                        lengths=[num_training, dataset_length - num_training],
                                        generator=torch.Generator().manual_seed(0))

        return DataLoader(dataset=train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=4,
                          collate_fn=collate_fn)

    def val_dataloader(self):
        dataset_length = len(self.train_dataset)
        num_training = int(dataset_length * self.train_size)

        _, val_dataset = random_split(self.train_dataset,
                                      lengths=[num_training, dataset_length - num_training],
                                      generator=torch.Generator().manual_seed(0))

        return DataLoader(dataset=val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          collate_fn=collate_fn)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=2,
                          collate_fn=collate_fn)
