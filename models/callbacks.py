import numpy as np
import pytorch_lightning as pl
import torch
import wandb
from torch.utils.data import DataLoader

from models.utils import make_grid_with_keypoints


class ImageCallback(pl.Callback):
    def __init__(self, val_dataloader: DataLoader, score_threshold: float):
        super(ImageCallback, self).__init__()

        self.val_dataloader = val_dataloader
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_images = []
        val_labels = []
        val_pred = []

        pl_module.eval()
        for images, labels in self.val_dataloader:
            inputs = list(image.to(pl_module.device) for image in images)
            labels = [{k: v for k, v in t.items()} for t in labels]
            predictions = pl_module(inputs)

            val_images += images
            val_labels += [label['keypoints'] for label in labels]
            val_pred += [pred['keypoints'][pred['scores'] > self.score_threshold].cpu() for pred in predictions]

        grid = make_grid_with_keypoints(val_images, val_labels, val_pred)

        grid = np.transpose(grid.numpy(), axes=(1, 2, 0))

        trainer.logger.experiment.log({
            "val/predictions": wandb.Image(grid, caption="Green is labels, red is predicted keypoints"),
            'global_step': trainer.global_step
        })


class ImageTestCallback(pl.Callback):
    def __init__(self, test_dataloader: DataLoader, score_threshold: float):
        super(ImageTestCallback, self).__init__()

        self.test_dataloader = test_dataloader
        self.score_threshold = score_threshold

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        test_images = []
        test_pred = []

        pl_module.eval()
        for images in self.test_dataloader:
            inputs = list(image.to(pl_module.device) for image in images)
            predictions = pl_module(inputs)

            test_images += images
            test_pred += [pred['keypoints'][pred['scores'] > self.score_threshold].cpu() for pred in predictions]

        empty_labels = [torch.empty(0, 2, 3, dtype=torch.float32) for _ in range(len(test_images))]

        grid = make_grid_with_keypoints(test_images, empty_labels, test_pred)
        grid = np.transpose(grid.numpy(), axes=(1, 2, 0))

        trainer.logger.experiment.log({
            "test/predictions": wandb.Image(grid, caption="Green is labels, red is predicted keypoints"),
            'global_step': trainer.global_step
        })
