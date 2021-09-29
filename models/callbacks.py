import numpy as np
import pytorch_lightning as pl
import wandb
from torch.utils.data import DataLoader

from models.utils import make_grid_with_keypoints


class ImageCallback(pl.Callback):
    def __init__(self, val_dataloader: DataLoader):
        super(ImageCallback, self).__init__()

        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        val_images = []
        val_labels = []
        val_pred = []

        pl_module.eval()
        for images, labels in self.val_dataloader:
            images = list(image for image in images)
            labels = [{k: v for k, v in t.items()} for t in labels]
            predictions = pl_module(images)

            val_images += images
            val_labels += [label['keypoints'] for label in labels]
            val_pred += [pred['keypoints'] for pred in predictions]

        grid = make_grid_with_keypoints(val_images, val_labels, val_pred)

        grid = np.transpose(grid.numpy(), axes=(1, 2, 0))

        trainer.logger.experiment.log({
            "val/predictions": wandb.Image(grid),
            'global_step': trainer.global_step
        })
