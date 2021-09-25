import pytorch_lightning as pl
import torchvision.utils
import wandb

from torch.utils.data import DataLoader


class ImageCallback(pl.Callback):
    def __init__(self, val_dataloader: DataLoader, num_samples: int):
        super(ImageCallback, self).__init__()

        self.num_samples = num_samples
        self.val_dataloader = val_dataloader

    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        current_counter = 0
        log_images = []

        for batch in self.val_dataloader:
            images, labels = batch

            for image in images:
                if current_counter < self.num_samples:
                    log_images.append(image)
                    current_counter += 1

                else:
                    break

        grid = torchvision.utils.make_grid(log_images)

        trainer.logger.experiment.log({
            "examples": wandb.Image(grid),
            "global_step": trainer.global_step
        })
