from collections import defaultdict

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.keypoint_rcnn import keypointrcnn_resnet50_fpn


class ChessKeypointDetection(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = keypointrcnn_resnet50_fpn(num_classes=cfg.dataset.num_classes,
                                               num_keypoints=cfg.dataset.num_keypoints,
                                               trainable_backbone_layers=True)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        images = list(image for image in images)
        labels = [{k: v for k, v in t.items()} for t in labels]

        losses: dict = self.model(images, labels)
        for name, value in losses.items():
            self.log(f"train/{name}", value)

        return None

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        images = list(image for image in images)

        self.model.train()
        with torch.no_grad():
            losses = self.model(images, labels)

        return losses

    def validation_epoch_end(self, outputs):
        # Add logging images (best and worst examples)

        loss_types = outputs[0].keys()
        avg_losses = defaultdict(torch.tensor)

        for name in loss_types:
            avg_loss = torch.stack([x[name] for x in outputs if name in x]).mean()
            self.log(f"val/{name}_epoch", avg_loss)
            avg_losses[name] = avg_loss

        total_mean_loss = torch.stack([loss for loss in avg_losses.values()]).mean()
        self.log('val/val_epoch_total_loss', total_mean_loss)

        return {"val_epoch_total_loss": total_mean_loss}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=self.cfg.model.lr)
        scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=self.cfg.model.lr_step)

        return [optimizer], [scheduler]
