import pytorch_lightning as pl
import torch
import torchvision
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from torchvision.models.detection.keypoint_rcnn import KeypointRCNN, keypointrcnn_resnet50_fpn
import torchvision.models as models
from torchvision.models.detection.anchor_utils import AnchorGenerator
import torch.nn as nn

from omegaconf import DictConfig


class ChessKeypointDetection(pl.LightningModule):
    def __init__(self, cfg: DictConfig):
        super().__init__()
        self.cfg = cfg

        self.model = keypointrcnn_resnet50_fpn(num_classes=cfg.dataset.num_classes,
                                               num_keypoints=cfg.dataset.num_keypoints)
        self.loss = nn.SmoothL1Loss()

    def training_step(self, batch, batch_idx) -> float:
        images, labels = batch
        print(f"Len labels: {len(labels)}")
        for key, value in labels[0].items():
            print(f"key: {key},  size: {value.size()}")
        output = self.model(images, labels)
        loss = self.loss(output['boxes'], labels)
        self.log("train/smooth_l1_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch
        output = self.model(images)
        loss = self.loss(output['boxes'], labels)
        return {"val_loss": loss}

    def validation_epoch_end(self, outputs):
        # Add logging images (best and worst examples)
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log("val/smooth_l1_loss_epoch", avg_loss)
        return {'val_loss_epoch': avg_loss}

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=self.cfg.model.lr)
        scheduler = StepLR(optimizer=optimizer, step_size=1, gamma=self.cfg.model.lr_step)

        return [optimizer], [scheduler]
