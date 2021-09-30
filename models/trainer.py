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

        self.model = keypointrcnn_resnet50_fpn(pretrained=False,
                                               num_classes=cfg.dataset.num_classes,
                                               num_keypoints=cfg.dataset.num_keypoints,
                                               pretrained_backbone=False,
                                               trainable_backbone_layers=5)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        images = list(image for image in images)
        labels = [{k: v for k, v in t.items()} for t in labels]

        losses: dict = self.model(images, labels)
        for name, value in losses.items():
            self.log(f"train/{name}", value)

        loss = torch.hstack(list(losses.values())).mean()

        self.log("train/train_mean_loss", loss)
        return {'loss': loss}

    def training_epoch_end(self, outputs: dict) -> None:
        avg_loss = torch.hstack([x['loss'] for x in outputs]).mean()
        self.log("train/train_epoch_loss", avg_loss)
        return None

    def validation_step(self, *args, **kwargs):
        return None

    def validation_epoch_end(self, outputs):
        return None

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        params = [p for p in self.parameters() if p.requires_grad]
        optimizer = Adam(params, lr=self.cfg.model.lr)
        scheduler = StepLR(optimizer=optimizer, step_size=10, gamma=self.cfg.model.lr_step)

        return [optimizer], [scheduler]
