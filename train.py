import os

import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.callbacks import ImageCallback
from models.dataloader import ChessDataloader
from models.trainer import ChessKeypointDetection

pl.seed_everything(0)


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    source_folder = hydra.utils.get_original_cwd()
    dataset_folder = os.path.join(source_folder, cfg.dataset.path)
    loader = ChessDataloader(folder=dataset_folder,
                             batch_size=cfg.dataset.batch_size,
                             train_size=cfg.dataset.train_size)

    model = ChessKeypointDetection(cfg)

    logger = WandbLogger(project="TmpChessKeypointDetection",
                         name="tmp_initial_experiment",
                         log_model=True,
                         config=cfg, )

    model_checkpoint = ModelCheckpoint(dirpath=cfg.logging.weights_path,
                                       save_last=False,
                                       save_top_k=1,
                                       monitor='val/val_epoch_total_loss',
                                       mode='min',
                                       filename="chess_epoch={epoch:0.2f}_val_loss={val/val_epoch_total_loss}",
                                       )
    lr_monitor = LearningRateMonitor(logging_interval='step')

    image_callback = ImageCallback(val_dataloader=loader.val_dataloader(), num_samples=5)

    trainer = pl.Trainer(
        log_gpu_memory='all',
        logger=logger,
        callbacks=[model_checkpoint, lr_monitor, image_callback],
        max_epochs=cfg.model.epochs,
        gpus=cfg.gpus,
        deterministic=True
    )

    trainer.fit(model, loader.train_dataloader(), loader.val_dataloader())


if __name__ == '__main__':
    wandb.login()
    train()
