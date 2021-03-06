import os

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

import wandb
from models.callbacks import ImageCallback, ImageTestCallback
from models.dataloader import ChessDataloader
from models.trainer import ChessKeypointDetection

pl.seed_everything(0)


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    source_folder = hydra.utils.get_original_cwd()
    print(f"saving folder: {source_folder}")
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
                                       save_last=True,
                                       save_top_k=1,
                                       monitor='train/train_epoch_loss',
                                       mode='min',
                                       filename="chess_epoch={epoch:0.2f}_val_loss={train/train_epoch_loss:0.2f}",
                                       save_weights_only=True,
                                       )
    lr_monitor = LearningRateMonitor(logging_interval='step')
    val_image_callback = ImageCallback(val_dataloader=loader.val_dataloader(),
                                       score_threshold=cfg.threshold.score)

    test_image_callback = ImageTestCallback(test_dataloader=loader.test_dataloader(),
                                            score_threshold=cfg.threshold.score)

    last_ckpt = os.path.join(cfg.logging.weights_path, 'last.ckpt')

    if os.path.isfile(last_ckpt):
        trainer = pl.Trainer(
            log_gpu_memory='all',
            logger=logger,
            callbacks=[model_checkpoint, lr_monitor, val_image_callback, test_image_callback],
            max_epochs=cfg.model.epochs,
            gpus=cfg.gpus,
            deterministic=True,
            resume_from_checkpoint=last_ckpt,
        )

    else:
        trainer = pl.Trainer(
            log_gpu_memory='all',
            logger=logger,
            callbacks=[model_checkpoint, lr_monitor, val_image_callback, test_image_callback],
            max_epochs=cfg.model.epochs,
            gpus=cfg.gpus,
            deterministic=True,
        )

    trainer.fit(model, loader.train_dataloader(), loader.val_dataloader())


if __name__ == '__main__':
    wandb.login()
    train()
