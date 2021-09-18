from pytorch_lightning.loggers import WandbLogger
import hydra
import pytorch_lightning as pl
from models.dataloader import ChessDataloader
from models.trainer import ChessKeypointDetection
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    print("Source path:", hydra.utils.get_original_cwd())
    loader = ChessDataloader(folder=cfg.dataset.path,
                             batch_size=cfg.dataset.batch_size,
                             train_size=cfg.dataset.train_size)

    model = ChessKeypointDetection(cfg)

    logger = WandbLogger(project="ChessKeypointDetection", log_model=True, **cfg)

    model_checkpoint = ModelCheckpoint(dirpath=cfg.logging.weights_path,
                                       save_last=False,
                                       save_top_k=1,
                                       monitor='val_loss_epoch',
                                       mode='min',
                                       filename="chess_epoch={epoch:0.2f}_val_loss={val/smooth_l1_loss_epoch}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=cfg.model.epochs,
        fast_dev_run=True,
        gpus=0)

    trainer.fit(model, loader.train_dataloader(), loader.val_dataloader())


if __name__ == '__main__':
    train()
