import hydra
import pytorch_lightning as pl
import wandb
from omegaconf import DictConfig
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger

from models.dataloader import ChessDataloader
from models.trainer import ChessKeypointDetection

pl.seed_everything(0)


@hydra.main(config_path="config", config_name="config")
def train(cfg: DictConfig):
    print("Source path:", hydra.utils.get_original_cwd())
    loader = ChessDataloader(folder=cfg.dataset.path,
                             batch_size=cfg.dataset.batch_size,
                             train_size=cfg.dataset.train_size)

    model = ChessKeypointDetection(cfg)

    logger = WandbLogger(project="TmpChessKeypointDetection", name="tmp_initial_experiment", log_model=True)

    logger.log_hyperparams(dict(cfg))
    logger.save()

    model_checkpoint = ModelCheckpoint(dirpath=cfg.logging.weights_path,
                                       save_last=False,
                                       save_top_k=1,
                                       monitor='val/val_epoch_total_loss',
                                       mode='min',
                                       filename="chess_epoch={epoch:0.2f}_val_loss={val/smooth_l1_loss_epoch}")
    lr_monitor = LearningRateMonitor(logging_interval='step')

    trainer = pl.Trainer(
        log_gpu_memory='all',
        logger=logger,
        callbacks=[model_checkpoint, lr_monitor],
        max_epochs=cfg.model.epochs,
        gpus=1,
    )

    trainer.fit(model, loader.train_dataloader(), loader.val_dataloader())


if __name__ == '__main__':
    wandb.login()
    train()
