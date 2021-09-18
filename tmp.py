import hydra
from omegaconf import DictConfig


@hydra.main(config_path="config/", config_name="config")
def train(cfg: DictConfig):
    print(cfg)


if __name__ == '__main__':
    train()

