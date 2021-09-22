# Chess keypoint detection

This repository contains source code of chess keypoint detection task.

## Link to colab example: [Google Colab](https://colab.research.google.com/drive/1TK1RDZjQB0jLCdSBGc90huPtYx6d6qwU?usp=sharing)

## Used technologies

- pytorch/pytorch-lightning for data extraction and training loop
- torchvision for keypoint detection model
- hydra for configuration model/data
- wandb for logging images/losses (optional)

## Visualisation of train data

![](jupyter-notebooks/train_data.png)

## Launch training

Download data from [Google Disk](https://drive.google.com/drive/folders/1-Fr_RzLVOTr7znADuxoricqlXy8OzGT6?usp=sharing)
and put it into ```data/``` folder.

Run in console:

```bash
python -m pip install -r requirements.txt
python train.py (or python train.py path/to/hygra/config.yaml)
```

## The structure of repository

- ```config/config.yaml``` - config, used for training model (lr, batch-size, etc.)
- ```data/``` - data folder, used for storing data (should contain ```xtest.npy  xtrain.npy  ytrain.npy``` files for
  training)
- ```models/```
  - ```dataloader.py and dataset.py``` - creating dataset and dataloader (overloading
    torch.utils.data.Dataset/Dataloader)
  - ```trainer.py``` - lighting training/testing loop
- ```jupyter-notebooks/``` - visualizing and training/testing notebooks

- ```train.py``` - training script fot

## Results of training

to be added

## Contributions

Contributions are welcome, please open PR and describe the implemented functionality.



