import os
from argparse import ArgumentParser
from typing import List

import optuna
import torch
import torchvision.transforms as tvt
import yaml
from pydantic import BaseSettings
from torch import nn

from data import make_loaders, read_manifest, get_loader
from loss import DiceLoss, TverskyLoss
from models import Unet3d, Unet2d
from train_loop import train_model
from transforms import Resize3DImageMask, Resize2DImageMask, ComposeImageMask, Normalize


class Dataset(BaseSettings):
    name: str
    type: str = "3d"
    loader_name: str
    image_key: str
    mask_key: str


class KaggleDataset(Dataset):
    kaggle_path: str


def get_dataset(dataset) -> Dataset:
    if dataset["type"] == "KaggleDataset":
        return KaggleDataset(**dataset["params"])
    elif dataset["type"] == "Dataset":
        return Dataset(**dataset["params"])
    else:
        raise ValueError(f"Unknown type {dataset['type']}")


class Params(BaseSettings):
    img_size = [512, 512]
    model_type = "unet2d"
    optim_type = {"name": "optim_type", "choices": ["Adam"]}
    loss_type = {"name": "loss_type", "choices": [
        "DiceLoss",
        "TverskyLoss",
    ]}
    hidden_dim = {"name": "hidden_dim", "low": 64, "high": 64, "log": True}
    learning_rate = {"name": "learning_rate", "low": 1e-4, "high": 1e-3, "log": True}
    weight_decay = {"name": "weight_decay", "low": 1e-5, "high": 0.1, "log": True}
    lr_decay = {"name": "lr_decay", "low": 1e-5, "high": 0.1, "log": True}


class Config(BaseSettings):
    num_workers: int = 2
    batch_size: int = 8
    random_state: int = 42
    test_size: float = 0.1

    n_trials: int = 1
    max_epochs: int = 10
    threshold: float = 0.5

    kaggle_username: str = None
    kaggle_api_key: str = None
    name: str = "learning_rate_search"
    data_dir: str = "./data"
    data_dir_2d: str = "./data2d"
    checkpoints_dir: str = "./checkpoints"
    logs_dir: str = "./logs"
    params: Params = Params()

    datasets: List[dict] = [
        {
            "type": "KaggleDataset",
            "params": dict(
                type="2d",
                name="covid19-ct-scans",
                kaggle_path="andrewmvd/covid19-ct-scans",
                loader_name="TIFFLoader",
                image_key="ct_scan",
                mask_key="infection_mask",
            )
        },
    ]


def main():
    parser = ArgumentParser()
    parser.add_argument("-u", "--kaggle-username", type=str, required=False)
    parser.add_argument("-k", "--kaggle-api-key", type=str, required=False)
    parser.add_argument("-c", "--config-file", type=str, required=False)
    console_args = parser.parse_args().__dict__

    config_file = console_args.pop("config_file")
    if config_file:
        with open(config_file, "r") as file:
            yaml_config = yaml.safe_load(file)
    else:
        yaml_config = {}

    config = {**yaml_config, **console_args}
    config = Config(**config)

    checkpoint_dir = create_dir(config.checkpoints_dir, config.name)
    logs_dir = create_dir(config.logs_dir, config.name)

    if config.params.model_type == "unet2d":
        transforms = {
            "train": ComposeImageMask([
                Resize2DImageMask(config.params.img_size, interpolation=tvt.InterpolationMode.NEAREST),
                Normalize(),
                # (tvt.RandomVerticalFlip(1), 0.5),
                # (tvt.RandomHorizontalFlip(1), 0.2),
            ]),
            "test": ComposeImageMask([
                Resize2DImageMask(config.params.img_size, interpolation=tvt.InterpolationMode.NEAREST),
                Normalize(),
            ])
        }
    elif config.params.model_type == "unet3d":
        transforms = {
            "train": ComposeImageMask([
                Resize3DImageMask(config.params.img_size, interpolation=tvt.InterpolationMode.NEAREST),
            ]),
            "test": ComposeImageMask([
                Resize3DImageMask(config.params.img_size, interpolation=tvt.InterpolationMode.NEAREST),
            ]),
        }
    else:
        raise ValueError("Unknown model type.")

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    def objective(trial: optuna.Trial) -> float:
        train_dataloaders = []
        test_dataloaders = []

        for dataset in config.datasets:
            dataset = get_dataset(dataset)
            loader = get_loader(dataset.loader_name)

            path = os.path.join(config.data_dir, dataset.name)
            os.makedirs(path, exist_ok=True)

            if config.params.model_type.endswith("2d"):
                path_2d = os.path.join(config.data_dir_2d, dataset.name)
                items = read_manifest(path_2d)
                loader_func = loader.load_2d_image
            else:
                items = read_manifest(path)
                loader_func = loader.load_3d_image

            train_dataloader, test_dataloader = make_loaders(config, items, transforms, dataset.image_key,
                                                             dataset.mask_key, loader_func)
            train_dataloaders.append(train_dataloader)
            test_dataloaders.append(test_dataloader)

        hidden_dim = trial.suggest_int(**config.params.hidden_dim)

        if config.params.model_type == "unet3d":
            model = Unet3d(hidden_dim)
        elif config.params.model_type == "unet2d":
            model = Unet2d(hidden_dim)
        else:
            raise ValueError(f"no such model {config.params.model_type}")

        loss_type = trial.suggest_categorical(**config.params.loss_type)

        if loss_type == "BCELoss":
            criterion = nn.BCELoss()
        elif loss_type == "DiceLoss":
            criterion = DiceLoss()
        elif loss_type == "TverskyLoss":
            criterion = TverskyLoss()
        else:
            raise ValueError(f"no such loss {loss_type}")

        optim_type = trial.suggest_categorical(**config.params.optim_type)
        learning_rate = trial.suggest_float(**config.params.learning_rate)

        if optim_type == "Adam":
            optimizer = torch.optim.Adam(model.parameters(), learning_rate)
        elif optim_type == "Adagrad":
            weight_decay = trial.suggest_float(**config.params.weight_decay)
            lr_decay = trial.suggest_float(**config.params.lr_decay)
            optimizer = torch.optim.Adagrad(model.parameters(), learning_rate, lr_decay, weight_decay)
        else:
            raise ValueError(f"no such optimizer {optim_type}")

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_epochs, eta_min=1e-6)

        score = train_model(
            config.params.model_type,
            config.max_epochs,
            config.threshold,
            trial,
            checkpoint_dir,
            train_dataloaders,
            test_dataloaders,
            model,
            optimizer,
            scheduler,
            criterion,
        )
        return score

    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=config.n_trials)


def create_dir(dirname, target_name):
    checkpoint_dir = os.path.join(dirname, target_name)
    if os.path.exists(checkpoint_dir):
        max_num = max(int(direct.replace(target_name, "").strip("_") or 0)
                      for direct in os.listdir(dirname) if direct.startswith(target_name))
        checkpoint_dir += f'_{max_num + 1}'
    return checkpoint_dir


if __name__ == "__main__":
    main()
