import os
from typing import List, cast

import optuna
import torch
import torchmetrics
from lightning.fabric import Fabric
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm


def train_model(
    model_type: str,
    max_epochs: int,
    threshold: float,
    trial: optuna.Trial,
    checkpoint_dir: str,
    train_dataloaders: List[DataLoader],
    test_dataloaders: List[DataLoader],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler,
    criterion: nn.Module,
):
    fabric = Fabric(
        accelerator="gpu",
        devices=1,
    )
    fabric.launch()

    metrics = torchmetrics.MetricCollection(
        {
            "dice": torchmetrics.Dice(threshold=threshold),
            "iou": torchmetrics.JaccardIndex(task="binary", threshold=threshold),
        }
    )

    model = fabric.setup_module(model)
    optimizer = fabric.setup_optimizers(optimizer)

    tracker = torchmetrics.MetricTracker(metrics, maximize=True)
    tracker.to(fabric.device)

    fabric.print(f"Started trial {trial.number} with params:")
    fabric.print(trial.params)

    for epoch in range(1, max_epochs + 1):
        fabric.print(f"Epoch ({epoch}/{max_epochs})")
        fabric.print("training")

        model.train()
        for train_dataloader in train_dataloaders:
            train_dataloader = fabric.setup_dataloaders(train_dataloader)
            pbar = tqdm(train_dataloader, disable=not fabric.is_global_zero)
            for image, target_mask in pbar:
                optimizer.zero_grad()
                predicted_mask = model(image)
                loss = criterion(predicted_mask, target_mask)
                fabric.backward(loss)
                optimizer.step()
                learning_rate = optimizer.param_groups[0]['lr']  # noqa
                pbar.set_description(f"train_loss={loss.item():.3f}, lr={learning_rate:.3f}")

        scheduler.step()
        model.eval()
        tracker.increment()
        for test_dataloader in test_dataloaders:
            test_dataloader = fabric.setup_dataloaders(test_dataloader)
            with torch.no_grad():
                fabric.print("validation")
                for image, mask in tqdm(test_dataloader, disable=not fabric.is_global_zero):
                    pred = model(image)
                    tracker.update(pred, mask.int())

        metrics = tracker.compute()
        print("Validation metrics", metrics)
        best_metric = tracker.best_metric()
        if metrics["dice"] == best_metric["dice"]:
            checkpoint = {
                "epoch": epoch,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "trial": trial,
                "metrics": metrics,
            }
            dice_score = metrics["dice"]
            path = os.path.join(checkpoint_dir, f"model={model_type}_{trial.number=}_{epoch=}_{dice_score=:.2f}.ckpt")
            fabric.save(path, checkpoint)

    return tracker.best_metric()["dice"]
