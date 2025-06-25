from datetime import datetime
from pathlib import Path

import requests
import torch
from joblib import Parallel, delayed
from omegaconf import DictConfig
from segmentation_models_pytorch.losses import DiceLoss
from torch import nn, optim
from torchmetrics import JaccardIndex


def is_empty(path: Path) -> bool:
    return not any(path.iterdir())


def root_dir() -> Path:
    return Path().absolute()


def download_image(url: str, path: Path) -> Path | None:
    try:
        resp = requests.get(url)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                f.write(resp.content)
            return path
        return None
    except Exception:
        return None


def download_images(urls_with_paths: list[tuple[str, Path]], n_jobs=8):
    results = Parallel(n_jobs=n_jobs, prefer="threads", verbose=5)(
        delayed(download_image)(url, filename)
        for url, filename in urls_with_paths
    )
    return results


def get_optimizer(model: nn.Module, config: DictConfig) -> optim.Optimizer:
    match config.optimizer.name:
        case "Adam":
            optimizer = optim.Adam
        case _:
            raise ValueError(
                f"Invalid optimizer name: {config.optimizer.name}"
            )
    optimizer = optimizer(
        model.parameters(), lr=config.optimizer.learning_rate
    )
    return optimizer


def get_loss(config: DictConfig) -> nn.Module:
    if len(config.loss.names) != len(config.loss.alphas):
        raise ValueError("Loss names and alphas must have the same length")
    losses = []
    for loss_name in config.loss.names:
        match loss_name:
            case "ce":
                losses.append(nn.CrossEntropyLoss())
            case "dice":
                losses.append(DiceLoss(**config.loss.dice_config))
            case _:
                raise ValueError(f"Invalid loss name: {loss_name}")

    def loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total_loss = 0
        for loss, alpha in zip(losses, config.loss.alphas, strict=False):
            loss_value = loss(pred, target)
            total_loss += alpha * loss_value
        return total_loss

    return loss


def get_metrics(config: DictConfig) -> dict[str, nn.Module]:
    metrics = {}
    for metric_name in config.metrics:
        match metric_name:
            case "jaccard":
                metrics[metric_name] = JaccardIndex(**config.jaccard).to(
                    config.device
                )
            case _:
                raise ValueError(f"Invalid metric name: {metric_name}")
    return metrics


def timestampt() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")
