from pathlib import Path

import torch
from loguru import logger
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter

from chest_segment.checkpoint_saver import (
    CheckpointSaver,
    CheckpointSaverConfig,
)


class TrainConfig(BaseModel):
    num_epochs: int
    device: str = "cuda"
    log_dir: Path = Path("logs")
    checkpoint_dir: Path = Path("checkpoints")


def train_epoch(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
) -> dict[str, float]:
    model.train()
    total_loss = 0.0
    num_batches = len(train_loader)
    for batch_idx, (images, masks) in enumerate(train_loader):
        images = images.to(device)
        masks = masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        if batch_idx % 10 == 0:
            logger.info(
                f"Epoch {epoch}, Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}"
            )
    return {
        "loss": total_loss / num_batches,
    }


def validate(
    model: nn.Module,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = len(val_loader)
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            for _, metric in metrics.items():
                metric.update(outputs, masks)
    metrics = {
        metric_name: metric.compute()
        for metric_name, metric in metrics.items()
    }
    for metric_name, metric in metrics.items():
        logger.info(f"Validation {metric_name}: {metric:.4f}")
    return {
        "loss": total_loss / num_batches,
        **metrics,
    }


def train(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    metrics: dict[str, nn.Module],
    config: DictConfig,
) -> None:
    config = TrainConfig(**config)
    config.log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    checkpoint_saver = CheckpointSaver(
        model, optimizer, CheckpointSaverConfig(save_dir=config.checkpoint_dir)
    )
    for epoch in range(config.num_epochs):
        train_metrics = train_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            config.device,
            epoch,
        )
        val_metrics = validate(
            model, val_loader, criterion, metrics, config.device
        )
        checkpoint_saver.save(epoch, val_metrics["loss"])
        writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
        writer.add_scalar("Loss/Validation", val_metrics["loss"], epoch)
        for metric_name, metric_value in train_metrics.items():
            writer.add_scalar(
                f"Metrics/{metric_name}/Train", metric_value, epoch
            )
        for metric_name, metric_value in val_metrics.items():
            writer.add_scalar(
                f"Metrics/{metric_name}/Validation", metric_value, epoch
            )
        logger.info(f"Epoch {epoch + 1}/{config.num_epochs}:")
        logger.info(f"  Train Loss: {train_metrics['loss']:.4f}")
        logger.info(f"  Val Loss: {val_metrics['loss']:.4f}")
    writer.close()
    logger.info("Training completed!")
    logger.info(f"Best validation loss: {val_metrics['loss']:.4f}")
