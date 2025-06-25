from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from loguru import logger
from omegaconf import DictConfig
from PIL import Image
from pydantic import BaseModel
from torch import nn
from torch.utils.tensorboard import SummaryWriter

from chest_segment.checkpoint_saver import (
    get_last_checkpoint,
    load_model_checkpoint,
)
from chest_segment.dataset import ChestDataset


class TestConfig(BaseModel):
    device: str = "cuda"
    log_dir: Path = Path("logs")
    checkpoint_path: Path = Path("checkpoints")


def test_epoch(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    num_batches = len(test_loader)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(test_loader):
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            total_loss += loss.item()
            if batch_idx % 10 == 0:
                logger.info(
                    f"Test Batch {batch_idx}/{num_batches}, Loss: {loss.item():.4f}"
                )
            for _, metric in metrics.items():
                metric.update(outputs, masks)
    metrics = {
        metric_name: metric.compute()
        for metric_name, metric in metrics.items()
    }
    for metric_name, metric in metrics.items():
        logger.info(f"Test {metric_name}: {metric:.4f}")
    return {
        "loss": total_loss / num_batches,
        **metrics,
    }


def evaluate(
    model: nn.Module,
    dataset: ChestDataset,
    idxes: list[int] = (0,),
    device: torch.device = "cpu",
) -> list[torch.Tensor]:
    model.eval()
    images = [dataset[i][0].to(device) for i in idxes]
    images = torch.stack(images)
    with torch.no_grad():
        outputs = model(images)
    preps = outputs.max(1).indices.cpu()
    preps_resized = []
    for idx in idxes:
        img = dataset.get_raw_image(idx)
        prep_resized = F.interpolate(
            preps[idx].float().unsqueeze(0).unsqueeze(0),
            size=img.size,
            mode="nearest",
        )
        preps_resized.append(prep_resized.squeeze().long())
    return preps_resized


def visualize(
    preds: list[torch.Tensor],
    dataset: ChestDataset,
    idxes: list[int] = (0,),
    image_size=(512, 512),
) -> Image.Image:
    images = [dataset.get_raw_image(i) for i in idxes]
    colors = np.array(
        [
            [0, 0, 0],  # black for background
            [255, 0, 0],  # red for left lung
            [0, 255, 0],  # green for right lung
        ],
        dtype=np.uint8,
    )
    imgs_rgb = []
    for i, img in enumerate(images):
        img = img.resize(image_size).convert("RGB")
        mask = Image.fromarray(preds[i].numpy().astype(np.uint8))
        mask = np.array(mask.resize(image_size, resample=Image.NEAREST))
        img_rgb = np.concatenate([np.array(img), colors[mask]], axis=0)
        imgs_rgb.append(img_rgb)
    img_rgb = np.concatenate(imgs_rgb, axis=1)
    return Image.fromarray(img_rgb)


def test(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    metrics: dict[str, nn.Module],
    config: DictConfig,
):
    config = TestConfig(**config)
    if config.checkpoint_path.suffix != ".pth":
        config.checkpoint_path = get_last_checkpoint(config.checkpoint_path)
    model, _, _, _ = load_model_checkpoint(
        config.checkpoint_path, model, map_location=config.device
    )
    config.log_dir.mkdir(exist_ok=True)
    writer = SummaryWriter(config.log_dir)
    logger.info("Starting test evaluation...")
    test_metrics = test_epoch(
        model, test_loader, criterion, metrics, torch.device(config.device)
    )
    for metric_name, metric_value in test_metrics.items():
        writer.add_scalar(f"Test/{metric_name}", metric_value, 0)
        logger.info(f"Test {metric_name}: {metric_value:.4f}")
    writer.close()
    logger.info("Testing completed!")
    logger.info(f"Test loss: {test_metrics['loss']:.4f}")
