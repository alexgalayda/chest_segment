from typing import Literal

import segmentation_models_pytorch as smp
from omegaconf import DictConfig
from pydantic import BaseModel
from torch import nn


class ModelConfig(BaseModel):
    in_channels: int = 1
    encoder_depth: int = 5
    classes: int = 3


class UnetConfig(ModelConfig):
    encoder_name: str = "resnet18"
    encoder_weights: str | None = None
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)


class UnetPlusPlusConfig(ModelConfig):
    encoder_name: str = "resnet34"
    encoder_weights: str | None = None
    decoder_channels: tuple[int, ...] = (256, 128, 64, 32, 16)


def get_model(
    model_name: Literal["unet", "unet++"], config: DictConfig
) -> nn.Module:
    match model_name:
        case "unet":
            return smp.Unet(**UnetConfig(**config).model_dump())
        case "unet++":
            return smp.UnetPlusPlus(
                **UnetPlusPlusConfig(**config).model_dump()
            )
        case _:
            raise ValueError(f"Invalid model name: {model_name}")


def get_model_from_config(config: DictConfig) -> nn.Module:
    model_name = config.model.name
    if model_config := config.get(model_name):
        return get_model(model_name, model_config).to(config.model.device)
    raise ValueError(f"Model config not found for {model_name}")
