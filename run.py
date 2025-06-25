import hydra
from omegaconf import DictConfig

from chest_segment.dataset import (
    ChestDataset,
    ChestDatasetConfig,
    get_dataloaders,
)
from chest_segment.models import get_model_from_config
from chest_segment.test import test
from chest_segment.train import train
from chest_segment.transforms import (
    ChestAllTransformsConfig,
    ChestImageTransformsConfig,
    ChestMaskTransformsConfig,
    get_all_transforms,
    get_image_transforms,
    get_mask_transforms,
)
from chest_segment.utils import get_loss, get_metrics, get_optimizer


@hydra.main(version_base=None, config_path="config", config_name="config")
def main(cfg: DictConfig) -> None:
    dataset_config = ChestDatasetConfig(**cfg.dataset)
    dataset = ChestDataset(dataset_config)
    model = get_model_from_config(cfg)
    criterion = get_loss(cfg)
    metrics = get_metrics(cfg)
    match cfg.mode:
        case "train":
            optimizer = get_optimizer(model, cfg)
            dataset.set_transforms(
                image_transforms=get_image_transforms(
                    ChestImageTransformsConfig(**cfg.train.image_transforms)
                )
                if cfg.train.image_transforms
                else None,
                mask_transforms=get_mask_transforms(
                    ChestMaskTransformsConfig(**cfg.train.mask_transforms)
                )
                if cfg.train.mask_transforms
                else None,
                all_transforms=get_all_transforms(
                    ChestAllTransformsConfig(**cfg.train.all_transforms)
                )
                if cfg.train.all_transforms
                else None,
            )
            train_loader, val_loader, _ = get_dataloaders(
                dataset, dataset_config
            )
            train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer,
                metrics,
                cfg.train,
            )
        case "test":
            dataset.set_transforms(
                image_transforms=get_image_transforms(
                    ChestImageTransformsConfig(**cfg.test.image_transforms)
                )
                if cfg.test.image_transforms
                else None,
                mask_transforms=get_mask_transforms(
                    ChestMaskTransformsConfig(**cfg.test.mask_transforms)
                )
                if cfg.test.mask_transforms
                else None,
                all_transforms=get_all_transforms(
                    ChestAllTransformsConfig(**cfg.test.all_transforms)
                )
                if cfg.test.all_transforms
                else None,
            )
            _, _, test_loader = get_dataloaders(dataset, dataset_config)
            test(model, test_loader, criterion, metrics, cfg.test)
        case _:
            raise ValueError(f"Invalid mode: {cfg.mode}")


if __name__ == "__main__":
    main()
