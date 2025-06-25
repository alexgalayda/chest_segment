import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2
from loguru import logger
from PIL import Image
from pydantic import BaseModel
from skimage.measure import label, regionprops
from torchvision import transforms as T


class ChestImageTransformsConfig(BaseModel):
    to_numpy: bool = False


class ChestMaskTransformsConfig(BaseModel):
    split_lungs: bool = False


class HorizontalFlipConfig(BaseModel):
    p: float = 0.5


class RotateConfig(BaseModel):
    limit: float = 10
    p: float = 0.5


class ChestAllTransformsConfig(BaseModel):
    image_size: tuple[int, int] = (256, 256)
    horizontal_config: HorizontalFlipConfig | None = None
    rotate_config: RotateConfig | None = None


def split_lungs(
    mask: np.ndarray, threshold: float = 0.1
) -> tuple[np.ndarray, np.ndarray]:
    labeled = label(mask)
    props = regionprops(labeled)
    max_area = max(p.area for p in props)
    props = [p for p in props if p.area / max_area > threshold]
    if len(props) != 2:
        logger.warning(f"Expected 2 lungs: {[p.area for p in props]}")
        raise ValueError(f"Expected 2 lungs, found {len(props)} regions")
    props_sorted = sorted(props, key=lambda r: r.centroid[1])
    left_label = props_sorted[0].label
    right_label = props_sorted[1].label
    left_mask = (labeled == left_label).astype(np.uint8)
    right_mask = (labeled == right_label).astype(np.uint8)
    return left_mask, right_mask


def transform_lungs(
    mask: np.ndarray | Image.Image, left_is_one: bool = True
) -> np.ndarray:
    if isinstance(mask, Image.Image):
        mask = np.array(mask)
    left, right = split_lungs(mask)
    mask = np.zeros_like(mask)
    mask[left == 1] = 1 if left_is_one else 2
    mask[right == 1] = 2 if left_is_one else 1
    mask = np.float32(mask)
    return mask


def get_mask_transforms(config: ChestMaskTransformsConfig | None = None):
    config = config or ChestMaskTransformsConfig()
    components = []
    if config.split_lungs:
        components.append(transform_lungs)
    return T.Compose(components)


def get_image_transforms(config: ChestImageTransformsConfig | None = None):
    config = config or ChestImageTransformsConfig()
    components = []
    if config.to_numpy:
        components.append(T.Lambda(lambda x: np.array(x)))
    return T.Compose(components)


def get_all_transforms(config: ChestAllTransformsConfig | None = None):
    config = config or ChestAllTransformsConfig()
    components = []
    if config.horizontal_config:
        components.append(A.HorizontalFlip(p=config.horizontal_config.p))
    if config.rotate_config:
        components.append(
            A.Rotate(
                limit=config.rotate_config.limit, p=config.rotate_config.p
            )
        )
    components.extend(
        [
            A.Resize(config.image_size[0], config.image_size[1]),
            ToTensorV2(),
        ]
    )
    transform = A.Compose(components, additional_targets={"mask": "mask"})
    return transform
