from __future__ import annotations

import importlib.util
from typing import Any


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def get_notebook_augment_spec() -> dict[str, Any]:
    """
    Throughput-first augmentation spec using Albumentations only.
    Each image is augmented by selecting 2-3 transforms uniformly at random.
    """
    return {
        "op_count_per_image": [2, 3],
        "backend": {
            "preferred": "albumentations",
            "albumentations_available": _module_available("albumentations"),
        },
        "ops_catalog": {
            "mosaic": {"enabled": True, "prob": 0.2},
            "mixup": {"enabled": True, "alpha_range": [0.2, 0.4], "prob": 0.1},
            "cutmix": {
                "enabled": False,
                "min_objects": 2,
                "max_objects": 3,
                "min_box_area": 0.0025,
            },
            "color": {
                "hue_shift_limit": 6,
                "sat_shift_limit": 24,
                "val_shift_limit": 20,
                "hsv_prob": 0.7,
                "brightness_limit": 0.12,
                "contrast_limit": 0.12,
                "brightness_contrast_prob": 0.5,
            },
            "geometric": {
                "shift_limit": 0.12,
                "scale_limit": 0.3,
                "rotate_limit": 12,
                "perspective_scale": [0.0, 0.0005],
                "perspective_prob": 0.3,
                "horizontal_flip_prob": 0.5,
                "vertical_flip_prob": 0.1,
                "random_resized_crop": {
                    "size": [640, 640],
                    "scale": [0.9, 1.0],
                    "ratio": [0.95, 1.05],
                    "prob": 0.25,
                },
            },
            "quality": {
                "motion_blur": {"blur_limit": [3, 5], "prob": 0.08},
                "gaussian_blur": {"blur_limit": [3, 3], "prob": 0.05},
                "gauss_noise": {"std_range": [0.01, 0.04], "prob": 0.08},
                "image_compression": {"quality_range": [70, 95], "prob": 0.06},
            },
        },
    }


def get_augment_profile() -> dict[str, Any]:
    """
    Backward-compatible entry point used by train_rf_detr_wandb.py.
    """
    return get_augment_train_kwargs()


def get_augment_train_kwargs() -> dict[str, Any]:
    """
    Direct YOLO equivalents mapped from notebook thresholds.
    No extra heuristic tuning is applied here.
    """
    spec = get_notebook_augment_spec()
    color = spec["ops_catalog"]["color"]
    geom = spec["ops_catalog"]["geometric"]

    return {
        "mosaic": 0.2,
        "mixup": 0.1,
        "copy_paste": 0.0,
        "close_mosaic": 12,
        "hsv_h": float(color["hue_shift_limit"]) / 180.0,
        "hsv_s": float(color["sat_shift_limit"]) / 255.0,
        "hsv_v": float(color["val_shift_limit"]) / 255.0,
        "degrees": float(geom["rotate_limit"]),
        "translate": float(geom["shift_limit"]),
        "scale": float(geom["scale_limit"]),
        "perspective": float(geom["perspective_scale"][1]),
        "fliplr": float(geom["horizontal_flip_prob"]),
        "flipud": float(geom["vertical_flip_prob"]),
        "shear": 0.0,
    }


def get_augment_spec_for_logging() -> dict[str, Any]:
    spec = get_notebook_augment_spec()
    return {
        "profile": "A2_rfdetr_l_random_2_3_ops_v1",
        "use_external_albumentations": spec["backend"]["albumentations_available"],
        "op_count_per_image": spec["op_count_per_image"],
        "ops_catalog": spec["ops_catalog"],
        "ops": get_augment_profile(),
    }


def build_albumentations_pipeline(img_size: int = 640) -> Any:
    if not _module_available("albumentations"):
        raise ImportError("albumentations is not installed")

    import albumentations as A

    spec = get_notebook_augment_spec()
    color = spec["ops_catalog"]["color"]
    geom = spec["ops_catalog"]["geometric"]
    quality = spec["ops_catalog"]["quality"]

    perspective_scale = (
        float(geom["perspective_scale"][0]),
        float(geom["perspective_scale"][1]),
    )
    noise_std_range = (
        float(quality["gauss_noise"]["std_range"][0]),
        float(quality["gauss_noise"]["std_range"][1]),
    )
    motion_blur_limit = (
        int(quality["motion_blur"]["blur_limit"][0]),
        int(quality["motion_blur"]["blur_limit"][1]),
    )

    transform_pool: list[Any] = [
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.Affine(
            rotate=(-float(geom["rotate_limit"]), float(geom["rotate_limit"])),
            translate_percent=(-float(geom["shift_limit"]), float(geom["shift_limit"])),
            scale=(1.0 - float(geom["scale_limit"]), 1.0 + float(geom["scale_limit"])),
            p=1.0,
        ),
        A.Perspective(scale=perspective_scale, p=1.0),
        A.HueSaturationValue(
            hue_shift_limit=int(color["hue_shift_limit"]),
            sat_shift_limit=int(color["sat_shift_limit"]),
            val_shift_limit=int(color["val_shift_limit"]),
            p=1.0,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=float(color["brightness_limit"]),
            contrast_limit=float(color["contrast_limit"]),
            p=1.0,
        ),
        A.GaussNoise(
            std_range=noise_std_range,
            p=1.0,
        ),
        A.MotionBlur(
            blur_limit=motion_blur_limit,
            p=1.0,
        ),
    ]

    n_min, n_max = int(spec["op_count_per_image"][0]), int(spec["op_count_per_image"][1])
    selection_block: list[Any] = [
        A.OneOf(
            [
                A.SomeOf(transform_pool, n=n_min, replace=False, p=1.0),
                A.SomeOf(transform_pool, n=n_max, replace=False, p=1.0),
            ],
            p=1.0,
        )
    ]
    pipeline: list[Any] = selection_block + [A.Resize(height=img_size, width=img_size, p=1.0)]

    return A.Compose(
        pipeline,
        bbox_params=A.BboxParams(format="yolo", label_fields=["class_labels"]),
    )
