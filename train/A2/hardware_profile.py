from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class HardwareProfile:
    name: str
    gpu_count: int
    gpu_vram_gb: int
    ram_gb: int
    cpu_cores: int
    img_size: int
    batch_size: int
    grad_accum_steps: int
    workers: int
    amp: bool
    cache_images: bool
    prefetch_factor: int
    pin_memory: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


PROFILE_1X_4090_24GB = HardwareProfile(
    name="1x-rtx4090-24gb",
    gpu_count=1,
    gpu_vram_gb=24,
    ram_gb=48,
    cpu_cores=32,
    img_size=640,
    # RF-DETR-L is heavy on memory. Keep per-step batch conservative and use grad accumulation.
    batch_size=8,
    grad_accum_steps=4,
    workers=16,
    amp=True,
    cache_images=True,
    prefetch_factor=4,
    pin_memory=True,
)


PROFILE_2X_A5000_24GB = HardwareProfile(
    name="2x-a5000-24gb",
    gpu_count=2,
    gpu_vram_gb=24,
    ram_gb=64,
    cpu_cores=32,
    # RFDETRMedium recommended resolution.
    img_size=576,
    # Per-GPU micro-batch for DDP; keep accumulation moderate for stability.
    batch_size=8,
    grad_accum_steps=2,
    # Per-process workers; total workers is workers * gpu_count.
    workers=8,
    amp=True,
    cache_images=True,
    prefetch_factor=4,
    pin_memory=True,
)


PROFILE_2X_RTX5070TI_15GB = HardwareProfile(
    name="2x-rtx5070ti-15gb",
    gpu_count=2,
    gpu_vram_gb=15,
    ram_gb=64,
    # Requested CPU topology: 36 physical cores / 72 threads.
    cpu_cores=36,
    # Keep RFDETRMedium recommended resolution.
    img_size=576,
    # 15GB VRAM/GPU: lower micro-batch, recover effective batch with accumulation.
    batch_size=4,
    grad_accum_steps=4,
    # Per-process workers for DDP; total workers is workers * gpu_count.
    workers=12,
    amp=True,
    cache_images=True,
    prefetch_factor=4,
    pin_memory=True,
)


PROFILE_2X_RTX5060TI_16GB = HardwareProfile(
    name="2x-rtx5060ti-16gb",
    gpu_count=2,
    gpu_vram_gb=16,
    ram_gb=64,
    # Requested CPU topology: 12 physical cores / 24 threads.
    cpu_cores=12,
    # Keep RFDETRMedium recommended resolution.
    img_size=576,
    # 16GB VRAM/GPU: conservative micro-batch, recover effective batch with accumulation.
    batch_size=4,
    grad_accum_steps=4,
    # Per-process workers for DDP on a 12-core host; total workers is workers * gpu_count.
    workers=4,
    amp=True,
    cache_images=True,
    prefetch_factor=4,
    pin_memory=True,
)


def get_profile(profile_name: str) -> HardwareProfile:
    normalized = profile_name.lower().strip()
    if normalized in {"1x4090", "4090", "1x-rtx4090-24gb"}:
        return PROFILE_1X_4090_24GB
    if normalized in {
        "2xa5000",
        "2x-a5000-24gb",
        "2x-rtx-a5000-24gb",
        "a5000x2",
    }:
        return PROFILE_2X_A5000_24GB
    if normalized in {
        "2x5070ti",
        "5070tix2",
        "2x-rtx5070ti-15gb",
        "2x-rtx-5070-ti-15gb",
        "2x-5070ti-15gb",
    }:
        return PROFILE_2X_RTX5070TI_15GB
    if normalized in {
        "2x5060ti",
        "5060tix2",
        "2x-rtx5060ti-16gb",
        "2x-rtx-5060-ti-16gb",
        "2x-5060ti-16gb",
    }:
        return PROFILE_2X_RTX5060TI_16GB

    raise ValueError(
        "Unknown hardware profile: "
        f"{profile_name}. Supported: 1x-rtx4090-24gb, 2x-a5000-24gb, "
        "2x-rtx5070ti-15gb, 2x-rtx5060ti-16gb"
    )
