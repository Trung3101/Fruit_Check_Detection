from __future__ import annotations

import argparse
import importlib
import os
from typing import Any

from rfdetr import RFDETRMedium
from rfdetr.training import RFDETRDataModule, RFDETREarlyStopping, RFDETRModelModule, build_trainer

from hardware_profile import HardwareProfile, get_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RF-DETR-Medium with native rfdetr API and W&B logging"
    )

    parser.add_argument(
        "-hp",
        "--hardware-profile",
        type=str,
        default="2x-a5000-24gb",
        help="Hardware profile preset (default tuned for 2x A5000 24GB)",
    )
    parser.add_argument("-dd", "--dataset-dir", type=str, default="Fruit-Dataset-11", help="Dataset root directory")
    parser.add_argument("-od", "--output-dir", type=str, default="runs/rfdetr_medium_a2", help="Output directory")

    parser.add_argument("-e", "--epochs", type=int, default=100, help="Training epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=-1, help="Per-step batch size (-1 to use profile)")
    parser.add_argument(
        "-ga",
        "--grad-accum-steps",
        type=int,
        default=-1,
        help="Gradient accumulation steps (-1 to use profile)",
    )
    parser.add_argument("-lr", "--lr", type=float, default=1e-4, help="Base learning rate")
    parser.add_argument("-lre", "--lr-encoder", type=float, default=1.5e-4, help="Backbone learning rate")
    parser.add_argument("-wd", "--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("-w", "--num-workers", type=int, default=-1, help="Dataloader workers (-1 to use profile)")
    parser.add_argument(
        "-r",
        "--resolution",
        type=int,
        default=-1,
        help="Input resolution (-1 to use profile, recommended: 576)",
    )
    parser.add_argument("-rs", "--resume", type=str, default=None, help="Checkpoint path to resume training")
    parser.add_argument(
        "-gc",
        "--gradient-checkpointing",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable gradient checkpointing for lower VRAM",
    )

    parser.add_argument(
        "-wb",
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable Weights & Biases logging via native rfdetr logger",
    )
    parser.add_argument(
        "-wbe",
        "--wandb-entity",
        type=str,
        default="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
        help="W&B entity",
    )
    parser.add_argument("-wp", "--wandb-project", type=str, default="RF-DETR", help="W&B project name")
    parser.add_argument(
        "-wbk",
        "--wandb-key",
        type=str,
        default=None,
        help="W&B API key (if omitted, WANDB_API_KEY env var is used)",
    )
    parser.add_argument(
        "-wbr",
        "--wandb-resume",
        type=str,
        choices=["never", "allow", "must", "auto"],
        default="never",
        help="W&B resume policy. Use 'never' to avoid stale-run 403 errors",
    )
    parser.add_argument(
        "-wbi",
        "--wandb-run-id",
        type=str,
        default=None,
        help="Explicit W&B run id when resuming (ignored when --wandb-resume=never)",
    )
    parser.add_argument("-n", "--run-name", type=str, default="A2_rfdetr_medium", help="W&B run name")

    parser.add_argument(
        "-pw",
        "--pretrain-weights",
        type=str,
        default="rf-detr-medium.pth",
        help="Pretrained checkpoint used by RFDETRMedium",
    )

    parser.add_argument("-dv", "--devices", type=int, default=-1, help="Number of GPUs (-1 to use profile)")
    parser.add_argument(
        "-st",
        "--strategy",
        type=str,
        default="ddp_find_unused_parameters_true",
        help="PyTorch Lightning distributed strategy",
    )
    parser.add_argument(
        "-ac",
        "--accelerator",
        type=str,
        default="gpu",
        help="PyTorch Lightning accelerator (gpu, cpu, mps, auto)",
    )
    parser.add_argument(
        "-sb",
        "--sync-bn",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable SyncBatchNorm in multi-GPU training",
    )

    parser.add_argument(
        "-pf",
        "--prefetch-factor",
        type=int,
        default=-1,
        help="Dataloader prefetch factor (-1 to use profile)",
    )
    parser.add_argument(
        "-pm",
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable pinned host memory for faster H2D transfer",
    )
    parser.add_argument(
        "-pwk",
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep dataloader workers alive across epochs",
    )

    parser.add_argument(
        "-esp",
        "--early-stopping-patience",
        type=int,
        default=30,
        help="Stop if val mAP@50 does not improve for N validation epochs",
    )
    parser.add_argument(
        "-esd",
        "--early-stopping-min-delta",
        type=float,
        default=0.0,
        help="Minimum strict improvement required for mAP@50",
    )

    return parser.parse_args()


def resolve_runtime_hparams(args: argparse.Namespace, profile: HardwareProfile) -> dict[str, Any]:
    return {
        "resolution": profile.img_size if args.resolution < 0 else args.resolution,
        "batch_size": profile.batch_size if args.batch_size < 0 else args.batch_size,
        "grad_accum_steps": profile.grad_accum_steps
        if args.grad_accum_steps < 0
        else args.grad_accum_steps,
        "num_workers": profile.workers if args.num_workers < 0 else args.num_workers,
        "prefetch_factor": profile.prefetch_factor if args.prefetch_factor < 0 else args.prefetch_factor,
        "pin_memory": profile.pin_memory if args.pin_memory is None else args.pin_memory,
        "devices": profile.gpu_count if args.devices < 0 else args.devices,
    }


def attach_map50_early_stopping(
    trainer: Any,
    patience: int,
    min_delta: float,
) -> None:
    callbacks = [cb for cb in trainer.callbacks if not isinstance(cb, RFDETREarlyStopping)]

    callbacks.append(
        RFDETREarlyStopping(
            patience=patience,
            min_delta=min_delta,
            use_ema=False,
            monitor_regular="val/mAP_50",
            monitor_ema="val/mAP_50",
            verbose=True,
        )
    )
    trainer.callbacks[:] = callbacks


def main() -> None:
    args = parse_args()

    if args.wandb:
        api_key = args.wandb_key or os.getenv("WANDB_API_KEY")
        if not api_key:
            raise ValueError(
                "Missing W&B API key. Provide -wbk <key> or set WANDB_API_KEY environment variable."
            )

        os.environ["WANDB_API_KEY"] = api_key
        if args.wandb_entity:
            os.environ["WANDB_ENTITY"] = args.wandb_entity

        # Prevent accidental resume from stale local run state that can trigger 403 errors.
        os.environ.pop("WANDB_ANONYMOUS", None)
        os.environ["WANDB_RESUME"] = args.wandb_resume
        if args.wandb_resume == "never":
            os.environ.pop("WANDB_RUN_ID", None)
        elif args.wandb_run_id:
            os.environ["WANDB_RUN_ID"] = args.wandb_run_id
        else:
            os.environ.pop("WANDB_RUN_ID", None)

        try:
            wandb_module = importlib.import_module("wandb")
        except ModuleNotFoundError as exc:
            raise ImportError("W&B logging requires package 'wandb'. Install with: pip install wandb") from exc

        rank = int(os.getenv("RANK", "-1"))
        if rank in {-1, 0}:
            wandb_module.login(key=api_key, relogin=True)

    profile = get_profile(args.hardware_profile)
    resolved = resolve_runtime_hparams(args=args, profile=profile)
    if args.accelerator == "gpu":
        try:
            import torch

            visible_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            visible_gpus = 0

        if visible_gpus <= 0:
            raise RuntimeError(
                "accelerator='gpu' but no CUDA device is visible. "
                "Set CUDA_VISIBLE_DEVICES correctly or run with --accelerator cpu"
            )

        if int(resolved["devices"]) > visible_gpus:
            print(
                f"[WARN] Requested devices={resolved['devices']} from profile '{profile.name}', "
                f"but only {visible_gpus} CUDA device(s) are visible. Falling back to {visible_gpus}."
            )
            resolved["devices"] = visible_gpus

    num_workers = int(resolved["num_workers"])
    prefetch_factor = int(resolved["prefetch_factor"]) if num_workers > 0 else None
    persistent_workers = bool(args.persistent_workers) and num_workers > 0
    strategy = args.strategy if int(resolved["devices"]) > 1 else "auto"

    # RFDETRMedium already uses a DINOv2-based backbone in native rfdetr configs.
    model = RFDETRMedium(
        encoder="dinov2_windowed_small",
        resolution=resolved["resolution"],
        pretrain_weights=args.pretrain_weights,
    )

    # Contrastive denoising is integrated in RF-DETR's native training pipeline.
    train_kwargs = {
        "dataset_dir": args.dataset_dir,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": resolved["batch_size"],
        "grad_accum_steps": resolved["grad_accum_steps"],
        "lr": args.lr,
        "lr_encoder": args.lr_encoder,
        "weight_decay": args.weight_decay,
        "num_workers": num_workers,
        "resolution": resolved["resolution"],
        "resume": args.resume,
        "gradient_checkpointing": args.gradient_checkpointing,
        "accelerator": args.accelerator,
        "devices": resolved["devices"],
        "strategy": strategy,
        "sync_bn": args.sync_bn and resolved["devices"] > 1,
        "prefetch_factor": prefetch_factor,
        "pin_memory": resolved["pin_memory"],
        "persistent_workers": persistent_workers,
        "early_stopping": False,
        "wandb": args.wandb,
        "project": args.wandb_project if args.wandb else None,
        "run": args.run_name if args.wandb else None,
    }

    filtered_kwargs = {k: v for k, v in train_kwargs.items() if v is not None}
    train_config = model.get_train_config(**filtered_kwargs)

    module = RFDETRModelModule(model.model_config, train_config)
    datamodule = RFDETRDataModule(model.model_config, train_config)
    trainer = build_trainer(train_config, model.model_config, accelerator=train_config.accelerator)

    attach_map50_early_stopping(
        trainer=trainer,
        patience=args.early_stopping_patience,
        min_delta=args.early_stopping_min_delta,
    )
    trainer.fit(module, datamodule, ckpt_path=train_config.resume or None)


if __name__ == "__main__":
    main()