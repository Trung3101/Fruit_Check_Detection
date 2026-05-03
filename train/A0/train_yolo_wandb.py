import argparse
import os
from pathlib import Path
from typing import Any

import wandb
from ultralytics import YOLO


def str2bool(value: str) -> bool:
    if isinstance(value, bool):
        return value
    value = value.lower().strip()
    if value in {"true", "1", "yes", "y", "on"}:
        return True
    if value in {"false", "0", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train YOLO with W&B logging and custom early stopping on val mAP50-95"
    )

    parser.add_argument("-m", "--model", type=str, default="weights/best.pt", help="Path to model checkpoint")
    parser.add_argument("-dt", "--data", type=str, default="Fruit-Dataset-11/data.yaml", help="Path to dataset yaml")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Total training epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=48, help="Global batch size")
    parser.add_argument("-is", "--img-size", type=int, default=640, help="Image size")
    parser.add_argument("-w", "--workers", type=int, default=12, help="Dataloader workers")
    parser.add_argument("-es", "--early-stopping", type=int, default=40, help="Patience for val mAP50-95 early stopping")
    parser.add_argument("-c", "--cache", type=str2bool, default=True, help="Cache images in RAM/disk (true/false)")
    parser.add_argument(
        "-d",
        "--device",
        type=int,
        nargs="+",
        default=[0, 1],
        help="GPU IDs, e.g. -d 0 1",
    )

    parser.add_argument(
        "-wbe",
        "--wandb-entity",
        type=str,
        default="caoanhdoan130605-ho-chi-minh-city-university-of-industry",
        help="W&B entity",
    )
    parser.add_argument("-wbp", "--wandb-project", type=str, default="Yolo_A0", help="W&B project")
    parser.add_argument(
        "-wbk",
        "--wandb-key",
        type=str,
        default=os.getenv("WANDB_API_KEY", ""),
        help="W&B API key (or use WANDB_API_KEY env var)",
    )
    parser.add_argument("-n", "--run-name", type=str, default="A0_bestpt", help="W&B run name")

    return parser.parse_args()


def normalize_metric(metrics: dict[str, Any], key_candidates: list[str]) -> float | None:
    for key in key_candidates:
        if key in metrics:
            value = metrics[key]
            if isinstance(value, (int, float)):
                return float(value)
            try:
                return float(value.item())
            except Exception:
                continue
    return None


def to_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value.item())
    except Exception:
        return None


def is_main_process() -> bool:
    rank = int(os.getenv("RANK", "-1"))
    return rank in {-1, 0}


def collect_train_losses(trainer: Any) -> dict[str, float]:
    payload: dict[str, float] = {}
    tloss = getattr(trainer, "tloss", None)
    if tloss is None:
        return payload

    loss_values: list[float] = []
    try:
        if hasattr(tloss, "detach"):
            loss_values = [float(v) for v in tloss.detach().cpu().tolist()]
        elif isinstance(tloss, (list, tuple)):
            loss_values = [float(v) for v in tloss]
        else:
            value = to_float(tloss)
            if value is not None:
                loss_values = [value]
    except Exception:
        return payload

    loss_names = getattr(trainer, "loss_names", None)
    if isinstance(loss_names, (list, tuple)) and len(loss_names) == len(loss_values):
        for name, value in zip(loss_names, loss_values):
            payload[f"train/{name}"] = value
    else:
        for index, value in enumerate(loss_values):
            payload[f"train/loss_{index}"] = value

    return payload


def collect_lrs(trainer: Any) -> dict[str, float]:
    payload: dict[str, float] = {}

    lr_dict = getattr(trainer, "lr", None)
    if isinstance(lr_dict, dict):
        for key, value in lr_dict.items():
            v = to_float(value)
            if v is not None:
                payload[f"lr/{key}"] = v

    if payload:
        return payload

    optimizer = getattr(trainer, "optimizer", None)
    if optimizer is None:
        return payload

    param_groups = getattr(optimizer, "param_groups", [])
    for index, group in enumerate(param_groups):
        lr_value = to_float(group.get("lr"))
        if lr_value is not None:
            payload[f"lr/pg{index}"] = lr_value

    return payload


def log_model_artifact(run: Any, save_dir: Path, run_name: str) -> None:
    weights_dir = save_dir / "weights"
    best_path = weights_dir / "best.pt"
    last_path = weights_dir / "last.pt"

    files_to_log: list[Path] = []
    if best_path.exists():
        files_to_log.append(best_path)
    if last_path.exists():
        files_to_log.append(last_path)

    if not files_to_log:
        print(f"[W&B] No model weights found in: {weights_dir}")
        return

    artifact = wandb.Artifact(name=f"{run_name}-weights", type="model")
    for file_path in files_to_log:
        artifact.add_file(str(file_path), name=file_path.name)
    run.log_artifact(artifact)
    print(f"[W&B] Logged artifact with files: {[p.name for p in files_to_log]}")


def main() -> None:
    args = parse_args()
    main_process = is_main_process()

    if not args.wandb_key:
        raise ValueError(
            "Missing W&B API key. Provide -wbk <key> or set WANDB_API_KEY environment variable."
        )

    run: Any | None = None
    if main_process:
        wandb.login(key=args.wandb_key)
        run = wandb.init(
            entity=args.wandb_entity,
            project=args.wandb_project,
            name=args.run_name,
            config={
                "model": args.model,
                "data": args.data,
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "img_size": args.img_size,
                "workers": args.workers,
                "early_stopping": args.early_stopping,
                "cache": args.cache,
                "device": args.device,
            },
            reinit=True,
        )

    model = YOLO(args.model)

    best_map = float("-inf")
    no_improve = 0

    def on_fit_epoch_end(trainer) -> None:
        nonlocal best_map, no_improve

        if run is None:
            return

        metrics = getattr(trainer, "metrics", {}) or {}
        epoch = int(getattr(trainer, "epoch", 0)) + 1

        val_map5095 = normalize_metric(
            metrics,
            [
                "metrics/mAP50-95(B)",
                "metrics/mAP50-95",
                "val/mAP50-95(B)",
                "val/mAP50-95",
            ],
        )

        log_payload: dict[str, Any] = {"epoch": epoch}
        for k, v in metrics.items():
            value = to_float(v)
            if value is not None:
                log_payload[k] = value

        log_payload.update(collect_train_losses(trainer))
        log_payload.update(collect_lrs(trainer))

        fitness = to_float(getattr(trainer, "fitness", None))
        if fitness is not None:
            log_payload["metrics/fitness"] = fitness

        if val_map5095 is not None:
            log_payload["custom/val_mAP50_95"] = val_map5095
            if val_map5095 > best_map:
                best_map = val_map5095
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= args.early_stopping:
                print(
                    f"[EarlyStopping] Stop at epoch {epoch}: val mAP50-95 did not improve for {args.early_stopping} epochs."
                )
                trainer.stop = True
            run.summary["best/val_mAP50_95"] = best_map
            run.summary["best/epoch"] = epoch - no_improve

        wandb.log(log_payload, step=epoch)

    if run is not None:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    try:
        train_results = model.train(
            data=args.data,
            epochs=args.epochs,
            batch=args.batch_size,
            imgsz=args.img_size,
            workers=args.workers,
            cache=args.cache,
            device=args.device,
            save=True,
            val=True,
            plots=True,
        )

        if run is not None:
            save_dir: Path | None = None
            if train_results is not None:
                result_save_dir = getattr(train_results, "save_dir", None)
                if result_save_dir is not None:
                    save_dir = Path(result_save_dir)

            if save_dir is None:
                trainer = getattr(model, "trainer", None)
                trainer_save_dir = getattr(trainer, "save_dir", None)
                if trainer_save_dir is not None:
                    save_dir = Path(trainer_save_dir)

            if save_dir is not None:
                run.summary["save_dir"] = str(save_dir)
                log_model_artifact(run=run, save_dir=save_dir, run_name=args.run_name)
            else:
                print("[W&B] Could not resolve save_dir for artifact logging.")
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
