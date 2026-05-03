import argparse
import csv
import importlib.util
import math
import os
from multiprocessing import cpu_count
from pathlib import Path
from typing import Any

import wandb
from ultralytics import YOLO

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_CSV_COLUMNS: list[str] = [
    "epoch",
    "time",
    "train/box_loss",
    "train/cls_loss",
    "train/dfl_loss",
    "metrics/precision(B)",
    "metrics/recall(B)",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "val/box_loss",
    "val/cls_loss",
    "val/dfl_loss",
    "lr/pg0",
    "lr/pg1",
    "lr/pg2",
    "lr/pg3",
    "lr/pg4",
    "lr/pg5",
    "lr/pg6",
    "lr/pg7",
]


def load_augment_getter() -> Any:
    augment_path = CURRENT_DIR / "augment.py"
    spec = importlib.util.spec_from_file_location("a1_augment", augment_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load augment module from: {augment_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    getter = getattr(module, "get_augment_train_kwargs", None)
    if getter is None:
        raise AttributeError("augment.py must expose get_augment_train_kwargs()")
    return getter


GET_AUGMENT_TRAIN_KWARGS = load_augment_getter()


def load_augment_spec_getter() -> Any:
    augment_path = CURRENT_DIR / "augment.py"
    spec = importlib.util.spec_from_file_location("a1_augment_spec", augment_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load augment module from: {augment_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    getter = getattr(module, "get_notebook_augment_spec", None)
    if getter is None:
        raise AttributeError("augment.py must expose get_notebook_augment_spec()")
    return getter


GET_NOTEBOOK_AUGMENT_SPEC = load_augment_spec_getter()


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
        description="Train YOLO A1 with W&B logging, custom augment setup, and custom early stopping"
    )

    parser.add_argument("-m", "--model", type=str, default="weights/yolo26l.pt", help="Path to model checkpoint")
    parser.add_argument("-dt", "--data", type=str, default="Fruit-Dataset-11/data.yaml", help="Path to dataset yaml")
    parser.add_argument("-e", "--epochs", type=int, default=300, help="Total training epochs")
    parser.add_argument("-bs", "--batch-size", type=int, default=64, help="Global batch size for 2x15GB GPU")
    parser.add_argument("-is", "--img-size", type=int, default=640, help="Image size")
    parser.add_argument("-w", "--workers", type=int, default=24, help="Dataloader workers for 32 CPU cores")
    parser.add_argument("-es", "--early-stopping", type=int, default=50, help="Patience for val mAP50-95 early stopping")
    parser.add_argument("-c", "--cache", type=str2bool, default=True, help="Cache images in RAM/disk (true/false)")
    parser.add_argument("--amp", type=str2bool, default=True, help="Enable Automatic Mixed Precision")
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
    parser.add_argument("-n", "--run-name", type=str, default="A1_aug_profile", help="W&B run name")

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
        parsed = float(value)
        return parsed if math.isfinite(parsed) else None

    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            parsed = float(text)
            return parsed if math.isfinite(parsed) else None
        except Exception:
            return None

    try:
        parsed = float(value.item())
        return parsed if math.isfinite(parsed) else None
    except Exception:
        return None


def read_latest_results_csv_row(results_csv_path: Path) -> dict[str, float]:
    if not results_csv_path.exists():
        return {}

    try:
        with results_csv_path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.DictReader(f)
            rows = [row for row in reader if row]
    except Exception:
        return {}

    if not rows:
        return {}

    latest_row = rows[-1]
    payload: dict[str, float] = {}
    for key in RESULTS_CSV_COLUMNS:
        if key == "epoch":
            continue

        raw_value = latest_row.get(key)
        if raw_value is None:
            continue

        value = to_float(raw_value)
        if value is not None:
            payload[key] = value

    return payload


def is_main_process() -> bool:
    rank = int(os.getenv("RANK", "-1"))
    return rank in {-1, 0}


def resolve_world_size(device_arg: list[int]) -> int:
    world_size_env = int(os.getenv("WORLD_SIZE", "0"))
    if world_size_env > 0:
        return world_size_env
    if device_arg:
        return len(device_arg)
    return 1


def resolve_effective_workers(requested_workers: int, world_size: int) -> int:
    total_cpus = cpu_count()
    # Keep some CPU headroom for augmentation, decode, and system tasks.
    cpu_budget = max(2, total_cpus - 2)
    per_rank_limit = max(2, cpu_budget // max(1, world_size))
    return max(2, min(requested_workers, per_rank_limit))


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


def log_model_artifact(
    run: Any,
    save_dir: Path,
    run_name: str,
    epoch: int | None = None,
    artifact_state: dict[str, float] | None = None,
) -> None:
    files_to_log: list[tuple[Path, str]] = []

    weight_dirs = [save_dir / "weights", save_dir / "weight"]
    found_weight_dir: Path | None = None
    for weight_dir in weight_dirs:
        if weight_dir.exists() and weight_dir.is_dir():
            found_weight_dir = weight_dir
            best_path = weight_dir / "best.pt"
            last_path = weight_dir / "last.pt"
            if best_path.exists():
                file_key = f"weights/{best_path.name}"
                mtime = best_path.stat().st_mtime
                if artifact_state is None or artifact_state.get(file_key) != mtime:
                    files_to_log.append((best_path, file_key))
                    if artifact_state is not None:
                        artifact_state[file_key] = mtime
            if last_path.exists():
                file_key = f"weights/{last_path.name}"
                mtime = last_path.stat().st_mtime
                if artifact_state is None or artifact_state.get(file_key) != mtime:
                    files_to_log.append((last_path, file_key))
                    if artifact_state is not None:
                        artifact_state[file_key] = mtime
            break

    curves_dir = save_dir / "curves"
    if curves_dir.exists() and curves_dir.is_dir():
        for file_path in sorted(curves_dir.rglob("*")):
            if file_path.is_file():
                relative_path = file_path.relative_to(curves_dir).as_posix()
                file_key = f"curves/{relative_path}"
                mtime = file_path.stat().st_mtime
                if artifact_state is None or artifact_state.get(file_key) != mtime:
                    files_to_log.append((file_path, file_key))
                    if artifact_state is not None:
                        artifact_state[file_key] = mtime

    if not files_to_log:
        print(f"[W&B] Artifact unchanged at epoch {epoch}, skip upload.")
        return

    artifact_name = f"{run_name}-artifacts"
    if epoch is not None:
        artifact_name = f"{run_name}-artifacts-epoch-{epoch:04d}"

    artifact = wandb.Artifact(name=artifact_name, type="model", metadata={"epoch": epoch})
    for file_path, artifact_name in files_to_log:
        artifact.add_file(str(file_path), name=artifact_name)
    run.log_artifact(artifact)
    print(f"[W&B] Logged artifact at epoch {epoch}: {[name for _, name in files_to_log]}")


def build_train_kwargs(args: argparse.Namespace) -> dict[str, Any]:
    world_size = resolve_world_size(args.device)
    effective_workers = resolve_effective_workers(args.workers, world_size)

    train_kwargs: dict[str, Any] = {
        "data": args.data,
        "epochs": args.epochs,
        "batch": args.batch_size,
        "imgsz": args.img_size,
        "workers": effective_workers,
        "cache": args.cache,
        "device": args.device,
        "amp": args.amp,
        "save": True,
        "val": True,
        "plots": True,
    }
    train_kwargs.update(GET_AUGMENT_TRAIN_KWARGS())

    return train_kwargs


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
                "amp": args.amp,
                "device": args.device,
                "augment_profile": "check_augment_v1",
                "augment_spec": GET_NOTEBOOK_AUGMENT_SPEC(),
            },
            reinit=True,
        )

    model = YOLO(args.model)

    best_map = float("-inf")
    best_epoch = 0
    no_improve = 0
    artifact_state: dict[str, float] = {}

    def on_fit_epoch_end(trainer) -> None:
        nonlocal best_map, best_epoch, no_improve

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

        trainer_save_dir = getattr(trainer, "save_dir", None)
        if trainer_save_dir is not None:
            results_csv_payload = read_latest_results_csv_row(Path(trainer_save_dir) / "results.csv")
            log_payload.update(results_csv_payload)

        fitness = to_float(getattr(trainer, "fitness", None))
        if fitness is not None:
            log_payload["metrics/fitness"] = fitness

        if val_map5095 is not None:
            log_payload["custom/val_mAP50_95"] = val_map5095
            if val_map5095 > best_map:
                best_map = val_map5095
                best_epoch = epoch
                no_improve = 0
            else:
                no_improve += 1

            log_payload["custom/no_improve_epochs"] = float(no_improve)

            if args.early_stopping > 0 and no_improve >= args.early_stopping:
                print(
                    f"[EarlyStopping] Stop at epoch {epoch}: val mAP50-95 has not improved for {no_improve} epochs (patience={args.early_stopping})."
                )
                trainer.stop = True
            run.summary["best/val_mAP50_95"] = best_map
            run.summary["best/epoch"] = best_epoch

        wandb.log(log_payload, step=epoch)

        if trainer_save_dir is not None:
            log_model_artifact(
                run=run,
                save_dir=Path(trainer_save_dir),
                run_name=args.run_name,
                epoch=epoch,
                artifact_state=artifact_state,
            )

    if run is not None:
        model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    train_kwargs = build_train_kwargs(args)

    try:
        train_results = model.train(**train_kwargs)

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
            else:
                print("[W&B] Could not resolve save_dir.")
    finally:
        if run is not None:
            wandb.finish()


if __name__ == "__main__":
    main()
