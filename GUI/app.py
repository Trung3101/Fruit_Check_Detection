import atexit
from collections import Counter, defaultdict
from datetime import datetime
import importlib
import logging
import os
import subprocess
import sys
import time
import urllib.request
import webbrowser
import warnings
from pathlib import Path
from threading import Lock
from typing import Any
from uuid import uuid4

import cv2
import numpy as np
import onnxruntime as ort
from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

try:
    ultralytics_module = importlib.import_module("ultralytics")
    YOLO = getattr(ultralytics_module, "YOLO", None)
except Exception:
    YOLO = None

try:
    rfdetr_module = importlib.import_module("rfdetr")
    RFDETRMedium = getattr(rfdetr_module, "RFDETRMedium", None)
except Exception:
    RFDETRMedium = None

try:
    supervision_module = importlib.import_module("supervision")
    SupervisionDetections = getattr(supervision_module, "Detections", None)
except Exception:
    supervision_module = None
    SupervisionDetections = None

try:
    trackers_module = importlib.import_module("trackers")
    ByteTrackTracker = getattr(trackers_module, "ByteTrackTracker", None)
except Exception:
    ByteTrackTracker = None

try:
    torch_module = importlib.import_module("torch")
except Exception:
    torch_module = None

try:
    import yaml
except Exception:
    yaml = None

app = Flask(__name__)

# --- App / model configuration ---
ROOT_DIR = Path(__file__).resolve().parents[1]
STATIC_RESULTS_DIR = Path(__file__).resolve().parent / "static" / "results"
STATIC_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DATA_YAML_PATH = ROOT_DIR / "data.yaml"
CONF_THRES = 0.6
RFDETR_INPUT_SIZE = 576
DISPLAY_IMAGE_WIDTH = 1080
DISPLAY_IMAGE_HEIGHT = 1920
FRESH_BOX_COLOR = (0, 200, 0)
NON_FRESH_BOX_COLOR = (0, 0, 255)
PROVIDERS = [
    ("TensorrtExecutionProvider", {"device_id": 0, "trt_fp16_enable": True}),
    "CUDAExecutionProvider",
    "CPUExecutionProvider",
]

VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}
MAX_FEEDBACK_RECORDS = 500

VIDEO_OUTPUT_PROFILES = [
    # Prefer browser-friendly outputs for direct playback in HTML <video>.
    {"suffix": ".webm", "fourcc": "VP80", "label": "vp8_webm"},
    {"suffix": ".webm", "fourcc": "VP90", "label": "vp9_webm"},
    {"suffix": ".mp4", "fourcc": "avc1", "label": "h264_avc1"},
    {"suffix": ".mp4", "fourcc": "H264", "label": "h264_mp4"},
    {"suffix": ".mp4", "fourcc": "mp4v", "label": "mp4v"},
    {"suffix": ".avi", "fourcc": "MJPG", "label": "mjpg_avi"},
]

MODEL_CONFIGS = {
    "rfdetr_a3": {
        "display_name": "RF-DETR Medium A3 (.pth, 576x576)",
        "type": "rfdetr",
        "path": ROOT_DIR / "weights" / "A3" / "checkpoint_best_total.pth",
    },
    "yolo26_a1": {
        "display_name": "YOLO26 A1 (.pt)",
        "type": "ultralytics",
        "path": ROOT_DIR / "weights" / "A1" / "detect" / "train" / "weights" / "best.pt",
    },
}
DEFAULT_MODEL_KEY = "rfdetr_a3"
MODEL_STORE: dict[str, dict[str, Any]] = {}
DESKTOP_SIDECAR_PROCESSES: list[subprocess.Popen[Any]] = []
_DESKTOP_BOOTSTRAP_REGISTERED = False


def _env_int(name: str, default: int) -> int:
    try:
        return int(str(os.environ.get(name, default)))
    except Exception:
        return int(default)


def _env_float(name: str, default: float) -> float:
    try:
        return float(str(os.environ.get(name, default)))
    except Exception:
        return float(default)


def _app_run_mode() -> str:
    # Default to desktop/all-in-one so overlay hotkey workflow starts automatically.
    return str(os.environ.get("APP_RUN_MODE", "desktop")).strip().lower()


FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = _env_int("FASTAPI_PORT", 8001)
FLASK_HOST = os.environ.get("FLASK_HOST", "0.0.0.0")
FLASK_PORT = _env_int("FLASK_PORT", 5000)

FEEDBACK_LABELS = {
    "correct": "Dự Đoán Đúng",
    "minor_issue": "Còn Sai Sót",
    "wrong": "Dự Đoán Sai Hoàn Toàn",
}
FEEDBACK_ROWS: list[dict[str, Any]] = []
FEEDBACK_LOCK = Lock()
ROTTEN_CLASS_IDS = {1, 3, 5, 7, 9, 11}
TRACKER_BUFFER = 30
TRACKER_ACTIVATION_THRESHOLD = 0.55
TRACKER_HIGH_CONF_THRESHOLD = 0.5
TRACKER_MIN_CONSECUTIVE_FRAMES = 1
TRACKER_MIN_IOU_THRESHOLD = 0.1
UNMATCHED_TRACK_MAX_AGE = 12
UNMATCHED_TRACK_MAX_DIST = 80.0

# Detection post-processing defaults tuned for fruit-only scenes.
ROTTEN_CONF_THRES = float(np.clip(_env_float("ROTTEN_CONF_THRES", 0.68), 0.0, 1.0))
NMS_IOU_THRES = float(np.clip(_env_float("NMS_IOU_THRES", 0.55), 0.0, 1.0))
CROSS_CLASS_NMS_IOU_THRES = float(np.clip(_env_float("CROSS_CLASS_NMS_IOU_THRES", 0.78), 0.0, 1.0))
MAX_BOX_AREA_RATIO = float(np.clip(_env_float("MAX_BOX_AREA_RATIO", 0.70), 0.05, 1.0))
MAX_BORDER_TOUCH_BOX_AREA_RATIO = float(
    np.clip(_env_float("MAX_BORDER_TOUCH_BOX_AREA_RATIO", 0.45), 0.05, 1.0)
)
MAX_BOX_SPAN_RATIO = float(np.clip(_env_float("MAX_BOX_SPAN_RATIO", 0.92), 0.10, 1.0))
MIN_BOX_SIDE_PIXELS = max(1, _env_int("MIN_BOX_SIDE_PIXELS", 6))


os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
logging.getLogger("rf-detr").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("transformers.configuration_utils").setLevel(logging.ERROR)
warnings.filterwarnings(
    "ignore",
    message=r"`use_return_dict` is deprecated! Use `return_dict` instead!",
)
warnings.filterwarnings(
    "ignore",
    message=r".*use_return_dict.*",
    category=FutureWarning,
)
warnings.filterwarnings(
    "ignore",
    message=r".*use_return_dict.*",
    category=UserWarning,
)


def to_int_with_default(value: object, default: int) -> int:
    try:
        return int(str(value))
    except Exception:
        return default


def load_label_names(yaml_path: Path) -> list[str]:
    if not yaml_path.exists():
        print(f"[WARN] data.yaml not found at: {yaml_path}")
        return []

    if yaml is None:
        print("[WARN] PyYAML is not available. Falling back to cls_<id> labels.")
        return []

    try:
        with yaml_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception as exc:
        print(f"[WARN] Failed to read labels from {yaml_path}: {exc}")
        return []

    names = data.get("names", [])
    if isinstance(names, list):
        return [str(name) for name in names]

    if isinstance(names, dict):
        def _key_fn(key: object) -> tuple[int, str]:
            key_text = str(key)
            if key_text.isdigit():
                return (0, f"{int(key_text):08d}")
            return (1, key_text)

        ordered_values = [names[key] for key in sorted(names.keys(), key=_key_fn)]
        return [str(name) for name in ordered_values]

    return []


LABEL_NAMES = load_label_names(DATA_YAML_PATH)


def label_from_class_id(class_id: int) -> str:
    if 0 <= class_id < len(LABEL_NAMES):
        return LABEL_NAMES[class_id]
    return f"cls_{class_id}"


def resolve_input_hw(shape: list[object] | tuple[object, ...]) -> tuple[int, int]:
    h_in = shape[2] if len(shape) > 2 else 640
    w_in = shape[3] if len(shape) > 3 else 640

    if isinstance(h_in, str) or h_in is None:
        h_in = 640
    if isinstance(w_in, str) or w_in is None:
        w_in = 640

    return to_int_with_default(h_in, 640), to_int_with_default(w_in, 640)


def squeeze_batch(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim >= 3 and arr.shape[0] == 1:
        arr = arr[0]
    return arr


def parse_onnx_outputs(outputs: list[np.ndarray], output_names: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    boxes = None
    scores = None
    labels = None
    logits = None

    for name, output in zip(output_names, outputs):
        arr = np.asarray(output)
        name_lower = name.lower()
        if "box" in name_lower:
            boxes = arr
        elif "score" in name_lower or "conf" in name_lower:
            scores = arr
        elif "label" in name_lower or "class" in name_lower:
            labels = arr
        elif "logit" in name_lower:
            logits = arr

    if boxes is None:
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim >= 2 and arr.shape[-1] == 4:
                boxes = arr
                break

    if logits is None:
        for output in outputs:
            arr = np.asarray(output)
            if arr.ndim >= 2 and arr.shape[-1] > 4:
                logits = arr
                break

    if boxes is None and len(outputs) == 1:
        pred = squeeze_batch(np.asarray(outputs[0]))
        if pred.ndim == 2 and pred.shape[1] >= 6:
            boxes = pred[:, :4]
            scores = pred[:, 4]
            labels = pred[:, 5]

    if boxes is None:
        raise RuntimeError("Unable to infer boxes tensor from ONNX outputs.")

    boxes = squeeze_batch(np.asarray(boxes)).reshape(-1, 4).astype(np.float32)

    if (scores is None or labels is None) and logits is not None:
        logits = squeeze_batch(np.asarray(logits))
        if logits.ndim == 1:
            logits = logits[:, None]

        if logits.ndim == 2 and logits.shape[1] == 1:
            scores = 1.0 / (1.0 + np.exp(-logits[:, 0]))
            labels = np.zeros_like(scores, dtype=np.int64)
        elif logits.ndim == 2 and logits.shape[1] > 1:
            shifted = logits - np.max(logits, axis=1, keepdims=True)
            exp_logits = np.exp(shifted)
            probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
            class_probs = probs[:, :-1] if probs.shape[1] > 1 else probs
            labels = np.argmax(class_probs, axis=1)
            scores = class_probs[np.arange(class_probs.shape[0]), labels]

    if scores is None:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    if labels is None:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

    scores = np.asarray(scores).reshape(-1).astype(np.float32)
    labels = np.asarray(labels).reshape(-1).astype(np.int64)

    n = min(len(boxes), len(scores), len(labels))
    return boxes[:n], scores[:n], labels[:n]


def decode_and_clip_boxes(
    boxes: np.ndarray,
    image_h: int,
    image_w: int,
    model_h: int,
    model_w: int,
) -> np.ndarray:
    if boxes.size == 0:
        return boxes.astype(np.float32)

    boxes = boxes.astype(np.float32).copy()

    if float(np.max(boxes)) <= 1.5:
        cx = boxes[:, 0] * image_w
        cy = boxes[:, 1] * image_h
        bw = boxes[:, 2] * image_w
        bh = boxes[:, 3] * image_h
        x1 = cx - (bw / 2.0)
        y1 = cy - (bh / 2.0)
        x2 = cx + (bw / 2.0)
        y2 = cy + (bh / 2.0)
        boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)
    else:
        scale_x = image_w / max(float(model_w), 1.0)
        scale_y = image_h / max(float(model_h), 1.0)
        boxes[:, [0, 2]] *= scale_x
        boxes[:, [1, 3]] *= scale_y

    x1 = np.minimum(boxes[:, 0], boxes[:, 2])
    y1 = np.minimum(boxes[:, 1], boxes[:, 3])
    x2 = np.maximum(boxes[:, 0], boxes[:, 2])
    y2 = np.maximum(boxes[:, 1], boxes[:, 3])
    boxes = np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, max(image_w - 1, 0))
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, max(image_h - 1, 0))
    return boxes


def box_iou_with_many(box: np.ndarray, others: np.ndarray) -> np.ndarray:
    if others.size == 0:
        return np.zeros((0,), dtype=np.float32)

    xx1 = np.maximum(float(box[0]), others[:, 0])
    yy1 = np.maximum(float(box[1]), others[:, 1])
    xx2 = np.minimum(float(box[2]), others[:, 2])
    yy2 = np.minimum(float(box[3]), others[:, 3])

    inter_w = np.maximum(0.0, xx2 - xx1)
    inter_h = np.maximum(0.0, yy2 - yy1)
    inter_area = inter_w * inter_h

    box_area = max(float(box[2]) - float(box[0]), 0.0) * max(float(box[3]) - float(box[1]), 0.0)
    others_area = np.maximum(0.0, others[:, 2] - others[:, 0]) * np.maximum(0.0, others[:, 3] - others[:, 1])
    union_area = box_area + others_area - inter_area

    iou = np.zeros_like(inter_area, dtype=np.float32)
    valid = union_area > 1e-6
    iou[valid] = inter_area[valid] / union_area[valid]
    return iou


def nms_indices(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> np.ndarray:
    n = int(boxes.shape[0])
    if n <= 1:
        return np.arange(n, dtype=np.int64)

    # Stable sort keeps deterministic ordering for equal scores.
    order = np.argsort(-scores, kind="mergesort")
    keep: list[int] = []

    while order.size > 0:
        current = int(order[0])
        keep.append(current)
        if order.size == 1:
            break

        remaining = order[1:]
        ious = box_iou_with_many(boxes[current], boxes[remaining])
        order = remaining[ious <= float(iou_threshold)]

    return np.asarray(keep, dtype=np.int64)


def build_predictions(
    boxes: np.ndarray,
    scores: np.ndarray,
    labels: np.ndarray,
    image_w: int | None = None,
    image_h: int | None = None,
) -> list[dict[str, Any]]:
    if boxes.size == 0:
        return []

    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
    scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)

    n = min(int(boxes.shape[0]), int(scores.size), int(labels.size))
    if n <= 0:
        return []

    boxes = boxes[:n]
    scores = scores[:n]
    labels = labels[:n]

    rotten_floor = max(float(CONF_THRES), float(ROTTEN_CONF_THRES))
    min_scores = np.full((n,), float(CONF_THRES), dtype=np.float32)
    min_scores[np.isin(labels, np.asarray(list(ROTTEN_CLASS_IDS), dtype=np.int64))] = rotten_floor

    keep = scores >= min_scores

    if image_w is not None and image_h is not None and image_w > 0 and image_h > 0:
        widths = np.maximum(0.0, boxes[:, 2] - boxes[:, 0])
        heights = np.maximum(0.0, boxes[:, 3] - boxes[:, 1])
        areas = widths * heights
        frame_area = float(image_w * image_h)

        area_ratio = areas / max(frame_area, 1.0)
        width_ratio = widths / max(float(image_w), 1.0)
        height_ratio = heights / max(float(image_h), 1.0)

        border_margin = 2.0
        touches_border = (
            (boxes[:, 0] <= border_margin)
            | (boxes[:, 1] <= border_margin)
            | (boxes[:, 2] >= (float(image_w) - 1.0 - border_margin))
            | (boxes[:, 3] >= (float(image_h) - 1.0 - border_margin))
        )

        keep &= widths >= float(MIN_BOX_SIDE_PIXELS)
        keep &= heights >= float(MIN_BOX_SIDE_PIXELS)
        keep &= area_ratio <= float(MAX_BOX_AREA_RATIO)
        keep &= np.maximum(width_ratio, height_ratio) <= float(MAX_BOX_SPAN_RATIO)
        keep &= ~(touches_border & (area_ratio >= float(MAX_BORDER_TOUCH_BOX_AREA_RATIO)))

    boxes = boxes[keep]
    scores = scores[keep]
    labels = labels[keep]

    if boxes.size == 0:
        return []

    if NMS_IOU_THRES > 0.0:
        keep_idx: list[int] = []
        for class_id in np.unique(labels):
            class_mask = labels == class_id
            class_indices = np.where(class_mask)[0]
            class_keep = nms_indices(boxes[class_indices], scores[class_indices], float(NMS_IOU_THRES))
            keep_idx.extend(class_indices[class_keep].tolist())

        ordered_keep = np.asarray(keep_idx, dtype=np.int64)
        if ordered_keep.size > 0:
            # Keep globally sorted by confidence after class-wise NMS.
            ordered_keep = ordered_keep[np.argsort(-scores[ordered_keep], kind="mergesort")]
            boxes = boxes[ordered_keep]
            scores = scores[ordered_keep]
            labels = labels[ordered_keep]

    if boxes.size > 1 and CROSS_CLASS_NMS_IOU_THRES > 0.0:
        keep_idx = nms_indices(boxes, scores, float(CROSS_CLASS_NMS_IOU_THRES))
        boxes = boxes[keep_idx]
        scores = scores[keep_idx]
        labels = labels[keep_idx]

    predictions: list[dict[str, Any]] = []
    for box, score, class_id in zip(boxes, scores, labels):
        class_id_int = int(class_id)
        predictions.append(
            {
                "class_id": class_id_int,
                "label": label_from_class_id(class_id_int),
                "confidence": float(score),
                "box": [float(v) for v in box.tolist()],
            }
        )
    return predictions


def summarize_predictions(predictions: list[dict[str, Any]]) -> tuple[dict[str, int], list[str]]:
    label_counts = Counter(pred["label"] for pred in predictions)
    return dict(label_counts), sorted(label_counts.keys())


def is_rotten_class_id(class_id: int) -> bool:
    return class_id in ROTTEN_CLASS_IDS


def load_onnx_model_entry(model_path: Path) -> dict[str, Any]:
    if not model_path.exists():
        return {"loaded": False, "error": f"Model file not found: {model_path}"}

    try:
        session = ort.InferenceSession(str(model_path), providers=PROVIDERS)
        provider = session.get_providers()[0]
        print(f"[SUCCESS] ONNX loaded: {model_path.name} via {provider}")
    except Exception as exc:
        print(f"[WARN] {exc}. Falling back to CPUExecutionProvider for {model_path.name}.")
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    output_names = [output.name for output in session.get_outputs()]
    input_h, input_w = resolve_input_hw(input_shape)

    print(
        f"[INFO] ONNX input={input_name}, shape={input_shape}, "
        f"resolved_hw=({input_h}, {input_w}), outputs={output_names}"
    )

    return {
        "loaded": True,
        "session": session,
        "input_name": input_name,
        "input_h": input_h,
        "input_w": input_w,
        "output_names": output_names,
        "runtime": session.get_providers()[0] if session.get_providers() else "unknown",
    }


def load_rfdetr_model_entry(model_path: Path) -> dict[str, Any]:
    if RFDETRMedium is None:
        return {
            "loaded": False,
            "error": "rfdetr is not installed in current environment.",
        }

    if not model_path.exists():
        return {"loaded": False, "error": f"Model file not found: {model_path}"}

    num_classes = len(LABEL_NAMES) if len(LABEL_NAMES) > 0 else 12
    model = None
    last_error: Exception | None = None

    # Support both old/new rfdetr APIs while prioritizing requested Medium 576 profile.
    constructor_candidates = [
        {
            "pretrain_weights": str(model_path),
            "resolution": RFDETR_INPUT_SIZE,
        },
        {
            "pretrain_weights": str(model_path),
            "resolution": RFDETR_INPUT_SIZE,
            "num_classes": num_classes,
        },
    ]

    for init_kwargs in constructor_candidates:
        try:
            model = RFDETRMedium(**init_kwargs)
            break
        except Exception as exc:
            last_error = exc

    if model is None:
        err_text = str(last_error) if last_error is not None else "unknown error"
        return {"loaded": False, "error": f"Failed to load RF-DETR checkpoint: {err_text}"}

    using_cuda = False
    if torch_module is not None:
        try:
            using_cuda = bool(torch_module.cuda.is_available())
        except Exception:
            using_cuda = False

    if using_cuda:
        try:
            inner_model = getattr(model, "model", None)
            if inner_model is not None and hasattr(inner_model, "to"):
                moved_model = inner_model.to("cuda")
                if moved_model is not None:
                    model.model = moved_model
        except Exception:
            # Keep CPU fallback when CUDA move is unsupported by current RF-DETR build.
            using_cuda = False

    # Reduce inference latency warning by enabling optimized inference path.
    if hasattr(model, "optimize_for_inference"):
        try:
            if using_cuda and torch_module is not None:
                model.optimize_for_inference(compile=False, batch_size=1, dtype=torch_module.float16)
            else:
                model.optimize_for_inference(compile=False, batch_size=1)
        except Exception:
            pass

    # Warm up once to fail fast if the checkpoint/runtime is invalid.
    try:
        warmup_image = np.zeros((RFDETR_INPUT_SIZE, RFDETR_INPUT_SIZE, 3), dtype=np.uint8)
        model.predict(warmup_image, threshold=CONF_THRES)
    except Exception as exc:
        return {"loaded": False, "error": f"RF-DETR warmup failed: {exc}"}

    print(
        f"[SUCCESS] RF-DETR Medium loaded: {model_path.name} "
        f"| input={RFDETR_INPUT_SIZE}x{RFDETR_INPUT_SIZE}"
    )
    device_text = "cuda" if using_cuda else "cpu"
    print(f"[INFO] RF-DETR runtime device: {device_text}")
    return {
        "loaded": True,
        "model": model,
        "runtime": f"rfdetr_medium_{RFDETR_INPUT_SIZE}:{device_text}",
    }


def load_ultralytics_model_entry(model_path: Path) -> dict[str, Any]:
    if YOLO is None:
        return {
            "loaded": False,
            "error": "ultralytics is not installed in current environment.",
        }

    if not model_path.exists():
        return {"loaded": False, "error": f"Model file not found: {model_path}"}

    try:
        model = YOLO(str(model_path))
    except Exception as exc:
        return {"loaded": False, "error": f"Failed to load YOLO model: {exc}"}

    device = "cpu"
    if torch_module is not None:
        try:
            if bool(torch_module.cuda.is_available()):
                device = "0"
        except Exception:
            device = "cpu"

    # Warm up one tiny inference to validate the selected execution device early.
    try:
        warmup_frame = np.zeros((64, 64, 3), dtype=np.uint8)
        model.predict(source=warmup_frame, conf=CONF_THRES, verbose=False, device=device)
    except Exception as exc:
        return {"loaded": False, "error": f"YOLO warmup failed on device {device}: {exc}"}

    print(f"[SUCCESS] YOLO loaded: {model_path.name}")
    return {"loaded": True, "model": model, "device": device, "runtime": f"torch:{device}"}


def preload_models() -> None:
    print("Loading models at startup...")

    for model_key, cfg in MODEL_CONFIGS.items():
        model_type = cfg["type"]
        model_path = cfg["path"]

        if model_type == "onnx":
            entry = load_onnx_model_entry(model_path)
        elif model_type == "rfdetr":
            entry = load_rfdetr_model_entry(model_path)
        elif model_type == "ultralytics":
            entry = load_ultralytics_model_entry(model_path)
        else:
            entry = {"loaded": False, "error": f"Unsupported model type: {model_type}"}

        MODEL_STORE[model_key] = entry

        if entry.get("loaded"):
            print(f"[READY] {model_key} | {cfg['display_name']}")
        else:
            print(f"[WARN] {model_key} failed: {entry.get('error', 'Unknown error')}")

if os.environ.get("SKIP_MODEL_PRELOAD", "0") == "1":
    print("[INFO] SKIP_MODEL_PRELOAD=1, skipping startup model loading.")
else:
    preload_models()
    print(f"[INFO] Loaded {len(LABEL_NAMES)} labels from {DATA_YAML_PATH.name}")

    if not MODEL_STORE.get(DEFAULT_MODEL_KEY, {}).get("loaded", False):
        raise RuntimeError(
            f"Default model '{DEFAULT_MODEL_KEY}' is unavailable: "
            f"{MODEL_STORE.get(DEFAULT_MODEL_KEY, {}).get('error', 'Unknown error')}"
        )


def resolve_model_key(requested_key: str) -> str:
    requested_key = (requested_key or "").strip()
    if requested_key in MODEL_CONFIGS:
        return requested_key
    return DEFAULT_MODEL_KEY


def infer_image_with_onnx(model_entry: dict[str, Any], image_bgr: np.ndarray) -> list[dict[str, Any]]:
    image_h, image_w = image_bgr.shape[:2]

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(image_rgb, (model_entry["input_w"], model_entry["input_h"]))

    img_input = resized.astype(np.float32) / 255.0
    img_input = np.transpose(img_input, (2, 0, 1))[None, ...]

    outputs = model_entry["session"].run(None, {model_entry["input_name"]: img_input})
    boxes, scores, labels = parse_onnx_outputs(outputs, model_entry["output_names"])
    boxes = decode_and_clip_boxes(
        boxes,
        image_h=image_h,
        image_w=image_w,
        model_h=model_entry["input_h"],
        model_w=model_entry["input_w"],
    )
    return build_predictions(boxes, scores, labels, image_w=image_w, image_h=image_h)


def infer_image_with_rfdetr(model_entry: dict[str, Any], image_bgr: np.ndarray) -> list[dict[str, Any]]:
    model = model_entry["model"]
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    detections = model.predict(image_rgb, threshold=CONF_THRES)

    boxes = np.asarray(getattr(detections, "xyxy", np.empty((0, 4), dtype=np.float32)), dtype=np.float32)
    if boxes.size == 0:
        return []

    boxes = boxes.reshape(-1, 4).astype(np.float32)

    confidence_values = getattr(detections, "confidence", None)
    class_id_values = getattr(detections, "class_id", None)

    if confidence_values is None:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)
    else:
        scores = np.asarray(confidence_values, dtype=np.float32).reshape(-1)

    if class_id_values is None:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)
    else:
        labels = np.asarray(class_id_values, dtype=np.int64).reshape(-1)

    image_h, image_w = image_bgr.shape[:2]
    boxes = decode_and_clip_boxes(
        boxes,
        image_h=image_h,
        image_w=image_w,
        model_h=image_h,
        model_w=image_w,
    )

    return build_predictions(boxes, scores, labels, image_w=image_w, image_h=image_h)


def infer_image_with_yolo(model_entry: dict[str, Any], image_bgr: np.ndarray) -> list[dict[str, Any]]:
    model = model_entry["model"]
    yolo_device = model_entry.get("device", "cpu")
    results = model.predict(source=image_bgr, conf=CONF_THRES, verbose=False, device=yolo_device)
    if not results:
        return []

    result = results[0]
    boxes_obj = getattr(result, "boxes", None)
    if boxes_obj is None or getattr(boxes_obj, "xyxy", None) is None:
        return []

    boxes = boxes_obj.xyxy.cpu().numpy().astype(np.float32)
    if boxes.size == 0:
        return []

    conf_tensor = getattr(boxes_obj, "conf", None)
    cls_tensor = getattr(boxes_obj, "cls", None)

    if conf_tensor is not None:
        scores = conf_tensor.cpu().numpy().astype(np.float32)
    else:
        scores = np.ones((boxes.shape[0],), dtype=np.float32)

    if cls_tensor is not None:
        labels = cls_tensor.cpu().numpy().astype(np.int64)
    else:
        labels = np.zeros((boxes.shape[0],), dtype=np.int64)

    image_h, image_w = image_bgr.shape[:2]
    boxes = decode_and_clip_boxes(
        boxes,
        image_h=image_h,
        image_w=image_w,
        model_h=image_h,
        model_w=image_w,
    )
    return build_predictions(boxes, scores, labels, image_w=image_w, image_h=image_h)


def infer_with_selected_model(model_key: str, image_bgr: np.ndarray) -> list[dict[str, Any]]:
    model_entry = MODEL_STORE.get(model_key, {})
    if not model_entry.get("loaded"):
        raise RuntimeError(f"Model '{model_key}' is unavailable.")

    model_type = MODEL_CONFIGS[model_key]["type"]
    if model_type == "onnx":
        return infer_image_with_onnx(model_entry, image_bgr)
    if model_type == "rfdetr":
        return infer_image_with_rfdetr(model_entry, image_bgr)
    if model_type == "ultralytics":
        return infer_image_with_yolo(model_entry, image_bgr)

    raise RuntimeError(f"Unsupported model type: {model_type}")


def infer_size_for_model(model_key: str, model_entry: dict[str, Any], image_bgr: np.ndarray) -> tuple[int, int]:
    src_h, src_w = image_bgr.shape[:2]
    model_type = MODEL_CONFIGS[model_key]["type"]

    if model_type == "onnx":
        width = to_int_with_default(model_entry.get("input_w", src_w), src_w)
        height = to_int_with_default(model_entry.get("input_h", src_h), src_h)
        return max(width, 1), max(height, 1)

    if model_type == "rfdetr":
        return RFDETR_INPUT_SIZE, RFDETR_INPUT_SIZE

    return src_w, src_h


def resize_for_model_inference(
    model_key: str,
    model_entry: dict[str, Any],
    image_bgr: np.ndarray,
) -> tuple[np.ndarray, int, int]:
    infer_w, infer_h = infer_size_for_model(model_key, model_entry, image_bgr)
    if image_bgr.shape[1] == infer_w and image_bgr.shape[0] == infer_h:
        return image_bgr, infer_w, infer_h

    resized = cv2.resize(image_bgr, (infer_w, infer_h))
    return resized, infer_w, infer_h


def scale_predictions_to_size(
    predictions: list[dict[str, Any]],
    src_w: int,
    src_h: int,
    dst_w: int,
    dst_h: int,
) -> list[dict[str, Any]]:
    if len(predictions) == 0:
        return []

    scale_x = dst_w / max(float(src_w), 1.0)
    scale_y = dst_h / max(float(src_h), 1.0)

    scaled_predictions: list[dict[str, Any]] = []
    for pred in predictions:
        pred_copy = dict(pred)
        box = pred_copy.get("box", [])

        if len(box) == 4:
            x1 = float(box[0]) * scale_x
            y1 = float(box[1]) * scale_y
            x2 = float(box[2]) * scale_x
            y2 = float(box[3]) * scale_y

            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)

            x1 = float(np.clip(x1, 0.0, max(dst_w - 1, 0)))
            y1 = float(np.clip(y1, 0.0, max(dst_h - 1, 0)))
            x2 = float(np.clip(x2, 0.0, max(dst_w - 1, 0)))
            y2 = float(np.clip(y2, 0.0, max(dst_h - 1, 0)))
            pred_copy["box"] = [x1, y1, x2, y2]

        scaled_predictions.append(pred_copy)

    return scaled_predictions


def color_for_label(label: str) -> tuple[int, int, int]:
    if str(label).strip().lower().endswith("fresh"):
        return FRESH_BOX_COLOR
    return NON_FRESH_BOX_COLOR


def annotation_style_for_image(image_bgr: np.ndarray) -> tuple[float, int, int, int]:
    image_h, image_w = image_bgr.shape[:2]
    base_size = max(min(image_h, image_w), 1)

    font_scale = float(np.clip(base_size / 3200.0, 0.22, 0.38))
    box_thickness = int(np.clip(round(base_size / 720.0), 1, 2))
    text_thickness = 1
    text_padding = int(np.clip(round(base_size / 640.0), 1, 2))
    return font_scale, box_thickness, text_thickness, text_padding


def draw_predictions_on_frame(
    frame_bgr: np.ndarray,
    predictions: list[dict[str, Any]],
    highlight_rotten: bool = False,
) -> np.ndarray:
    output = frame_bgr.copy()
    image_h, image_w = output.shape[:2]
    font_scale, box_thickness, text_thickness, text_padding = annotation_style_for_image(output)

    for pred in predictions:
        box = pred.get("box", [])
        if len(box) != 4:
            continue

        x1_raw, y1_raw, x2_raw, y2_raw = [int(round(float(v))) for v in box]
        x1 = int(np.clip(min(x1_raw, x2_raw), 0, max(image_w - 1, 0)))
        y1 = int(np.clip(min(y1_raw, y2_raw), 0, max(image_h - 1, 0)))
        x2 = int(np.clip(max(x1_raw, x2_raw), 0, max(image_w - 1, 0)))
        y2 = int(np.clip(max(y1_raw, y2_raw), 0, max(image_h - 1, 0)))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(pred.get("label", "unknown"))
        conf = float(pred.get("confidence", 0.0))
        color = color_for_label(label)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)

        text = f"{label} {conf * 100:.1f}%"
        (tw, th), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            text_thickness,
        )
        text_box_w = tw + (2 * text_padding)
        text_box_h = th + baseline + (2 * text_padding)

        text_x = x1
        if text_x + text_box_w > image_w:
            text_x = max(image_w - text_box_w, 0)

        if y1 - text_box_h >= 0:
            text_y = y1 - text_box_h
        else:
            text_y = min(y2 + text_padding, max(image_h - text_box_h, 0))

        cv2.rectangle(
            output,
            (text_x, text_y),
            (text_x + text_box_w, text_y + text_box_h),
            color,
            -1,
        )
        cv2.putText(
            output,
            text,
            (text_x + text_padding, text_y + text_padding + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    return output


def parse_line_points_from_form(form_data: Any) -> tuple[tuple[float, float], tuple[float, float]]:
    fields = ["line_x1", "line_y1", "line_x2", "line_y2"]
    values: list[float] = []

    for field_name in fields:
        raw_value = form_data.get(field_name, None)
        if raw_value is None:
            raise ValueError(f"Missing line field: {field_name}")
        try:
            value = float(str(raw_value).strip())
        except Exception:
            raise ValueError(f"Invalid line field '{field_name}': {raw_value}")
        if not np.isfinite(value):
            raise ValueError(f"Line field '{field_name}' must be finite.")
        values.append(value)

    x1, y1, x2, y2 = values
    if abs(x1 - x2) < 1e-6 and abs(y1 - y2) < 1e-6:
        raise ValueError("Line endpoints cannot be identical.")

    return (x1, y1), (x2, y2)


def clip_line_points_to_frame(
    line_points: tuple[tuple[float, float], tuple[float, float]],
    frame_w: int,
    frame_h: int,
) -> tuple[tuple[int, int], tuple[int, int]]:
    max_x = max(frame_w - 1, 0)
    max_y = max(frame_h - 1, 0)
    (x1, y1), (x2, y2) = line_points

    p1 = (
        int(np.clip(round(float(x1)), 0, max_x)),
        int(np.clip(round(float(y1)), 0, max_y)),
    )
    p2 = (
        int(np.clip(round(float(x2)), 0, max_x)),
        int(np.clip(round(float(y2)), 0, max_y)),
    )
    if p1 == p2:
        raise ValueError("Line is outside frame bounds or too short after clipping.")
    return p1, p2


def point_side_against_line(
    point_xy: tuple[float, float],
    line_start_xy: tuple[float, float],
    line_end_xy: tuple[float, float],
) -> float:
    px, py = point_xy
    x1, y1 = line_start_xy
    x2, y2 = line_end_xy
    return ((px - x1) * (y2 - y1)) - ((py - y1) * (x2 - x1))


def has_crossed_line(prev_side: float | None, curr_side: float, margin: float = 2.0) -> bool:
    if prev_side is None:
        return False

    prev = 0.0 if abs(float(prev_side)) <= margin else float(prev_side)
    curr = 0.0 if abs(float(curr_side)) <= margin else float(curr_side)

    if prev == 0.0 or curr == 0.0:
        return False
    return (prev < 0.0 and curr > 0.0) or (prev > 0.0 and curr < 0.0)


def _orientation(a: tuple[float, float], b: tuple[float, float], c: tuple[float, float]) -> float:
    return ((b[1] - a[1]) * (c[0] - b[0])) - ((b[0] - a[0]) * (c[1] - b[1]))


def _point_on_segment(a: tuple[float, float], b: tuple[float, float], p: tuple[float, float]) -> bool:
    return (
        min(a[0], b[0]) <= p[0] <= max(a[0], b[0])
        and min(a[1], b[1]) <= p[1] <= max(a[1], b[1])
    )


def segments_intersect(
    p1: tuple[float, float],
    p2: tuple[float, float],
    q1: tuple[float, float],
    q2: tuple[float, float],
) -> bool:
    o1 = _orientation(p1, p2, q1)
    o2 = _orientation(p1, p2, q2)
    o3 = _orientation(q1, q2, p1)
    o4 = _orientation(q1, q2, p2)

    if (o1 > 0 and o2 < 0 or o1 < 0 and o2 > 0) and (o3 > 0 and o4 < 0 or o3 < 0 and o4 > 0):
        return True

    eps = 1e-6
    if abs(o1) <= eps and _point_on_segment(p1, p2, q1):
        return True
    if abs(o2) <= eps and _point_on_segment(p1, p2, q2):
        return True
    if abs(o3) <= eps and _point_on_segment(q1, q2, p1):
        return True
    if abs(o4) <= eps and _point_on_segment(q1, q2, p2):
        return True
    return False


def is_line_touching_box(
    line_start_xy: tuple[float, float],
    line_end_xy: tuple[float, float],
    box_xyxy: list[float] | tuple[float, float, float, float],
) -> bool:
    if len(box_xyxy) != 4:
        return False

    x1, y1, x2, y2 = [float(v) for v in box_xyxy]
    left = min(x1, x2)
    right = max(x1, x2)
    top = min(y1, y2)
    bottom = max(y1, y2)

    def _inside_box(pt: tuple[float, float]) -> bool:
        return left <= pt[0] <= right and top <= pt[1] <= bottom

    if _inside_box(line_start_xy) or _inside_box(line_end_xy):
        return True

    edges = [
        ((left, top), (right, top)),
        ((right, top), (right, bottom)),
        ((right, bottom), (left, bottom)),
        ((left, bottom), (left, top)),
    ]
    for edge_start, edge_end in edges:
        if segments_intersect(line_start_xy, line_end_xy, edge_start, edge_end):
            return True
    return False


def normalize_optional_array(values: Any, n: int, default_value: Any, dtype: Any) -> np.ndarray:
    if values is None:
        return np.full((n,), default_value, dtype=dtype)

    arr = np.asarray(values, dtype=dtype).reshape(-1)
    if arr.size == n:
        return arr
    if arr.size <= 0:
        return np.full((n,), default_value, dtype=dtype)

    normalized = np.full((n,), default_value, dtype=dtype)
    copy_n = min(n, int(arr.size))
    normalized[:copy_n] = arr[:copy_n]
    return normalized


def predictions_to_tracking_detections(predictions: list[dict[str, Any]]) -> Any:
    if SupervisionDetections is None:
        raise RuntimeError("supervision is not installed in current environment.")

    if len(predictions) <= 0:
        return SupervisionDetections(
            xyxy=np.zeros((0, 4), dtype=np.float32),
            confidence=np.zeros((0,), dtype=np.float32),
            class_id=np.zeros((0,), dtype=np.int32),
        )

    boxes = np.asarray([pred.get("box", [0, 0, 0, 0]) for pred in predictions], dtype=np.float32).reshape(-1, 4)
    conf = np.asarray([float(pred.get("confidence", 0.0)) for pred in predictions], dtype=np.float32).reshape(-1)
    cls = np.asarray([int(pred.get("class_id", -1)) for pred in predictions], dtype=np.int32).reshape(-1)

    return SupervisionDetections(
        xyxy=boxes,
        confidence=conf,
        class_id=cls,
    )


def tracked_detections_to_predictions(tracked_detections: Any) -> list[dict[str, Any]]:
    if tracked_detections is None:
        return []

    boxes = np.asarray(getattr(tracked_detections, "xyxy", np.zeros((0, 4), dtype=np.float32)), dtype=np.float32)
    if boxes.size == 0:
        return []

    boxes = boxes.reshape(-1, 4)
    n = int(boxes.shape[0])

    conf_values = normalize_optional_array(
        getattr(tracked_detections, "confidence", None),
        n,
        default_value=1.0,
        dtype=np.float32,
    )
    class_values = normalize_optional_array(
        getattr(tracked_detections, "class_id", None),
        n,
        default_value=-1,
        dtype=np.int64,
    )
    track_values_raw = normalize_optional_array(
        getattr(tracked_detections, "tracker_id", None),
        n,
        default_value=-1,
        dtype=object,
    )

    rows: list[dict[str, Any]] = []
    for idx in range(n):
        track_raw = track_values_raw[idx]
        try:
            track_id = int(track_raw)
        except Exception:
            track_id = -1

        class_id = int(class_values[idx])
        rows.append(
            {
                "class_id": class_id,
                "label": label_from_class_id(class_id),
                "confidence": float(conf_values[idx]),
                "box": [float(v) for v in boxes[idx].tolist()],
                "track_id": track_id,
            }
        )

    return rows


def draw_tracked_predictions_on_frame(
    frame_bgr: np.ndarray,
    tracked_predictions: list[dict[str, Any]],
    line_points: tuple[tuple[int, int], tuple[int, int]],
    line_cross_counts: dict[str, int],
    total_cross_count: int,
) -> np.ndarray:
    output = frame_bgr.copy()
    image_h, image_w = output.shape[:2]
    base_size = max(min(image_h, image_w), 1)

    # Keep labels intentionally small for the tracking/counting mode.
    font_scale = float(np.clip(base_size / 4600.0, 0.15, 0.24))
    box_thickness = int(np.clip(round(base_size / 900.0), 1, 2))
    text_thickness = 1
    text_padding = 1

    line_start, line_end = line_points
    cv2.line(output, line_start, line_end, (255, 220, 0), 2)

    for pred in tracked_predictions:
        box = pred.get("box", [])
        if len(box) != 4:
            continue

        x1_raw, y1_raw, x2_raw, y2_raw = [int(round(float(v))) for v in box]
        x1 = int(np.clip(min(x1_raw, x2_raw), 0, max(image_w - 1, 0)))
        y1 = int(np.clip(min(y1_raw, y2_raw), 0, max(image_h - 1, 0)))
        x2 = int(np.clip(max(x1_raw, x2_raw), 0, max(image_w - 1, 0)))
        y2 = int(np.clip(max(y1_raw, y2_raw), 0, max(image_h - 1, 0)))
        if x2 <= x1 or y2 <= y1:
            continue

        label = str(pred.get("label", "unknown"))
        conf = float(pred.get("confidence", 0.0))
        track_id = int(pred.get("track_id", -1))
        color = color_for_label(label)

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)
        if track_id >= 0:
            text = f"ID {track_id} {label} {conf * 100:.1f}%"
        else:
            text = f"{label} {conf * 100:.1f}%"

        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness)
        text_box_w = tw + (2 * text_padding)
        text_box_h = th + baseline + (2 * text_padding)

        text_x = x1
        if text_x + text_box_w > image_w:
            text_x = max(image_w - text_box_w, 0)
        if y1 - text_box_h >= 0:
            text_y = y1 - text_box_h
        else:
            text_y = min(y2 + text_padding, max(image_h - text_box_h, 0))

        cv2.rectangle(output, (text_x, text_y), (text_x + text_box_w, text_y + text_box_h), color, -1)
        cv2.putText(
            output,
            text,
            (text_x + text_padding, text_y + text_padding + th),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA,
        )

    summary_text = f"Line Count: {int(total_cross_count)}"
    (sum_w, sum_h), sum_baseline = cv2.getTextSize(summary_text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(output, (8, 8), (16 + sum_w, 16 + sum_h + sum_baseline), (30, 30, 30), -1)
    cv2.putText(output, summary_text, (12, 12 + sum_h), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1, cv2.LINE_AA)

    if len(line_cross_counts) > 0:
        row_text = " | ".join(f"{k}:{v}" for k, v in sorted(line_cross_counts.items()))
        cv2.putText(output, row_text, (12, 34 + sum_h), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 255, 220), 1, cv2.LINE_AA)

    return output


def is_valid_fps(fps_value: float) -> bool:
    return bool(np.isfinite(fps_value) and 0.5 <= fps_value <= 240.0)


def estimate_fps_from_timestamps(
    cap: cv2.VideoCapture,
    sample_size: int = 12,
) -> tuple[float | None, list[np.ndarray]]:
    buffered_frames: list[np.ndarray] = []
    timestamps_ms: list[float] = []

    for _ in range(sample_size):
        ok, frame = cap.read()
        if not ok or frame is None:
            break

        buffered_frames.append(frame)
        timestamp = float(cap.get(cv2.CAP_PROP_POS_MSEC) or 0.0)
        if np.isfinite(timestamp) and timestamp > 0:
            timestamps_ms.append(timestamp)

    if len(timestamps_ms) < 2:
        return None, buffered_frames

    deltas = np.diff(np.asarray(timestamps_ms, dtype=np.float64))
    positive_deltas = deltas[deltas > 1e-3]
    if positive_deltas.size == 0:
        return None, buffered_frames

    fps_est = float(1000.0 / np.median(positive_deltas))
    if is_valid_fps(fps_est):
        return fps_est, buffered_frames

    return None, buffered_frames


def create_output_video_writer(
    output_stem: Path,
    fps: float,
    frame_size: tuple[int, int],
) -> tuple[cv2.VideoWriter, Path, dict[str, str]]:
    fourcc_fn = getattr(cv2, "VideoWriter_fourcc", None)
    if callable(fourcc_fn):
        make_fourcc = fourcc_fn
    else:
        make_fourcc = cv2.VideoWriter.fourcc

    for profile in VIDEO_OUTPUT_PROFILES:
        output_path = output_stem.with_suffix(profile["suffix"])
        try:
            fourcc = make_fourcc(*profile["fourcc"])
            writer = cv2.VideoWriter(
                str(output_path),
                to_int_with_default(fourcc, 0),
                fps,
                frame_size,
            )
        except Exception:
            continue

        if writer.isOpened():
            print(f"[INFO] Video writer selected codec={profile['label']} file={output_path.name}")
            return writer, output_path, profile

        writer.release()

    raise RuntimeError("Cannot initialize output video writer with available codecs.")


def save_annotated_image(frame_bgr: np.ndarray, source_name: str) -> str:
    source_stem = Path(source_name).stem or "image"
    output_name = f"out_{uuid4().hex}_{source_stem}.jpg"
    output_path = STATIC_RESULTS_DIR / output_name

    if not cv2.imwrite(str(output_path), frame_bgr):
        raise RuntimeError("Cannot save annotated image output.")

    return f"/static/results/{output_name}"


def detect_video_and_save(media_file, model_key: str, source_name: str) -> dict[str, Any]:
    job_id = uuid4().hex
    source_suffix = Path(source_name).suffix.lower()
    if source_suffix not in VIDEO_EXTENSIONS:
        source_suffix = ".mp4"

    input_path = STATIC_RESULTS_DIR / f"in_{job_id}{source_suffix}"
    output_stem = STATIC_RESULTS_DIR / f"out_{job_id}"

    media_file.save(str(input_path))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("Cannot open uploaded video.")

    writer = None
    writer_profile: dict[str, str] = {}
    label_counter: Counter[str] = Counter()
    total_detections = 0
    processed_frames = 0
    fps = 20.0

    try:
        first_ok, first_frame = cap.read()
        if not first_ok or first_frame is None:
            raise RuntimeError("Uploaded video has no readable frame.")

        h, w = first_frame.shape[:2]

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        estimated_fps, buffered_frames = estimate_fps_from_timestamps(cap)
        if is_valid_fps(source_fps):
            fps = float(source_fps)
        elif estimated_fps is not None:
            fps = float(estimated_fps)
        else:
            fps = 20.0

        print(
            f"[INFO] Video FPS source={source_fps:.3f}, "
            f"estimated={estimated_fps if estimated_fps is not None else 'n/a'}, selected={fps:.3f}"
        )

        # Preload all frames first so inference can run continuously with less decode bottleneck.
        preloaded_frames: list[np.ndarray] = [first_frame]
        preloaded_frames.extend(buffered_frames)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            preloaded_frames.append(frame)

        if len(preloaded_frames) <= 0:
            raise RuntimeError("Uploaded video has no decodable frame data.")

        print(f"[INFO] Preloaded {len(preloaded_frames)} video frames before inference.")

        writer, output_path, writer_profile = create_output_video_writer(
            output_stem=output_stem,
            fps=fps,
            frame_size=(w, h),
        )

        def _process_one_frame(frame_bgr: np.ndarray) -> None:
            nonlocal total_detections, processed_frames
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = cv2.resize(frame_bgr, (w, h))
            predictions = infer_with_selected_model(model_key, frame_bgr)
            total_detections += len(predictions)
            processed_frames += 1
            for pred in predictions:
                label_counter[str(pred["label"])] += 1
            drawn = draw_predictions_on_frame(frame_bgr, predictions)
            writer.write(drawn)

        for frame in preloaded_frames:
            _process_one_frame(frame)

        if processed_frames <= 0:
            raise RuntimeError("No frame was written to output video.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass

    return {
        "video_url": f"/static/results/{output_path.name}",
        "count": int(total_detections),
        "label_counts": dict(label_counter),
        "classes": sorted(label_counter.keys()),
        "frames": int(processed_frames),
        "fps": float(fps),
        "duration_sec": float(processed_frames / fps) if fps > 0 else 0.0,
        "format": str(writer_profile.get("suffix", ".mp4")),
        "codec": str(writer_profile.get("label", "unknown")),
    }


def detect_video_track_and_count(
    media_file: Any,
    model_key: str,
    source_name: str,
    line_points: tuple[tuple[float, float], tuple[float, float]],
) -> dict[str, Any]:
    if ByteTrackTracker is None:
        raise RuntimeError("trackers.ByteTrackTracker is not available in current environment.")
    if SupervisionDetections is None:
        raise RuntimeError("supervision is not available in current environment.")

    job_id = uuid4().hex
    source_suffix = Path(source_name).suffix.lower()
    if source_suffix not in VIDEO_EXTENSIONS:
        source_suffix = ".mp4"

    input_path = STATIC_RESULTS_DIR / f"in_line_{job_id}{source_suffix}"
    output_stem = STATIC_RESULTS_DIR / f"out_line_{job_id}"

    media_file.save(str(input_path))

    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass
        raise RuntimeError("Cannot open uploaded video for tracking.")

    writer = None
    writer_profile: dict[str, str] = {}
    processed_frames = 0
    fps = 20.0

    detected_label_counter: Counter[str] = Counter()
    crossed_label_counter: Counter[str] = Counter()
    counted_track_ids: set[int] = set()
    track_last_side: dict[int, float] = {}
    pseudo_track_last_pos: dict[int, tuple[float, float]] = {}
    pseudo_track_last_frame: dict[int, int] = {}
    pseudo_track_last_label: dict[int, str] = {}
    pseudo_track_counter = 0
    total_cross_count = 0

    try:
        first_ok, first_frame = cap.read()
        if not first_ok or first_frame is None:
            raise RuntimeError("Uploaded video has no readable frame.")

        h, w = first_frame.shape[:2]
        clipped_line_points = clip_line_points_to_frame(line_points, frame_w=w, frame_h=h)

        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        estimated_fps, buffered_frames = estimate_fps_from_timestamps(cap)
        if is_valid_fps(source_fps):
            fps = float(source_fps)
        elif estimated_fps is not None:
            fps = float(estimated_fps)
        else:
            fps = 20.0

        print(
            f"[INFO] Tracking video FPS source={source_fps:.3f}, "
            f"estimated={estimated_fps if estimated_fps is not None else 'n/a'}, selected={fps:.3f}"
        )

        preloaded_frames: list[np.ndarray] = [first_frame]
        preloaded_frames.extend(buffered_frames)
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            preloaded_frames.append(frame)

        if len(preloaded_frames) <= 0:
            raise RuntimeError("Uploaded video has no decodable frame data.")

        writer, output_path, writer_profile = create_output_video_writer(
            output_stem=output_stem,
            fps=fps,
            frame_size=(w, h),
        )

        try:
            tracker = ByteTrackTracker(
                lost_track_buffer=TRACKER_BUFFER,
                frame_rate=max(float(fps), 1.0),
                track_activation_threshold=TRACKER_ACTIVATION_THRESHOLD,
                minimum_consecutive_frames=TRACKER_MIN_CONSECUTIVE_FRAMES,
                minimum_iou_threshold=TRACKER_MIN_IOU_THRESHOLD,
                high_conf_det_threshold=TRACKER_HIGH_CONF_THRESHOLD,
            )
        except TypeError:
            tracker = ByteTrackTracker()

        def _resolve_track_id(
            pred: dict[str, Any],
            frame_index: int,
        ) -> tuple[int, float, float]:
            nonlocal pseudo_track_counter

            box = pred.get("box", [])
            center_x = (float(box[0]) + float(box[2])) * 0.5
            center_y = (float(box[1]) + float(box[3])) * 0.5

            track_id = int(pred.get("track_id", -1))
            if track_id >= 0:
                return track_id, center_x, center_y

            label = str(pred.get("label", "unknown"))
            best_track_id = None
            best_dist = float("inf")

            for candidate_id, (px, py) in pseudo_track_last_pos.items():
                if pseudo_track_last_label.get(candidate_id) != label:
                    continue

                frame_age = frame_index - int(pseudo_track_last_frame.get(candidate_id, -99999))
                if frame_age < 0 or frame_age > UNMATCHED_TRACK_MAX_AGE:
                    continue

                dist = float(np.hypot(center_x - px, center_y - py))
                if dist <= UNMATCHED_TRACK_MAX_DIST and dist < best_dist:
                    best_dist = dist
                    best_track_id = candidate_id

            if best_track_id is None:
                pseudo_track_counter += 1
                best_track_id = 10_000_000 + pseudo_track_counter

            pseudo_track_last_pos[best_track_id] = (center_x, center_y)
            pseudo_track_last_frame[best_track_id] = frame_index
            pseudo_track_last_label[best_track_id] = label
            return best_track_id, center_x, center_y

        def _process_one_frame(frame_bgr: np.ndarray, frame_index: int) -> None:
            nonlocal processed_frames, total_cross_count
            if frame_bgr.shape[0] != h or frame_bgr.shape[1] != w:
                frame_bgr = cv2.resize(frame_bgr, (w, h))

            predictions = infer_with_selected_model(model_key, frame_bgr)
            tracking_detections = predictions_to_tracking_detections(predictions)
            tracked_detections = tracker.update(tracking_detections)
            tracked_predictions = tracked_detections_to_predictions(tracked_detections)

            for pred in tracked_predictions:
                label = str(pred.get("label", "unknown"))
                detected_label_counter[label] += 1

                box = pred.get("box", [])
                if len(box) != 4:
                    continue

                track_id, center_x, center_y = _resolve_track_id(pred, frame_index)
                curr_side = point_side_against_line(
                    (center_x, center_y),
                    clipped_line_points[0],
                    clipped_line_points[1],
                )
                prev_side = track_last_side.get(track_id)

                crossed_line = has_crossed_line(prev_side, curr_side)
                touched_line = is_line_touching_box(
                    clipped_line_points[0],
                    clipped_line_points[1],
                    box,
                )

                if track_id not in counted_track_ids and (crossed_line or touched_line):
                    counted_track_ids.add(track_id)
                    crossed_label_counter[label] += 1
                    total_cross_count += 1

                track_last_side[track_id] = curr_side

            drawn = draw_tracked_predictions_on_frame(
                frame_bgr,
                tracked_predictions,
                line_points=clipped_line_points,
                line_cross_counts=dict(crossed_label_counter),
                total_cross_count=total_cross_count,
            )
            writer.write(drawn)
            processed_frames += 1

        for frame_index, frame in enumerate(preloaded_frames):
            _process_one_frame(frame, frame_index)

        if processed_frames <= 0:
            raise RuntimeError("No frame was written to tracked output video.")

    finally:
        cap.release()
        if writer is not None:
            writer.release()
        try:
            input_path.unlink(missing_ok=True)
        except Exception:
            pass

    (line_x1, line_y1), (line_x2, line_y2) = clipped_line_points

    return {
        "video_url": f"/static/results/{output_path.name}",
        "count": int(total_cross_count),
        "label_counts": dict(crossed_label_counter),
        "classes": sorted(crossed_label_counter.keys()),
        "detected_label_counts": dict(detected_label_counter),
        "unique_track_ids_counted": int(len(counted_track_ids)),
        "frames": int(processed_frames),
        "fps": float(fps),
        "duration_sec": float(processed_frames / fps) if fps > 0 else 0.0,
        "format": str(writer_profile.get("suffix", ".mp4")),
        "codec": str(writer_profile.get("label", "unknown")),
        "line": {
            "x1": int(line_x1),
            "y1": int(line_y1),
            "x2": int(line_x2),
            "y2": int(line_y2),
        },
    }


# --- ROUTES ---

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/models", methods=["GET"])
def models():
    rows = []
    for model_key, cfg in MODEL_CONFIGS.items():
        state = MODEL_STORE.get(model_key, {})
        rows.append(
            {
                "key": model_key,
                "name": cfg["display_name"],
                "loaded": bool(state.get("loaded", False)),
                "error": state.get("error", ""),
                "runtime": state.get("runtime", ""),
            }
        )

    return jsonify(
        {
            "status": "success",
            "default_model": DEFAULT_MODEL_KEY,
            "models": rows,
            "confidence_threshold": CONF_THRES,
        }
    )


@app.route("/feedbacks", methods=["GET"])
def feedbacks():
    with FEEDBACK_LOCK:
        rows = list(FEEDBACK_ROWS)

    return jsonify({"status": "success", "rows": rows})


@app.route("/feedback", methods=["POST"])
def feedback():
    payload = request.get_json(silent=True) or {}

    feedback_key = str(payload.get("feedback_key", "")).strip()
    if feedback_key not in FEEDBACK_LABELS:
        return jsonify({"status": "error", "error": "Invalid feedback type."}), 400

    model_key = resolve_model_key(str(payload.get("model_key", "")).strip())
    model_name = str(payload.get("model_name") or MODEL_CONFIGS[model_key]["display_name"])
    media_type = str(payload.get("media_type") or "unknown")
    source_name = str(payload.get("source_name") or "unknown")
    detection_count = to_int_with_default(payload.get("count", 0), 0)

    row = {
        "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "feedback_key": feedback_key,
        "feedback_label": FEEDBACK_LABELS[feedback_key],
        "model_key": model_key,
        "model_name": model_name,
        "media_type": media_type,
        "source_name": source_name,
        "count": detection_count,
    }

    with FEEDBACK_LOCK:
        FEEDBACK_ROWS.insert(0, row)
        if len(FEEDBACK_ROWS) > MAX_FEEDBACK_RECORDS:
            del FEEDBACK_ROWS[MAX_FEEDBACK_RECORDS:]

    return jsonify({"status": "success", "row": row})


@app.route("/predict", methods=["POST"])
def predict():
    try:
        model_key = resolve_model_key(request.form.get("model_key", DEFAULT_MODEL_KEY))
        model_entry = MODEL_STORE.get(model_key, {})

        if not model_entry.get("loaded", False):
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": f"Model '{model_key}' is not available: {model_entry.get('error', 'unknown error')}",
                    }
                ),
                400,
            )

        media_files = request.files.getlist("media")
        if len(media_files) == 0:
            fallback_file = request.files.get("media") or request.files.get("image") or request.files.get("video")
            if fallback_file is not None:
                media_files = [fallback_file]

        if len(media_files) == 0:
            return jsonify({"status": "error", "error": "Missing media file in request."}), 400

        upload_entries: list[dict[str, Any]] = []
        for idx, media_file in enumerate(media_files):
            source_name = secure_filename(media_file.filename or "") or f"upload_{idx + 1}.bin"
            mime_type = (media_file.mimetype or "").lower()
            suffix = Path(source_name).suffix.lower()
            upload_entries.append(
                {
                    "file": media_file,
                    "source_name": source_name,
                    "is_video": bool(mime_type.startswith("video/") or suffix in VIDEO_EXTENSIONS),
                }
            )

        has_any_video = any(entry["is_video"] for entry in upload_entries)

        if has_any_video:
            if len(upload_entries) != 1 or not upload_entries[0]["is_video"]:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error": "Chỉ hỗ trợ upload nhiều file cho ảnh. Video chỉ được xử lý khi upload 1 file.",
                        }
                    ),
                    400,
                )

            video_entry = upload_entries[0]
            video_result = detect_video_and_save(
                video_entry["file"],
                model_key=model_key,
                source_name=video_entry["source_name"],
            )
            return jsonify(
                {
                    "status": "success",
                    "media_type": "video",
                    "source_name": video_entry["source_name"],
                    "model_key": model_key,
                    "model_name": MODEL_CONFIGS[model_key]["display_name"],
                    "count": int(video_result["count"]),
                    "classes": video_result["classes"],
                    "label_counts": video_result["label_counts"],
                    "video_url": video_result["video_url"],
                    "processed_frames": int(video_result.get("frames", 0)),
                    "output_fps": float(video_result.get("fps", 0.0)),
                    "output_duration_sec": float(video_result.get("duration_sec", 0.0)),
                    "output_format": str(video_result.get("format", "")),
                    "output_codec": str(video_result.get("codec", "")),
                    "confidence_threshold": CONF_THRES,
                }
            )

        label_counter: Counter[str] = Counter()
        rotten_counter: Counter[str] = Counter()
        rotten_images_by_label: defaultdict[str, set[str]] = defaultdict(set)
        image_results: list[dict[str, Any]] = []
        total_count = 0

        for entry in upload_entries:
            source_name = entry["source_name"]
            media_file = entry["file"]

            img = cv2.imdecode(np.frombuffer(media_file.read(), np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error": f"Cannot decode image: {source_name}",
                        }
                    ),
                    400,
                )

            infer_img, infer_w, infer_h = resize_for_model_inference(model_key, model_entry, img)
            predictions = infer_with_selected_model(model_key, infer_img)
            predictions = scale_predictions_to_size(
                predictions,
                src_w=infer_w,
                src_h=infer_h,
                dst_w=DISPLAY_IMAGE_WIDTH,
                dst_h=DISPLAY_IMAGE_HEIGHT,
            )
            for pred in predictions:
                class_id = int(pred.get("class_id", -1))
                pred["is_rotten"] = is_rotten_class_id(class_id)

            image_label_counts, image_classes = summarize_predictions(predictions)
            image_rotten_counter: Counter[str] = Counter(
                str(pred["label"])
                for pred in predictions
                if bool(pred.get("is_rotten", False))
            )

            for label, count in image_label_counts.items():
                label_counter[str(label)] += int(count)

            for label, count in image_rotten_counter.items():
                rotten_counter[str(label)] += int(count)
                rotten_images_by_label[str(label)].add(source_name)

            display_img = cv2.resize(img, (DISPLAY_IMAGE_WIDTH, DISPLAY_IMAGE_HEIGHT))
            drawn = draw_predictions_on_frame(display_img, predictions, highlight_rotten=True)
            image_url = save_annotated_image(drawn, source_name)

            image_results.append(
                {
                    "source_name": source_name,
                    "count": int(len(predictions)),
                    "classes": image_classes,
                    "label_counts": image_label_counts,
                    "predictions": predictions,
                    "image_url": image_url,
                    "rotten_count": int(sum(image_rotten_counter.values())),
                    "rotten_counts": dict(image_rotten_counter),
                    "rotten_classes": sorted(image_rotten_counter.keys()),
                }
            )
            total_count += int(len(predictions))

        classes = sorted(label_counter.keys())
        rotten_counts = dict(sorted(rotten_counter.items()))
        rotten_images = {
            label: sorted(list(source_names))
            for label, source_names in sorted(rotten_images_by_label.items())
        }
        images_with_rotten = sorted(
            [
                row["source_name"]
                for row in image_results
                if int(row.get("rotten_count", 0)) > 0
            ]
        )

        primary_source_name = (
            image_results[0]["source_name"]
            if len(image_results) == 1
            else f"{len(image_results)} images"
        )

        primary_predictions = image_results[0]["predictions"] if len(image_results) == 1 else []
        primary_image_url = image_results[0]["image_url"] if len(image_results) == 1 else ""

        return jsonify(
            {
                "status": "success",
                "media_type": "image",
                "source_name": primary_source_name,
                "model_key": model_key,
                "model_name": MODEL_CONFIGS[model_key]["display_name"],
                "is_batch": bool(len(image_results) > 1),
                "count": int(total_count),
                "classes": classes,
                "label_counts": dict(label_counter),
                "predictions": primary_predictions,
                "image_url": primary_image_url,
                "image_results": image_results,
                "rotten_class_ids": sorted(ROTTEN_CLASS_IDS),
                "total_rotten_count": int(sum(rotten_counter.values())),
                "rotten_counts": rotten_counts,
                "rotten_images_by_label": rotten_images,
                "images_with_rotten": images_with_rotten,
                "confidence_threshold": CONF_THRES,
            }
        )

    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


@app.route("/predict-object-check", methods=["POST"])
def predict_object_check():
    try:
        model_key = resolve_model_key(request.form.get("model_key", DEFAULT_MODEL_KEY))
        model_entry = MODEL_STORE.get(model_key, {})
        if not model_entry.get("loaded", False):
            return (
                jsonify(
                    {
                        "status": "error",
                        "error": f"Model '{model_key}' is not available: {model_entry.get('error', 'unknown error')}",
                    }
                ),
                400,
            )

        video_file = request.files.get("video") or request.files.get("media")
        if video_file is None:
            return jsonify({"status": "error", "error": "Missing video file in request."}), 400

        source_name = secure_filename(video_file.filename or "") or "upload_video.mp4"
        mime_type = (video_file.mimetype or "").lower()
        suffix = Path(source_name).suffix.lower()
        is_video = bool(mime_type.startswith("video/") or suffix in VIDEO_EXTENSIONS)
        if not is_video:
            return jsonify({"status": "error", "error": "Uploaded file must be a video."}), 400

        line_points = parse_line_points_from_form(request.form)

        video_result = detect_video_track_and_count(
            video_file,
            model_key=model_key,
            source_name=source_name,
            line_points=line_points,
        )

        summary_rows = [
            {"label": label, "count": int(count)}
            for label, count in sorted(video_result.get("label_counts", {}).items())
        ]

        return jsonify(
            {
                "status": "success",
                "media_type": "video",
                "source_name": source_name,
                "model_key": model_key,
                "model_name": MODEL_CONFIGS[model_key]["display_name"],
                "count": int(video_result.get("count", 0)),
                "classes": list(video_result.get("classes", [])),
                "label_counts": dict(video_result.get("label_counts", {})),
                "summary_rows": summary_rows,
                "video_url": str(video_result.get("video_url", "")),
                "processed_frames": int(video_result.get("frames", 0)),
                "output_fps": float(video_result.get("fps", 0.0)),
                "output_duration_sec": float(video_result.get("duration_sec", 0.0)),
                "output_format": str(video_result.get("format", "")),
                "output_codec": str(video_result.get("codec", "")),
                "line": dict(video_result.get("line", {})),
                "unique_track_ids_counted": int(video_result.get("unique_track_ids_counted", 0)),
                "confidence_threshold": CONF_THRES,
            }
        )

    except ValueError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    except RuntimeError as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 500


def _build_desktop_env() -> dict[str, str]:
    env = dict(os.environ)
    env["FASTAPI_HOST"] = FASTAPI_HOST
    env["FASTAPI_PORT"] = str(FASTAPI_PORT)
    env.setdefault("DESKTOP_FASTAPI_URL", f"http://{FASTAPI_HOST}:{FASTAPI_PORT}")
    env.setdefault("DESKTOP_MODEL_KEY", DEFAULT_MODEL_KEY)
    return env


def _spawn_module(module_name: str, env: dict[str, str]) -> subprocess.Popen[Any]:
    return subprocess.Popen(
        [sys.executable, "-m", module_name],
        cwd=str(ROOT_DIR),
        env=env,
    )


def _wait_for_fastapi_ready(timeout_sec: float = 18.0) -> bool:
    health_url = f"http://{FASTAPI_HOST}:{FASTAPI_PORT}/api/health"
    deadline = time.time() + max(float(timeout_sec), 1.0)

    while time.time() < deadline:
        try:
            with urllib.request.urlopen(health_url, timeout=1.4) as resp:
                if int(getattr(resp, "status", 0)) == 200:
                    return True
        except Exception:
            pass
        time.sleep(0.3)

    return False


def _terminate_child(process: subprocess.Popen[Any] | None, name: str) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        process.terminate()
        process.wait(timeout=4.0)
    except Exception:
        try:
            process.kill()
        except Exception:
            print(f"[WARN] Unable to stop {name} process cleanly.")


def _can_bootstrap_desktop_pipeline() -> bool:
    run_mode = _app_run_mode()
    return run_mode not in {"web", "web-only", "flask"}


def _spawn_sidecar_module(
    module_name: str,
    extra_env: dict[str, str] | None = None,
) -> subprocess.Popen[Any]:
    env = _build_desktop_env()
    if extra_env:
        env.update(extra_env)

    process = _spawn_module(module_name, env)
    DESKTOP_SIDECAR_PROCESSES.append(process)
    return process


def _cleanup_desktop_sidecars() -> None:
    while len(DESKTOP_SIDECAR_PROCESSES) > 0:
        process = DESKTOP_SIDECAR_PROCESSES.pop()
        _terminate_child(process, "desktop_sidecar")


def bootstrap_desktop_pipeline() -> None:
    global _DESKTOP_BOOTSTRAP_REGISTERED

    if not _can_bootstrap_desktop_pipeline():
        return

    _spawn_sidecar_module("GUI.fastapi_service")
    _spawn_sidecar_module("GUI.overlay_app")

    if not _DESKTOP_BOOTSTRAP_REGISTERED:
        atexit.register(_cleanup_desktop_sidecars)
        _DESKTOP_BOOTSTRAP_REGISTERED = True


def run_all_in_one() -> None:
    env = _build_desktop_env()
    fastapi_process: subprocess.Popen[Any] | None = None
    overlay_process: subprocess.Popen[Any] | None = None

    try:
        fastapi_process = _spawn_module("GUI.fastapi_service", env)

        # Start overlay immediately so hotkeys are available right after app launch.
        overlay_process = _spawn_module("GUI.overlay_app", env)
        print("[READY] Overlay hotkey app started.")

        if _wait_for_fastapi_ready():
            print(f"[READY] FastAPI sidecar at http://{FASTAPI_HOST}:{FASTAPI_PORT}")
        else:
            print("[WARN] FastAPI sidecar took too long to become healthy.")

        local_url = f"http://127.0.0.1:{FLASK_PORT}"
        if os.environ.get("AUTO_OPEN_BROWSER", "1") == "1":
            try:
                webbrowser.open(local_url)
            except Exception:
                pass

        print(f"[READY] Flask GUI available at {local_url}")
        app.run(host=FLASK_HOST, port=FLASK_PORT, use_reloader=False)
    finally:
        _terminate_child(overlay_process, "overlay")
        _terminate_child(fastapi_process, "fastapi")


if __name__ == "__main__":
    run_mode = _app_run_mode()
    if run_mode in {"web", "web-only", "flask"}:
        app.run(host=FLASK_HOST, port=FLASK_PORT, use_reloader=False)
    else:
        run_all_in_one()
