from __future__ import annotations

import base64
import os
import time
from collections import Counter
from threading import Lock
from typing import Any

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from GUI import app as flask_app_module

FASTAPI_HOST = os.environ.get("FASTAPI_HOST", "127.0.0.1")
FASTAPI_PORT = int(os.environ.get("FASTAPI_PORT", "8001"))
MODEL_KEY = "rfdetr_a3"
PREDICT_LOCK = Lock()

app = FastAPI(title="Fruit Detection Sidecar API", version="1.0.0")


class PredictBase64Request(BaseModel):
    image_base64: str = Field(..., min_length=10)
    model_key: str | None = None


def _decode_base64_to_bgr(image_base64: str) -> np.ndarray:
    try:
        image_bytes = base64.b64decode(image_base64, validate=True)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image_base64 payload: {exc}")

    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Cannot decode image from base64 payload.")
    return image


def _validate_model_key(requested_model_key: str | None) -> str:
    if requested_model_key and str(requested_model_key).strip() != MODEL_KEY:
        raise HTTPException(
            status_code=400,
            detail=f"Only model '{MODEL_KEY}' is supported in desktop pipeline.",
        )

    model_entry = flask_app_module.MODEL_STORE.get(MODEL_KEY, {})
    if not model_entry.get("loaded", False):
        raise HTTPException(
            status_code=503,
            detail=(
                f"Model '{MODEL_KEY}' is unavailable: "
                f"{model_entry.get('error', 'unknown error')}"
            ),
        )

    return MODEL_KEY


def _enrich_predictions(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    enriched: list[dict[str, Any]] = []
    for pred in predictions:
        pred_copy = dict(pred)
        class_id = int(pred_copy.get("class_id", -1))
        pred_copy["is_rotten"] = bool(flask_app_module.is_rotten_class_id(class_id))
        enriched.append(pred_copy)
    return enriched


@app.get("/api/health")
def health() -> dict[str, Any]:
    model_state = flask_app_module.MODEL_STORE.get(MODEL_KEY, {})
    return {
        "status": "ok",
        "model_key": MODEL_KEY,
        "loaded": bool(model_state.get("loaded", False)),
        "error": model_state.get("error", ""),
    }


@app.get("/api/models")
def models() -> dict[str, Any]:
    cfg = flask_app_module.MODEL_CONFIGS.get(MODEL_KEY, {})
    state = flask_app_module.MODEL_STORE.get(MODEL_KEY, {})
    return {
        "status": "success",
        "default_model": MODEL_KEY,
        "models": [
            {
                "key": MODEL_KEY,
                "name": cfg.get("display_name", MODEL_KEY),
                "loaded": bool(state.get("loaded", False)),
                "runtime": state.get("runtime", ""),
                "error": state.get("error", ""),
            }
        ],
        "rotten_class_ids": sorted(flask_app_module.ROTTEN_CLASS_IDS),
        "confidence_threshold": float(flask_app_module.CONF_THRES),
    }


@app.post("/api/predict-base64")
def predict_base64(payload: PredictBase64Request) -> dict[str, Any]:
    model_key = _validate_model_key(payload.model_key)
    image_bgr = _decode_base64_to_bgr(payload.image_base64)

    image_h, image_w = image_bgr.shape[:2]
    started_at = time.perf_counter()

    with PREDICT_LOCK:
        infer_image, infer_w, infer_h = flask_app_module.resize_for_model_inference(
            model_key,
            flask_app_module.MODEL_STORE.get(model_key, {}),
            image_bgr,
        )
        predictions = flask_app_module.infer_with_selected_model(model_key, infer_image)
        predictions = flask_app_module.scale_predictions_to_size(
            predictions,
            src_w=infer_w,
            src_h=infer_h,
            dst_w=image_w,
            dst_h=image_h,
        )

    predictions = _enrich_predictions(predictions)
    infer_time_ms = (time.perf_counter() - started_at) * 1000.0

    label_counts = Counter(str(pred.get("label", "unknown")) for pred in predictions)
    rotten_counts = Counter(
        str(pred.get("label", "unknown"))
        for pred in predictions
        if bool(pred.get("is_rotten", False))
    )

    return {
        "status": "success",
        "media_type": "image",
        "model_key": model_key,
        "model_name": flask_app_module.MODEL_CONFIGS[model_key]["display_name"],
        "image_width": int(image_w),
        "image_height": int(image_h),
        "count": int(len(predictions)),
        "classes": sorted(label_counts.keys()),
        "label_counts": dict(label_counts),
        "predictions": predictions,
        "rotten_class_ids": sorted(flask_app_module.ROTTEN_CLASS_IDS),
        "total_rotten_count": int(sum(rotten_counts.values())),
        "rotten_counts": dict(rotten_counts),
        "inference_time_ms": float(infer_time_ms),
        "confidence_threshold": float(flask_app_module.CONF_THRES),
    }


def run() -> None:
    import uvicorn

    uvicorn.run(
        "GUI.fastapi_service:app",
        host=FASTAPI_HOST,
        port=FASTAPI_PORT,
        reload=False,
    )


if __name__ == "__main__":
    run()
