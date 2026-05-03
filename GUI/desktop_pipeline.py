from __future__ import annotations

import base64
import hashlib
import importlib
import io
from dataclasses import dataclass
from typing import Any

import requests

try:
    from PIL import Image
    from PIL import ImageGrab
except Exception:
    Image = None
    ImageGrab = None

try:
    keyboard = importlib.import_module("keyboard")
except Exception:
    keyboard = None


@dataclass(frozen=True)
class CaptureEvent:
    image_base64: str
    digest: str
    width: int
    height: int
    png_bytes: bytes


def _require_pillow() -> None:
    if Image is None:
        raise RuntimeError("Pillow is required for clipboard image processing.")


def pil_image_to_png_bytes(image: Any) -> bytes:
    _require_pillow()
    if image is None:
        raise ValueError("Image must not be None.")

    rgb_image = image.convert("RGB")
    buffer = io.BytesIO()
    rgb_image.save(buffer, format="PNG")
    return buffer.getvalue()


def pil_image_to_base64(image: Any) -> str:
    png_bytes = pil_image_to_png_bytes(image)
    return base64.b64encode(png_bytes).decode("ascii")


def compute_image_digest(image: Any) -> str:
    png_bytes = pil_image_to_png_bytes(image)
    return hashlib.sha256(png_bytes).hexdigest()


class ClipboardCaptureMonitor:
    def __init__(self, poll_interval_sec: float = 0.35) -> None:
        self.poll_interval_sec = max(float(poll_interval_sec), 0.05)
        self._last_digest: str | None = None

    def clear_history(self) -> None:
        self._last_digest = None

    def read_clipboard_image(self) -> Any | None:
        if ImageGrab is None:
            return None

        clip_data = ImageGrab.grabclipboard()
        if clip_data is None:
            return None

        # Windows clipboard can contain file paths or raw image data.
        if Image is not None and isinstance(clip_data, Image.Image):
            return clip_data

        return None

    def poll_event(self) -> CaptureEvent | None:
        image = self.read_clipboard_image()
        if image is None:
            return None

        png_bytes = pil_image_to_png_bytes(image)
        digest = hashlib.sha256(png_bytes).hexdigest()
        if digest == self._last_digest:
            return None

        self._last_digest = digest
        image_base64 = base64.b64encode(png_bytes).decode("ascii")
        width, height = image.convert("RGB").size
        return CaptureEvent(
            image_base64=image_base64,
            digest=digest,
            width=int(width),
            height=int(height),
            png_bytes=png_bytes,
        )


class GlobalHotkeyListener:
    def __init__(self, hotkey: str, callback) -> None:
        self.hotkey = str(hotkey or "windows+shift+s")
        self.callback = callback
        self._handle = None

    def start(self) -> bool:
        if keyboard is None:
            return False

        try:
            self._handle = keyboard.add_hotkey(self.hotkey, self.callback)
            return True
        except Exception:
            self._handle = None
            return False

    def stop(self) -> None:
        if keyboard is None:
            return

        if self._handle is None:
            return

        try:
            keyboard.remove_hotkey(self._handle)
        except Exception:
            pass
        finally:
            self._handle = None


class ApiPredictClient:
    def __init__(
        self,
        base_url: str,
        model_key: str = "rfdetr_a3",
        timeout_sec: float = 12.0,
        session: requests.Session | None = None,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model_key = model_key
        self.timeout_sec = float(timeout_sec)
        self.session = session or requests.Session()

    def predict_event(self, capture_event: CaptureEvent) -> dict[str, Any]:
        payload = {
            "model_key": self.model_key,
            "image_base64": capture_event.image_base64,
        }
        response = self.session.post(
            f"{self.base_url}/api/predict-base64",
            json=payload,
            timeout=self.timeout_sec,
        )
        response.raise_for_status()
        return dict(response.json() or {})
