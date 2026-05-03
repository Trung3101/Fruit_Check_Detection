from __future__ import annotations

import importlib
import os
import sys
import threading
from typing import Any

import cv2
import numpy as np


_desktop_pipeline_module = importlib.import_module("GUI.desktop_pipeline")
ApiPredictClient = getattr(_desktop_pipeline_module, "ApiPredictClient")
ClipboardCaptureMonitor = getattr(_desktop_pipeline_module, "ClipboardCaptureMonitor")
GlobalHotkeyListener = getattr(_desktop_pipeline_module, "GlobalHotkeyListener")

FRESH_BOX_COLOR = (0, 200, 0)
ROTTEN_BOX_COLOR = (0, 0, 255)


def annotation_style_for_image(image_bgr: np.ndarray) -> tuple[float, int, int, int]:
    image_h, image_w = image_bgr.shape[:2]
    base_size = max(min(image_h, image_w), 1)

    font_scale = float(np.clip(base_size / 3200.0, 0.22, 0.36))
    box_thickness = int(np.clip(round(base_size / 720.0), 1, 2))
    text_thickness = 1
    text_padding = int(np.clip(round(base_size / 640.0), 1, 2))
    return font_scale, box_thickness, text_thickness, text_padding


def build_overlay_summary(payload: dict[str, Any]) -> dict[str, Any]:
    model_name = str(payload.get("model_name", "RF-DETR A3"))
    total_count = int(payload.get("count", 0))
    rotten_count = int(payload.get("total_rotten_count", 0))
    inference_ms = float(payload.get("inference_time_ms", 0.0))
    label_counts = payload.get("label_counts", {}) or {}

    if rotten_count > 0:
        title = f"ALERT | Rotten detected: {rotten_count}"
        warning = f"{model_name} found rotten fruit."
    else:
        title = "OK | No rotten fruit"
        warning = ""

    class_lines = [f"{name}: {int(count)}" for name, count in sorted(label_counts.items())]

    return {
        "title": title,
        "warning": warning,
        "stats": f"Detected={total_count} | Inference={inference_ms:.1f} ms",
        "class_lines": class_lines,
    }


def draw_predictions_on_bgr(image_bgr: np.ndarray, predictions: list[dict[str, Any]]) -> np.ndarray:
    output = image_bgr.copy()
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
        confidence = float(pred.get("confidence", 0.0))
        is_rotten = bool(pred.get("is_rotten", False))
        color = ROTTEN_BOX_COLOR if is_rotten else FRESH_BOX_COLOR

        cv2.rectangle(output, (x1, y1), (x2, y2), color, box_thickness)
        text = f"{label} {confidence * 100:.1f}%"
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


def decode_png_bytes_to_bgr(png_bytes: bytes) -> np.ndarray:
    arr = np.frombuffer(png_bytes, dtype=np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("Cannot decode PNG bytes from clipboard.")
    return image


def run_overlay() -> int:
    try:
        from PyQt6 import QtCore, QtGui, QtWidgets
    except Exception as exc:
        print(f"[ERROR] PyQt6 is required for overlay app: {exc}")
        return 1

    Qt = QtCore.Qt
    pyqt_signal = QtCore.pyqtSignal

    class ClickableLabel(QtWidgets.QLabel):
        clicked = pyqt_signal()

        def mousePressEvent(self, ev):
            if ev.button() == Qt.MouseButton.LeftButton:
                self.clicked.emit()
            super().mousePressEvent(ev)

    class PreviewDialog(QtWidgets.QDialog):
        def __init__(self, parent, pixmap):
            super().__init__(parent)
            self._pixmap = pixmap
            self.setWindowTitle("Prediction Preview")
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Dialog
            )
            self.setModal(True)
            self.resize(920, 700)

            self._image_label = QtWidgets.QLabel()
            self._image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self._image_label.setStyleSheet("background:#101418;border-radius:14px;")

            close_btn = QtWidgets.QPushButton("x")
            close_btn.setFixedSize(30, 30)
            close_btn.setStyleSheet(
                "QPushButton{background:#bf1f2f;color:white;border:none;border-radius:15px;font-weight:bold;}"
                "QPushButton:hover{background:#d32639;}"
            )

            def _close_dialog() -> None:
                self.close()

            close_btn.clicked.connect(_close_dialog)

            top_row = QtWidgets.QHBoxLayout()
            top_row.addStretch(1)
            top_row.addWidget(close_btn)

            layout = QtWidgets.QVBoxLayout(self)
            layout.setContentsMargins(14, 14, 14, 14)
            layout.setSpacing(10)
            layout.addLayout(top_row)
            layout.addWidget(self._image_label, 1)
            self._refresh_pixmap()

        def resizeEvent(self, a0):
            super().resizeEvent(a0)
            self._refresh_pixmap()

        def _refresh_pixmap(self):
            if self._pixmap.isNull():
                return
            self._image_label.setPixmap(
                self._pixmap.scaled(
                    self._image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

    class CaptureWorker(QtCore.QThread):
        prediction_ready = pyqt_signal(dict, bytes)
        status_update = pyqt_signal(str)
        error_update = pyqt_signal(str)
        capture_requested = pyqt_signal()

        def __init__(self, api_url, model_key, hotkey, poll_interval_sec):
            super().__init__()
            self.api_url = api_url
            self.model_key = model_key
            self.hotkey = hotkey
            self.poll_interval_sec = float(poll_interval_sec)
            self._stop_event = threading.Event()
            self._awaiting_capture = threading.Event()
            self._hotkey_enabled = False

        def stop(self):
            self._stop_event.set()

        def run(self):
            monitor = ClipboardCaptureMonitor(poll_interval_sec=self.poll_interval_sec)
            client = ApiPredictClient(self.api_url, model_key=self.model_key, timeout_sec=12.0)

            def _on_hotkey():
                self._awaiting_capture.set()
                self.capture_requested.emit()
                self.status_update.emit("Hotkey pressed. Waiting for new clipboard image...")

            hotkey_listener = GlobalHotkeyListener(self.hotkey, _on_hotkey)
            self._hotkey_enabled = hotkey_listener.start()
            if self._hotkey_enabled:
                self.status_update.emit(f"Hotkey listener enabled: {self.hotkey}")
            else:
                self.status_update.emit(
                    "Hotkey listener unavailable. Clipboard watch is still active."
                )

            while not self._stop_event.is_set():
                try:
                    event = monitor.poll_event()
                except Exception as exc:
                    self.error_update.emit(f"Clipboard read error: {exc}")
                    self.msleep(int(monitor.poll_interval_sec * 1000))
                    continue

                if event is None:
                    self.msleep(int(monitor.poll_interval_sec * 1000))
                    continue

                if self._hotkey_enabled and not self._awaiting_capture.is_set():
                    self.msleep(45)
                    continue

                if self._hotkey_enabled:
                    self._awaiting_capture.clear()
                    self.status_update.emit("Clipboard image captured. Running detection...")

                try:
                    payload = client.predict_event(event)
                    self.prediction_ready.emit(payload, event.png_bytes)
                    if self._hotkey_enabled:
                        self.status_update.emit("Detection complete. Press hotkey for next capture.")
                except Exception as exc:
                    self.error_update.emit(f"API request error: {exc}")

                self.msleep(80)

            hotkey_listener.stop()

    class OverlayWindow(QtWidgets.QWidget):
        def __init__(self, api_url, model_key, hotkey):
            super().__init__()
            self.api_url = api_url
            self.model_key = model_key
            self.hotkey = hotkey
            self._drag_offset = None
            self._latest_pixmap = QtGui.QPixmap()
            self._is_compact_mode = False
            self._expanded_size = QtCore.QSize(336, 446)
            self._compact_size = QtCore.QSize(232, 56)

            self._setup_window()
            self._build_ui()
            self._move_to_top_right()
            self._start_worker()

        def _setup_window(self):
            self.setWindowTitle("Fruit Overlay")
            self.resize(self._expanded_size)
            self.setWindowFlags(
                Qt.WindowType.FramelessWindowHint
                | Qt.WindowType.WindowStaysOnTopHint
                | Qt.WindowType.Tool
            )
            self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)

        def _build_ui(self):
            root = QtWidgets.QVBoxLayout(self)
            root.setContentsMargins(6, 6, 6, 6)

            self.setStyleSheet(
                "QFrame#overlay_shell{" 
                "background:qlineargradient(x1:0,y1:0,x2:1,y2:1,stop:0 #ffffff, stop:1 #f2f6f9);"
                "border:1px solid #c9d6df;border-radius:16px;}"
                "QFrame#compact_shell{"
                "background:#f5f8fb;border:1px solid #c5d2db;border-radius:14px;}"
                "QPushButton#circle_btn{"
                "background:#e8eef3;border:1px solid #c6d2da;color:#234356;border-radius:11px;font-weight:700;}"
                "QPushButton#circle_btn:hover{background:#dbe7ef;}"
                "QPushButton#danger_btn{"
                "background:#ca3643;color:white;border:none;border-radius:11px;font-weight:700;}"
                "QPushButton#danger_btn:hover{background:#de4250;}"
                "QPushButton#ghost_btn{"
                "background:#dfeaf1;color:#143749;border:none;border-radius:10px;font-weight:700;padding:4px 10px;}"
                "QPushButton#ghost_btn:hover{background:#cfdfe9;}"
            )

            shell = QtWidgets.QFrame()
            shell.setObjectName("overlay_shell")

            layout = QtWidgets.QVBoxLayout(shell)
            layout.setContentsMargins(12, 12, 12, 12)
            layout.setSpacing(8)

            self.expanded_panel = QtWidgets.QWidget()
            expanded_layout = QtWidgets.QVBoxLayout(self.expanded_panel)
            expanded_layout.setContentsMargins(0, 0, 0, 0)
            expanded_layout.setSpacing(8)

            top_row = QtWidgets.QHBoxLayout()
            top_row.setContentsMargins(0, 0, 0, 0)

            title_box = QtWidgets.QVBoxLayout()
            title_box.setContentsMargins(0, 0, 0, 0)
            title_box.setSpacing(1)

            self.title_label = QtWidgets.QLabel("Fruit QA Overlay")
            self.title_label.setStyleSheet("font-size:15px;font-weight:700;color:#0f3446;")

            self.hotkey_hint_label = QtWidgets.QLabel(f"Hotkey: {self.hotkey}")
            self.hotkey_hint_label.setStyleSheet("font-size:11px;color:#4f6d7c;")

            title_box.addWidget(self.title_label)
            title_box.addWidget(self.hotkey_hint_label)

            compact_btn = QtWidgets.QPushButton("-")
            compact_btn.setObjectName("circle_btn")
            compact_btn.setFixedSize(22, 22)
            compact_btn.setToolTip("Thu gon")
            compact_btn.clicked.connect(lambda: self._set_compact_mode(True))

            close_btn = QtWidgets.QPushButton("x")
            close_btn.setObjectName("danger_btn")
            close_btn.setFixedSize(22, 22)
            close_btn.setToolTip("Dong overlay")

            def _close_overlay() -> None:
                self.close()

            close_btn.clicked.connect(_close_overlay)

            top_row.addLayout(title_box)
            top_row.addStretch(1)
            top_row.addWidget(compact_btn)
            top_row.addWidget(close_btn)

            self.warning_label = QtWidgets.QLabel("")
            self.warning_label.setWordWrap(True)
            self.warning_label.setStyleSheet(
                "background:#ffe7ea;color:#7e1f2f;padding:7px 10px;border-radius:10px;font-size:12px;font-weight:700;"
            )
            self.warning_label.hide()

            self.stats_label = QtWidgets.QLabel("Press hotkey to capture and detect fruit.")
            self.stats_label.setWordWrap(True)
            self.stats_label.setStyleSheet(
                "background:#e9f2f8;border:1px solid #c6d9e8;border-radius:10px;padding:7px 10px;color:#204357;font-size:12px;"
            )

            self.image_label = ClickableLabel()
            self.image_label.setFixedHeight(172)
            self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            self.image_label.setStyleSheet(
                "background:#dce8ef;border:1px solid #b3c5d1;border-radius:11px;color:#335162;font-size:12px;"
            )
            self.image_label.setText("No prediction image yet")
            self.image_label.clicked.connect(self._open_preview)

            self.class_label = QtWidgets.QLabel("Classes: -")
            self.class_label.setWordWrap(True)
            self.class_label.setStyleSheet("color:#1d4556;font-size:12px;")

            self.footer_label = QtWidgets.QLabel("Desktop pipeline running")
            self.footer_label.setWordWrap(True)
            self.footer_label.setStyleSheet("color:#4a6776;font-size:11px;")

            expanded_layout.addLayout(top_row)
            expanded_layout.addWidget(self.warning_label)
            expanded_layout.addWidget(self.stats_label)
            expanded_layout.addWidget(self.image_label)
            expanded_layout.addWidget(self.class_label)
            expanded_layout.addStretch(1)
            expanded_layout.addWidget(self.footer_label)

            self.compact_panel = QtWidgets.QFrame()
            self.compact_panel.setObjectName("compact_shell")
            compact_layout = QtWidgets.QHBoxLayout(self.compact_panel)
            compact_layout.setContentsMargins(10, 8, 10, 8)
            compact_layout.setSpacing(6)

            self.compact_state = QtWidgets.QLabel("READY")
            self.compact_state.setStyleSheet(
                "background:#1e7f5b;color:white;border-radius:8px;padding:2px 7px;font-size:10px;font-weight:700;"
            )
            self.compact_summary = QtWidgets.QLabel("Waiting hotkey")
            self.compact_summary.setStyleSheet("color:#1f3f52;font-size:11px;font-weight:600;")
            self.compact_summary.setWordWrap(False)

            open_btn = QtWidgets.QPushButton("Open")
            open_btn.setObjectName("ghost_btn")
            open_btn.setFixedHeight(22)
            open_btn.clicked.connect(lambda: self._set_compact_mode(False))

            compact_layout.addWidget(self.compact_state)
            compact_layout.addWidget(self.compact_summary, 1)
            compact_layout.addWidget(open_btn)
            self.compact_panel.hide()

            layout.addWidget(self.expanded_panel)
            layout.addWidget(self.compact_panel)
            root.addWidget(shell)

        def _move_to_top_right(self, extension_anchor=False):
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is None:
                return

            area = screen.availableGeometry()
            x_margin = 10 if extension_anchor else 22
            y_margin = 8 if extension_anchor else 22
            x = area.x() + area.width() - self.width() - x_margin
            y = area.y() + y_margin
            self.move(int(x), int(y))

        def _set_compact_mode(self, compact):
            compact = bool(compact)
            if compact == self._is_compact_mode:
                return

            old_top_right = self.frameGeometry().topRight()
            self._is_compact_mode = compact
            self.expanded_panel.setVisible(not compact)
            self.compact_panel.setVisible(compact)

            if compact:
                self.setFixedSize(self._compact_size)
                self._move_to_top_right(extension_anchor=True)
            else:
                self.setFixedSize(self._expanded_size)
                self._restore_position_from_anchor(old_top_right)
                self.raise_()
                self.activateWindow()

        def _restore_position_from_anchor(self, old_top_right):
            screen = QtWidgets.QApplication.primaryScreen()
            if screen is None:
                return

            area = screen.availableGeometry()
            x = int(old_top_right.x() - self.width() + 1)
            y = int(old_top_right.y() + 12)
            x = max(area.x() + 10, min(x, area.x() + area.width() - self.width() - 10))
            y = max(area.y() + 10, min(y, area.y() + area.height() - self.height() - 10))
            self.move(x, y)

        def _start_worker(self):
            self.worker = CaptureWorker(
                api_url=self.api_url,
                model_key=self.model_key,
                hotkey=self.hotkey,
                poll_interval_sec=0.35,
            )
            self.worker.prediction_ready.connect(self._on_prediction)
            self.worker.status_update.connect(self._on_status)
            self.worker.error_update.connect(self._on_error)
            self.worker.capture_requested.connect(self._on_capture_requested)
            self.worker.start()

        def _set_compact_text(self, text):
            compact_text = str(text).strip() or "Waiting"
            if len(compact_text) > 34:
                compact_text = compact_text[:31] + "..."
            self.compact_summary.setText(compact_text)

        def _on_status(self, text):
            self.footer_label.setText(str(text))
            self._set_compact_text(text)

        def _on_error(self, text):
            self.footer_label.setText(str(text))
            self.compact_state.setText("ERROR")
            self.compact_state.setStyleSheet(
                "background:#ad2f3f;color:white;border-radius:8px;padding:2px 7px;font-size:10px;font-weight:700;"
            )
            self._set_compact_text(text)

        def _on_capture_requested(self):
            self.footer_label.setText("Hotkey pressed. Waiting for clipboard capture...")
            self.compact_state.setText("CAPTURE")
            self.compact_state.setStyleSheet(
                "background:#3478c9;color:white;border-radius:8px;padding:2px 7px;font-size:10px;font-weight:700;"
            )
            self._set_compact_text("Waiting for clipboard image")

        def _on_prediction(self, payload, png_bytes):
            summary = build_overlay_summary(payload)
            self.title_label.setText(str(summary["title"]))
            self.stats_label.setText(str(summary["stats"]))

            warning_text = str(summary["warning"])
            if warning_text:
                self.warning_label.setText(warning_text)
                self.warning_label.show()
            else:
                self.warning_label.hide()

            class_lines = summary.get("class_lines", [])
            self.class_label.setText("Classes:\n" + "\n".join(class_lines or ["-"]))

            if summary["warning"]:
                self.compact_state.setText("ALERT")
                self.compact_state.setStyleSheet(
                    "background:#b13343;color:white;border-radius:8px;padding:2px 7px;font-size:10px;font-weight:700;"
                )
            else:
                self.compact_state.setText("OK")
                self.compact_state.setStyleSheet(
                    "background:#1e7f5b;color:white;border-radius:8px;padding:2px 7px;font-size:10px;font-weight:700;"
                )
            self._set_compact_text(summary["title"])

            try:
                image_bgr = decode_png_bytes_to_bgr(png_bytes)
                predictions = payload.get("predictions", []) or []
                drawn = draw_predictions_on_bgr(image_bgr, list(predictions))
                pixmap = self._bgr_to_pixmap(drawn)
                self._latest_pixmap = pixmap
                self._set_image_pixmap(pixmap)
            except Exception as exc:
                self.image_label.setText(f"Image render error: {exc}")

            if self._is_compact_mode:
                self._set_compact_mode(False)
            self.raise_()
            self.activateWindow()

        def _set_image_pixmap(self, pixmap):
            if pixmap.isNull():
                return
            self.image_label.setPixmap(
                pixmap.scaled(
                    self.image_label.size(),
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )

        def _bgr_to_pixmap(self, image_bgr):
            rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            h, w, c = rgb.shape
            qimg = QtGui.QImage(rgb.data, w, h, c * w, QtGui.QImage.Format.Format_RGB888).copy()
            return QtGui.QPixmap.fromImage(qimg)

        def _open_preview(self):
            if self._latest_pixmap.isNull():
                return
            dialog = PreviewDialog(self, self._latest_pixmap)
            dialog.exec()

        def resizeEvent(self, a0):
            super().resizeEvent(a0)
            if not self._latest_pixmap.isNull() and not self._is_compact_mode:
                self._set_image_pixmap(self._latest_pixmap)

        def closeEvent(self, a0):
            if hasattr(self, "worker") and self.worker is not None:
                self.worker.stop()
                self.worker.wait(1000)
            super().closeEvent(a0)

        def mousePressEvent(self, a0):
            if a0.button() == Qt.MouseButton.LeftButton and (
                self._is_compact_mode or a0.position().y() <= 52
            ):
                global_pos = a0.globalPosition().toPoint()
                frame_pos = self.frameGeometry().topLeft()
                self._drag_offset = (
                    int(global_pos.x() - frame_pos.x()),
                    int(global_pos.y() - frame_pos.y()),
                )
            else:
                self._drag_offset = None
            super().mousePressEvent(a0)

        def mouseMoveEvent(self, a0):
            if self._drag_offset is not None and a0.buttons() & Qt.MouseButton.LeftButton:
                global_pos = a0.globalPosition().toPoint()
                self.move(
                    int(global_pos.x() - self._drag_offset[0]),
                    int(global_pos.y() - self._drag_offset[1]),
                )
            super().mouseMoveEvent(a0)

        def mouseReleaseEvent(self, a0):
            self._drag_offset = None
            super().mouseReleaseEvent(a0)

    api_url = os.environ.get("DESKTOP_FASTAPI_URL", "http://127.0.0.1:8001")
    model_key = os.environ.get("DESKTOP_MODEL_KEY", "rfdetr_a3")
    hotkey = os.environ.get("DESKTOP_HOTKEY", "windows+shift+s")

    qt_app = QtWidgets.QApplication(sys.argv)
    window = OverlayWindow(api_url=api_url, model_key=model_key, hotkey=hotkey)
    window.show()
    return int(qt_app.exec())


def main() -> None:
    raise SystemExit(run_overlay())


if __name__ == "__main__":
    main()
