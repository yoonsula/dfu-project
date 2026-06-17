from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import gradio as gr
import numpy as np
import torch
from fastrtc import VideoStreamHandler, WebRTC
from PIL import Image

from inference.pipeline import SegmentationConfig
from inference.pipeline import render_overlay
from inference.pipeline import run_gated_segmentation
from infer import (
    load_model,
    resolve_image_size_from_checkpoint,
)
from paths import DINOV3_CHECKPOINT as DEFAULT_DINOV3_CHECKPOINT
from paths import DINOV3_REPO as DEFAULT_DINOV3_REPO
from utils.runtime import resolve_device
UI_REFRESH_SEC = 0.5


@dataclass(frozen=True)
class RuntimeConfig:
    foot_head_checkpoint: Path
    ulcer_head_checkpoint: Path
    dinov3_repo: Path
    dinov3_checkpoint: Path
    image_size: int
    display_max_size: int
    device_name: str
    foot_threshold: float
    ulcer_threshold: float
    min_foot_ratio: float
    max_foot_ratio: float
    center_tolerance: float
    min_ulcer_ratio: float
    overlay_alpha: float
    amp: bool
    stream_time_limit: int
    stream_skip_frames: bool
    panel_width: int
    panel_height: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DFU segmentation Gradio app (WebRTC).")
    parser.add_argument("--foot-head-checkpoint", type=Path, required=True)
    parser.add_argument("--ulcer-head-checkpoint", type=Path, required=True)
    parser.add_argument("--dinov3-repo", type=Path, default=DEFAULT_DINOV3_REPO)
    parser.add_argument("--dinov3-checkpoint", type=Path, default=DEFAULT_DINOV3_CHECKPOINT)
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Model input resolution (match training checkpoint, usually 384).",
    )
    parser.add_argument(
        "--display-max-size",
        type=int,
        default=512,
        help="Max edge length for overlay returned to browser (0 = webcam resolution).",
    )
    parser.add_argument("--stream-time-limit", type=int, default=3600, help="WebRTC session limit (seconds).")
    parser.add_argument(
        "--no-stream-skip-frames",
        action="store_true",
        help="Do not skip webcam frames while inference is running.",
    )
    parser.add_argument("--panel-width", type=int, default=480, help="Panel width (px); WebRTC / image column.")
    parser.add_argument(
        "--panel-height",
        type=int,
        default=400,
        help="Single image height (px). WebRTC height = 2x; JSON height = 2x + margin.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--foot-threshold", type=float, default=0.5)
    parser.add_argument("--ulcer-threshold", type=float, default=0.5)
    parser.add_argument("--min-foot-ratio", type=float, default=0.08)
    parser.add_argument("--max-foot-ratio", type=float, default=0.5)
    parser.add_argument("--center-tolerance", type=float, default=0.25)
    parser.add_argument("--min-ulcer-ratio", type=float, default=0.001)
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--server-name", type=str, default="127.0.0.1")
    parser.add_argument("--server-port", type=int, default=7861)
    parser.add_argument("--share", action="store_true")
    return parser.parse_args()


def downscale_for_display(image: Image.Image, max_size: int) -> Image.Image:
    if max_size <= 0:
        return image
    width, height = image.size
    scale = min(max_size / width, max_size / height, 1.0)
    if scale >= 1.0:
        return image
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    return image.resize((new_width, new_height), Image.Resampling.BILINEAR)


class RealtimeDFUSegmenter:
    def __init__(self, config: RuntimeConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device_name)
        self.model = load_model(self._to_infer_args(config), self.device)
        self.use_amp = bool(config.amp and self.device.type == "cuda")

        self._last_guidance = "이미지 탭: 업로드 후 Run. 실시간 탭: WebRTC Start."
        self._last_metrics: dict[str, Any] = {}
        self._webrtc_frames = 0
        self._last_json_serialized: str | None = None

    @staticmethod
    def _to_infer_args(config: RuntimeConfig) -> argparse.Namespace:
        return argparse.Namespace(
            foot_head_checkpoint=config.foot_head_checkpoint,
            ulcer_head_checkpoint=config.ulcer_head_checkpoint,
            dinov3_repo=config.dinov3_repo,
            dinov3_checkpoint=config.dinov3_checkpoint,
        )

    @staticmethod
    def _bgr_to_rgb(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return frame[..., ::-1].copy()
        return frame

    @staticmethod
    def _rgb_to_bgr(frame: np.ndarray) -> np.ndarray:
        if frame.ndim == 3 and frame.shape[-1] == 3:
            return frame[..., ::-1].copy()
        return frame

    def process_webrtc(self, frame: np.ndarray | None) -> np.ndarray | None:
        """
        FastRTC send-receive handler.
        Returns BGR frame directly to WebRTC (no gr.Image per-frame refresh).
        """
        if frame is None:
            return None

        overlay_rgb, guidance, metrics = self._infer_frame(self._bgr_to_rgb(frame))
        self._webrtc_frames += 1
        self._last_guidance = guidance
        self._last_metrics = metrics
        return self._rgb_to_bgr(overlay_rgb)

    def _result_payload(self) -> dict[str, Any] | None:
        if not self._last_metrics:
            return None
        return {"guidance": self._last_guidance, **self._last_metrics}

    def _json_output(self, force: bool = False) -> Any:
        payload = self._result_payload()
        if payload is None:
            return gr.skip()
        serialized = json.dumps(payload, sort_keys=True, default=str)
        if not force and serialized == self._last_json_serialized:
            return gr.skip()
        self._last_json_serialized = serialized
        return payload

    def predict_upload(
        self,
        frame: np.ndarray | None,
    ) -> tuple[Any, Any]:
        if frame is None:
            return None, gr.skip()
        # Gradio Image(type="numpy") is RGB; WebRTC frames are BGR (handled in process_webrtc).
        overlay, guidance, metrics = self._infer_frame(frame)
        self._last_guidance = guidance
        self._last_metrics = metrics
        return overlay, self._json_output(force=True)

    def predict_upload_tabs(self, frame: np.ndarray | None) -> tuple[Any, Any, Any]:
        overlay, result = self.predict_upload(frame)
        return overlay, result, result

    def read_cached_metrics_tabs(self) -> tuple[Any, Any]:
        result = self.read_cached_metrics()
        return result, result

    def read_cached_metrics(self) -> Any:
        return self._json_output(force=False)

    @torch.inference_mode()
    def _infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str, dict[str, Any]]:
        total_start = perf_counter()
        image = Image.fromarray(frame.astype(np.uint8), mode="RGB")
        display_image = downscale_for_display(image, self.config.display_max_size)
        display_size = display_image.size

        segmentation = run_gated_segmentation(
            self.model,
            image,
            SegmentationConfig(
                image_size=self.config.image_size,
                foot_threshold=self.config.foot_threshold,
                ulcer_threshold=self.config.ulcer_threshold,
                min_foot_ratio=self.config.min_foot_ratio,
                max_foot_ratio=self.config.max_foot_ratio,
                center_tolerance=self.config.center_tolerance,
                min_ulcer_ratio=self.config.min_ulcer_ratio,
            ),
            self.device,
            output_size=display_size,
            autocast_context=self._autocast_context,
        )
        overlay = np.asarray(
            render_overlay(
                display_image,
                segmentation.foot_mask,
                segmentation.ulcer_mask,
                self.config.overlay_alpha,
            ),
            dtype=np.uint8,
        )
        total_ms = (perf_counter() - total_start) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        metrics = {
            "device": str(self.device),
            "transport": "fastrtc",
            "model_image_size": self.config.image_size,
            "display_max_size": self.config.display_max_size,
            "display_width": display_size[0],
            "display_height": display_size[1],
            "foot_detected": segmentation.foot_detected,
            "foot_area_ratio": round(segmentation.foot_area_ratio, 4),
            "foot_centered": segmentation.foot_centered,
            "foot_center_x": round(segmentation.foot_center_x, 4)
            if segmentation.foot_center_x is not None
            else None,
            "foot_center_y": round(segmentation.foot_center_y, 4)
            if segmentation.foot_center_y is not None
            else None,
            "ulcer_enabled": segmentation.ulcer_enabled,
            "ulcer_detected": segmentation.ulcer_detected,
            "ulcer_area_ratio": round(segmentation.ulcer_area_ratio, 4),
            "preprocess_ms": round(segmentation.preprocess_ms, 2),
            "backbone_ms": round(segmentation.backbone_ms, 2),
            "foot_head_ms": round(segmentation.foot_head_ms, 2),
            "model_ms": round(segmentation.model_ms, 2),
            "ulcer_head_ms": round(segmentation.ulcer_head_ms, 2),
            "postprocess_ms": round(segmentation.postprocess_ms, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 2),
            "webrtc_frames": self._webrtc_frames,
        }
        return overlay, segmentation.capture_guidance, metrics

    def _autocast_context(self):
        if self.use_amp:
            return torch.amp.autocast(device_type="cuda")
        return torch.amp.autocast(device_type=self.device.type, enabled=False)


def build_app(
    segmenter: RealtimeDFUSegmenter,
    time_limit: int,
    skip_frames: bool,
    panel_width: int,
    panel_height: int,
) -> gr.Blocks:
    video_handler = VideoStreamHandler(
        segmenter.process_webrtc,
        skip_frames=skip_frames,
        fps=30,
    )

    # =========================================================
    # PANEL SIZE
    # =========================================================
    image_panel_height = panel_height
    live_panel_height = panel_height

    # JSON 영역 크게
    json_panel_height = int(panel_height * 1.5)

    media_column_min_width = panel_width + 48

    # =========================================================
    # CSS
    # =========================================================
    custom_css = f"""
    footer {{
        display: none !important;
    }}

    .gradio-container {{
        max-width: 1800px !important;
    }}

    /* JSON PANEL */
    #image-json,
    #live-json {{
        min-height: {json_panel_height}px !important;
    }}

    #image-json .json-container,
    #live-json .json-container {{
        min-height: {json_panel_height}px !important;
        max-height: {json_panel_height}px !important;
        overflow-y: auto !important;
    }}

    /* 이미지/영상 panel */
    .media-panel {{
        gap: 12px;
    }}

    /* 버튼 */
    .run-btn {{
        margin-top: 10px;
    }}

    /* section title */
    .section-title {{
        margin-bottom: 8px;
    }}

    /* JSON 자체 padding */
    #image-json,
    #live-json {{
        padding-bottom: 10px;
    }}

    /* 모바일 대응 */
    @media (max-width: 900px) {{
        .gradio-container {{
            padding: 8px !important;
        }}
    }}
    """

    # =========================================================
    # APP
    # =========================================================
    with gr.Blocks(
        title="DFU Realtime Segmentation (WebRTC)",
        css=custom_css,
    ) as app:

        gr.Markdown(
            """
            # DFU Realtime Foot / Ulcer Segmentation
            """,
            elem_classes=["section-title"],
        )

        # =====================================================
        # TABS
        # =====================================================
        with gr.Tabs():

            # =================================================
            # IMAGE TAB
            # =================================================
            with gr.Tab("이미지", id="tab_image"):

                # equal_height=False 중요
                with gr.Row(equal_height=False):

                    # LEFT
                    with gr.Column(
                        scale=1,
                        min_width=media_column_min_width,
                    ):

                        with gr.Column(elem_classes=["media-panel"]):

                            input_image = gr.Image(
                                label="Input",
                                sources=["upload", "webcam", "clipboard"],
                                type="numpy",
                                width=panel_width,
                                height=image_panel_height,
                            )

                            output_snapshot = gr.Image(
                                label="Overlay",
                                type="numpy",
                                width=panel_width,
                                height=image_panel_height,
                            )

                            run_button = gr.Button(
                                "Run Segmentation",
                                variant="primary",
                                size="lg",
                                elem_classes=["run-btn"],
                            )

                    # RIGHT
                    with gr.Column(
                        scale=1,
                        min_width=360,
                    ):

                        image_result_json = gr.JSON(
                            label="Result",
                            height=json_panel_height,
                            elem_id="image-json",
                        )

            # =================================================
            # REALTIME TAB
            # =================================================
            with gr.Tab("실시간", id="tab_realtime"):

                with gr.Row(equal_height=False):

                    # LEFT
                    with gr.Column(
                        scale=1,
                        min_width=media_column_min_width,
                    ):

                        output_live = WebRTC(
                            label="Realtime (WebRTC)",
                            mode="send-receive",
                            modality="video",
                            full_screen=False,
                            height=live_panel_height,
                            width=panel_width,
                        )

                    # RIGHT
                    with gr.Column(
                        scale=1,
                        min_width=360,
                    ):

                        live_result_json = gr.JSON(
                            label="Result",
                            height=json_panel_height,
                            elem_id="live-json",
                        )

        # =====================================================
        # STREAM
        # =====================================================
        output_live.stream(
            fn=video_handler,
            inputs=[output_live],
            outputs=[output_live],
            time_limit=time_limit,
        )

        # =====================================================
        # TIMER
        # =====================================================
        timer = gr.Timer(
            value=UI_REFRESH_SEC,
            active=True,
        )

        timer.tick(
            fn=segmenter.read_cached_metrics_tabs,
            outputs=[image_result_json, live_result_json],
            show_progress=False,
        )

        # =====================================================
        # UPLOAD INFERENCE
        # =====================================================
        run_button.click(
            fn=segmenter.predict_upload_tabs,
            inputs=input_image,
            outputs=[
                output_snapshot,
                image_result_json,
                live_result_json,
            ],
        )

    return app


def main() -> None:
    args = parse_args()
    image_size = resolve_image_size_from_checkpoint(args.foot_head_checkpoint, args.image_size)
    if image_size != args.image_size:
        print(f"Using image_size={image_size} from foot head checkpoint (CLI default was {args.image_size}).")
    config = RuntimeConfig(
        foot_head_checkpoint=args.foot_head_checkpoint,
        ulcer_head_checkpoint=args.ulcer_head_checkpoint,
        dinov3_repo=args.dinov3_repo,
        dinov3_checkpoint=args.dinov3_checkpoint,
        image_size=image_size,
        display_max_size=args.display_max_size,
        device_name=args.device,
        foot_threshold=args.foot_threshold,
        ulcer_threshold=args.ulcer_threshold,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
        center_tolerance=args.center_tolerance,
        min_ulcer_ratio=args.min_ulcer_ratio,
        overlay_alpha=args.overlay_alpha,
        amp=args.amp,
        stream_time_limit=args.stream_time_limit,
        stream_skip_frames=not args.no_stream_skip_frames,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
    )
    segmenter = RealtimeDFUSegmenter(config)
    app = build_app(
        segmenter,
        time_limit=config.stream_time_limit,
        skip_frames=config.stream_skip_frames,
        panel_width=config.panel_width,
        panel_height=config.panel_height,
    )
    app.launch(server_name=args.server_name, server_port=args.server_port, share=args.share)


if __name__ == "__main__":
    main()
