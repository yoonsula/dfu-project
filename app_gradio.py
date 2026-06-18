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
from infer import load_model
from inference.checkpoints import resolve_image_size_from_checkpoint
from inference.classification import classify_shared_features, load_dfu_head_bundle
from paths import DINOV3_MODEL_PATH as DEFAULT_DINOV3_MODEL_PATH
from utils.runtime import resolve_device
UI_REFRESH_SEC = 0.5


@dataclass(frozen=True)
class RuntimeConfig:
    foot_head_checkpoint: Path
    wound_head_checkpoint: Path
    dinov3_model: Path
    image_size: int
    display_max_size: int
    device_name: str
    foot_threshold: float
    wound_threshold: float
    guide_enabled: bool
    min_foot_ratio: float
    max_foot_ratio: float
    center_tolerance: float
    min_wound_ratio: float
    wound_feature_crop: bool
    wound_crop_margin: float
    overlay_alpha: float
    amp: bool
    stream_time_limit: int
    stream_skip_frames: bool
    panel_width: int
    panel_height: int
    dfu_head_checkpoint: Path | None
    classification_top_k: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch DFU segmentation Gradio app (WebRTC).")
    parser.add_argument("--foot-head-checkpoint", type=Path, required=True)
    parser.add_argument("--wound-head-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--dinov3-model",
        type=Path,
        default=DEFAULT_DINOV3_MODEL_PATH,
        help="Local Hugging Face snapshot directory for the frozen DINOv3 ViT-S/16 backbone.",
    )
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
    parser.add_argument("--wound-threshold", type=float, default=0.5)
    parser.add_argument(
        "--no-guide",
        action="store_true",
        help="Disable capture guidance and do not let guidance gates block wound stages.",
    )
    parser.add_argument("--min-foot-ratio", type=float, default=0.08)
    parser.add_argument("--max-foot-ratio", type=float, default=0.5)
    parser.add_argument("--center-tolerance", type=float, default=0.25)
    parser.add_argument("--min-wound-ratio", type=float, default=0.001)
    parser.add_argument("--wound-crop-margin", type=float, default=0.1)
    parser.add_argument("--no-wound-feature-crop", action="store_true")
    parser.add_argument("--overlay-alpha", type=float, default=0.4)
    parser.add_argument(
        "--dfu-head-checkpoint",
        type=Path,
        default=None,
        help="DFU classification head checkpoint that consumes shared DINOv3 feature maps.",
    )
    parser.add_argument(
        "--no-classification",
        action="store_true",
        help="Disable DFU classification even when --dfu-head-checkpoint is set.",
    )
    parser.add_argument("--classification-top-k", type=int, default=3)
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
        self.dfu_head_bundle = None
        if config.dfu_head_checkpoint is not None:
            self.dfu_head_bundle = load_dfu_head_bundle(config.dfu_head_checkpoint, self.device)
            print(f"Loaded DFU classification head: {config.dfu_head_checkpoint}")
        self.classification_top_k = config.classification_top_k

        self._last_guidance: str | None = (
            "이미지 탭: 업로드 후 Run. 실시간 탭: WebRTC Start."
            if config.guide_enabled
            else None
        )
        self._last_metrics: dict[str, Any] = {}
        self._webrtc_frames = 0
        self._last_json_serialized: str | None = None

    @staticmethod
    def _to_infer_args(config: RuntimeConfig) -> argparse.Namespace:
        return argparse.Namespace(
            foot_head_checkpoint=config.foot_head_checkpoint,
            wound_head_checkpoint=config.wound_head_checkpoint,
            dinov3_model=config.dinov3_model,
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
        if self._last_guidance is None:
            return dict(self._last_metrics)
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
    def _infer_frame(self, frame: np.ndarray) -> tuple[np.ndarray, str | None, dict[str, Any]]:
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
                wound_threshold=self.config.wound_threshold,
                guide_enabled=self.config.guide_enabled,
                min_foot_ratio=self.config.min_foot_ratio,
                max_foot_ratio=self.config.max_foot_ratio,
                center_tolerance=self.config.center_tolerance,
                min_wound_ratio=self.config.min_wound_ratio,
                wound_feature_crop=self.config.wound_feature_crop,
                wound_crop_margin=self.config.wound_crop_margin,
            ),
            self.device,
            output_size=display_size,
            autocast_context=self._autocast_context,
        )
        overlay = np.asarray(
            render_overlay(
                display_image,
                segmentation.foot_mask,
                segmentation.wound_mask,
                self.config.overlay_alpha,
                segmentation.wound_crop_bbox,
            ),
            dtype=np.uint8,
        )
        total_ms = (perf_counter() - total_start) * 1000.0
        fps = 1000.0 / total_ms if total_ms > 0 else 0.0

        classification_result = classify_shared_features(
            segmentation.features,
            self.dfu_head_bundle,
            enabled=self.dfu_head_bundle is not None and segmentation.foot_detected,
            top_k=self.classification_top_k,
        )
        classification_top_k = [
            {"class_name": score.class_name, "probability": round(score.probability, 4)}
            for score in classification_result.top_k
        ]

        metrics = {
            "device": str(self.device),
            "transport": "fastrtc",
            "model_image_size": self.config.image_size,
            "guide_enabled": self.config.guide_enabled,
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
            "wound_enabled": segmentation.wound_enabled,
            "wound_detected": segmentation.wound_detected,
            "wound_area_ratio": round(segmentation.wound_area_ratio, 4),
            "wound_crop_bbox": segmentation.wound_crop_bbox,
            "wound_feature_crop": self.config.wound_feature_crop,
            "wound_crop_margin": self.config.wound_crop_margin,
            "preprocess_ms": round(segmentation.preprocess_ms, 2),
            "backbone_ms": round(segmentation.backbone_ms, 2),
            "foot_head_ms": round(segmentation.foot_head_ms, 2),
            "model_ms": round(segmentation.model_ms, 2),
            "wound_head_ms": round(segmentation.wound_head_ms, 2),
            "postprocess_ms": round(segmentation.postprocess_ms, 2),
            "total_ms": round(total_ms, 2),
            "fps": round(fps, 2),
            "webrtc_frames": self._webrtc_frames,
            "classification_enabled": classification_result.enabled,
            "classification_predicted_class": classification_result.predicted_class,
            "classification_confidence": (
                round(classification_result.confidence, 4)
                if classification_result.confidence is not None
                else None
            ),
            "classification_top_k": classification_top_k,
            "classification_ms": classification_result.classification_ms,
            "classification_checkpoint_path": classification_result.checkpoint_path,
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
            # DFU Realtime Foot / Wound Segmentation + Classification
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
        wound_head_checkpoint=args.wound_head_checkpoint,
        dinov3_model=args.dinov3_model,
        image_size=image_size,
        display_max_size=args.display_max_size,
        device_name=args.device,
        foot_threshold=args.foot_threshold,
        wound_threshold=args.wound_threshold,
        guide_enabled=not args.no_guide,
        min_foot_ratio=args.min_foot_ratio,
        max_foot_ratio=args.max_foot_ratio,
        center_tolerance=args.center_tolerance,
        min_wound_ratio=args.min_wound_ratio,
        wound_feature_crop=not args.no_wound_feature_crop,
        wound_crop_margin=args.wound_crop_margin,
        overlay_alpha=args.overlay_alpha,
        amp=args.amp,
        stream_time_limit=args.stream_time_limit,
        stream_skip_frames=not args.no_stream_skip_frames,
        panel_width=args.panel_width,
        panel_height=args.panel_height,
        dfu_head_checkpoint=None if args.no_classification else args.dfu_head_checkpoint,
        classification_top_k=args.classification_top_k,
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
