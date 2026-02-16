# src/scope_object_tracking/pipeline.py
from typing import TYPE_CHECKING
import torch
from scope.core.pipelines.interface import Pipeline, Requirements

if TYPE_CHECKING:
    from scope.core.pipelines.base_schema import BasePipelineConfig

class ObjectTrackingConfig:
    # If you already have a Pydantic config pattern, keep it.
    # I’m keeping this minimal; the key is the Pipeline interface usage.
    pipeline_id = "object-tracking"
    pipeline_name = "Object Tracking"
    pipeline_description = "Simple object tracking (data only for now)"

    conf: float = 0.35
    detect_every: int = 4
    iou_thresh: float = 0.3
    max_misses: int = 10


from .tracker import SimpleIoUTracker

class ObjectTrackingPipeline(Pipeline):
    CONFIG_CLASS: type["BasePipelineConfig"] = ObjectTrackingConfig  # Scope expects this pattern

    @classmethod
    def get_config_class(cls) -> type["BasePipelineConfig"]:
        return cls.CONFIG_CLASS

    def __init__(self, device: torch.device | None = None, **kwargs):
        self.device = device if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.detector = None
        self.tracker = SimpleIoUTracker(
            iou_thresh=getattr(self.CONFIG_CLASS, "iou_thresh", 0.3),
            max_misses=getattr(self.CONFIG_CLASS, "max_misses", 10),
        )
        self.frame_idx = 0
        self.last_dets = []
        self._detect_every = getattr(self.CONFIG_CLASS, "detect_every", 4)
        self._conf = getattr(self.CONFIG_CLASS, "conf", 0.35)

    def prepare(self, **kwargs) -> Requirements:
        # Per-frame pipeline
        return Requirements(input_size=1)

    def _init_detector(self):
        if self.detector is None:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8n.pt")

    @torch.no_grad()
    def __call__(self, **kwargs) -> dict:
        video = kwargs.get("video")
        if video is None:
            raise ValueError("object-tracking requires video input")

        # Scope frames are typically list of tensors shaped (1,H,W,C) in [0,255]
        frame = video[0].squeeze(0)  # (H,W,C)
        frame_u8 = frame.to(device=self.device, dtype=torch.uint8)

        # Run detection every N frames
        if (self.frame_idx % int(self._detect_every)) == 0:
            self._init_detector()

            # Ultralytics can take torch BCHW; convert HWC -> CHW -> BCHW
            img = frame_u8.permute(2, 0, 1).unsqueeze(0)  # (1,C,H,W)

            r = self.detector.predict(source=img, conf=float(self._conf), verbose=False)[0]
            dets = []
            if r.boxes is not None:
                boxes = r.boxes.xyxy.detach().cpu().tolist()
                clses = r.boxes.cls.detach().cpu().tolist()
                names = r.names
                for box, cls_id in zip(boxes, clses):
                    dets.append({"bbox": box, "label": str(names.get(int(cls_id), int(cls_id)))})
            self.last_dets = dets

        tracks = self.tracker.update(self.last_dets)

        # For now: return video unchanged (data-only)
        # If Scope supports side-metadata, later we can return {"video": ..., "tracking": tracks}
        self.frame_idx += 1

        # Return in [0,1] THWC expected by Scope
        out = frame_u8.to(torch.float32) / 255.0
        out = out.unsqueeze(0)  # (1,H,W,C)
        return {"video": out}
