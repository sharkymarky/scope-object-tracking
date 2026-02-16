import torch
from scope.core.pipelines.base import BasePipeline
from scope.core.pipelines.base_schema import BasePipelineConfig

from .tracker import SimpleIoUTracker

class ObjectTrackingConfig(BasePipelineConfig):
    pipeline_id = "object-tracking"
    pipeline_name = "Object Tracking"
    pipeline_description = "Simple YOLO-based object tracking"

    detect_every: int = 4
    conf: float = 0.35
    iou_thresh: float = 0.3
    max_misses: int = 10


class ObjectTrackingPipeline(BasePipeline):

    Config = ObjectTrackingConfig

    def __init__(self, config: ObjectTrackingConfig):
        super().__init__(config)
        self.detector = None
        self.tracker = SimpleIoUTracker(
            iou_thresh=config.iou_thresh,
            max_misses=config.max_misses
        )
        self.frame_idx = 0
        self.last_dets = []

    def _init_detector(self, device):
        if self.detector is None:
            from ultralytics import YOLO
            self.detector = YOLO("yolov8n.pt")

    def __call__(self, **kwargs):
        frames = kwargs["video"]
        output = []

        for frame in frames:
            self._init_detector(frame.device)

            if self.frame_idx % self.config.detect_every == 0:
                img = (frame.clamp(0,1) * 255).to(torch.uint8).unsqueeze(0)
                results = self.detector.predict(
                    source=img,
                    conf=self.config.conf,
                    verbose=False
                )[0]

                self.last_dets = []
                if results.boxes is not None:
                    boxes = results.boxes.xyxy.cpu().tolist()
                    clses = results.boxes.cls.cpu().tolist()
                    names = results.names
                    for box, cls in zip(boxes, clses):
                        self.last_dets.append({
                            "bbox": box,
                            "label": names[int(cls)]
                        })

            tracks = self.tracker.update(self.last_dets)

            # For now we do not draw anything
            output.append(frame)

            self.frame_idx += 1

        return {"video": output}
