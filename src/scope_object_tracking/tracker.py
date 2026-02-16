def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    iw = max(0.0, inter_x2 - inter_x1)
    ih = max(0.0, inter_y2 - inter_y1)
    inter = iw * ih

    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)

    union = area_a + area_b - inter + 1e-9
    return inter / union


class SimpleIoUTracker:
    def __init__(self, iou_thresh=0.3, max_misses=10):
        self.iou_thresh = iou_thresh
        self.max_misses = max_misses
        self.next_id = 1
        self.tracks = []

    def update(self, detections):
        for t in self.tracks:
            t["_matched"] = False

        for det in detections:
            best_iou = 0
            best_track = None

            for t in self.tracks:
                if t["_matched"]:
                    continue
                v = iou_xyxy(det["bbox"], t["bbox"])
                if v > best_iou:
                    best_iou = v
                    best_track = t

            if best_track and best_iou > self.iou_thresh:
                best_track["bbox"] = det["bbox"]
                best_track["misses"] = 0
                best_track["_matched"] = True
            else:
                self.tracks.append({
                    "id": self.next_id,
                    "bbox": det["bbox"],
                    "misses": 0,
                    "_matched": True
                })
                self.next_id += 1

        new_tracks = []
        for t in self.tracks:
            if not t["_matched"]:
                t["misses"] += 1
            if t["misses"] <= self.max_misses:
                new_tracks.append(t)

        self.tracks = new_tracks

        for t in self.tracks:
            t.pop("_matched", None)

        return self.tracks
