"""
物体検知（土台・触らないでください）

YOLOX（Apache-2.0）の ONNX モデルを ONNX Runtime で動かす。
PyTorch を使わないので、Windows のスマートアプリコントロールが有効な PC でも動く。
Jetson などで onnxruntime-gpu が入っていれば、自動で GPU を使う。
"""
import cv2
import numpy as np
import onnxruntime as ort

COCO_CLASSES = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
)

PREFERRED_PROVIDERS = ("CUDAExecutionProvider", "CPUExecutionProvider")
NMS_IOU = 0.45


class Detector:
    def __init__(self, model_path):
        available = ort.get_available_providers()
        providers = [p for p in PREFERRED_PROVIDERS if p in available]
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.device = "GPU" if "CUDAExecutionProvider" in self.session.get_providers() else "CPU"

        model_input = self.session.get_inputs()[0]
        self.input_name = model_input.name
        self.size = model_input.shape[2]

        grids, strides = [], []
        for stride in (8, 16, 32):
            n = self.size // stride
            xs, ys = np.meshgrid(np.arange(n), np.arange(n))
            grids.append(np.stack((xs, ys), axis=2).reshape(-1, 2))
            strides.append(np.full((n * n, 1), stride))
        self.grids = np.concatenate(grids).astype(np.float32)
        self.strides = np.concatenate(strides).astype(np.float32)

    def __call__(self, frame, conf, class_ids):
        """frame（BGR画像）から検出し、[(x1, y1, x2, y2, score, class_id), ...] を返す"""
        h, w = frame.shape[:2]
        ratio = min(self.size / h, self.size / w)
        nh, nw = int(h * ratio), int(w * ratio)
        padded = np.full((self.size, self.size, 3), 114, dtype=np.uint8)
        padded[:nh, :nw] = cv2.resize(frame, (nw, nh), interpolation=cv2.INTER_LINEAR)
        blob = padded.transpose(2, 0, 1)[np.newaxis].astype(np.float32)

        out = self.session.run(None, {self.input_name: blob})[0][0]
        centers = (out[:, :2] + self.grids) * self.strides
        sizes = np.exp(out[:, 2:4]) * self.strides
        scores_all = out[:, 4:5] * out[:, 5:]
        classes = scores_all.argmax(axis=1)
        scores = scores_all[np.arange(len(classes)), classes]

        keep = (scores >= conf) & np.isin(classes, class_ids)
        if not keep.any():
            return []
        centers, sizes, scores, classes = centers[keep], sizes[keep], scores[keep], classes[keep]
        boxes = np.concatenate((centers - sizes / 2, centers + sizes / 2), axis=1) / ratio

        results = []
        for i in _nms(boxes, scores, classes):
            x1, y1, x2, y2 = boxes[i]
            results.append((
                int(max(0, x1)), int(max(0, y1)), int(min(w - 1, x2)), int(min(h - 1, y2)),
                float(scores[i]), int(classes[i]),
            ))
        return results


def _nms(boxes, scores, classes):
    """クラスごとに、重なった枠のうち一番自信のあるものだけ残す"""
    offset = classes[:, None] * 10000.0  # クラスが違う枠同士は重ならないようにずらす
    b = boxes + offset
    areas = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    order = scores.argsort()[::-1]
    kept = []
    while order.size > 0:
        i = order[0]
        kept.append(i)
        xx1 = np.maximum(b[i, 0], b[order[1:], 0])
        yy1 = np.maximum(b[i, 1], b[order[1:], 1])
        xx2 = np.minimum(b[i, 2], b[order[1:], 2])
        yy2 = np.minimum(b[i, 3], b[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter)
        order = order[1:][iou <= NMS_IOU]
    return kept
