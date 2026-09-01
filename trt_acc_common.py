#!/usr/bin/env python3
"""Pre/post-processing shared **verbatim** by the danlu reference run and the Jetson
TensorRT run, so an output difference can only come from the engine.

Why this file exists at all
--------------------------
The reported mAP came from PyTorch FP32 on danlu; the latency came from a TRT FP16
engine on the Jetson. Nothing has ever checked that the FP16 engine still produces
those boxes. To check it we must feed both sides the *same* tensor, and the two
machines do not agree on library versions:

    danlu   cv2 5.0.0   torch 2.6.0   torchvision 0.21.0   python 3.11
    Jetson  cv2 4.13.0  torch 2.1.0   torchvision 0.16.0   python 3.8

So the preprocessing is written once, shipped to both, and then *verified* by
diffing a few danlu-produced tensors against the Jetson's own reproduction
(`run_trt_infer.py --verify-pre`). An assumption of equality is not good enough
across a cv2 major version; it has to be measured.

Python 3.8 / numpy 1.19 compatible (that is what the Jetson has).

Each model mirrors its OWN eval path, which are genuinely different:

YOLOv13-s  ultralytics `model.predict()`
    cv2 BGR read -> LetterBox(aspect-preserving, pad 114) -> BGR2RGB -> CHW -> /255
    graph output [1,5,8400] = cxcywh in letterbox pixel space + sigmoid'd class score
    NMS is NOT in the graph, so it lives here.

RF-DETR-small  `RFDETR.predict()` (rfdetr/detr.py:2275-2279)
    PIL RGB -> to_tensor -> resize [512,512] antialias=False (NOT aspect-preserving,
    the image is squashed) -> normalize(ImageNet mean/std)
    graph output dets [1,300,4] = normalised cxcywh, labels [1,300,2] = raw logits.

    NOTE: rfdetr/export/_onnx/inference.py preprocesses with PIL BILINEAR and claims
    it matches torchvision's default. For *downscaling* it does not - PIL BILINEAR
    applies a resampling filter (equivalent to antialias=True) while predict() passes
    antialias=False. 1920->512 is a 3.75x downscale, so the two differ. We follow
    predict(), because predict() is what produced the reported mAP.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

# --- constants lifted from the two eval paths, not invented here ---------------
YOLO_IMGSZ = 640
YOLO_STRIDE = 32
YOLO_PAD_VALUE = 114  # ultralytics LetterBox: value=(114, 114, 114)
YOLO_CONF_EXPORT = 0.001  # eval_yolov13.py:50 export_thr - mAP needs the full PR curve
YOLO_IOU = 0.7  # cfg/default.yaml - predict() default, eval did not override
YOLO_MAX_DET = 300  # cfg/default.yaml
RF_RES = 512  # training_config.json resolution
RF_MEANS = (0.485, 0.456, 0.406)  # rfdetr/detr.py:367
RF_STDS = (0.229, 0.224, 0.225)  # rfdetr/detr.py:368
RF_CONF_EXPORT = 0.001  # eval_rfdetr.py:270 export_thr


def self_sha256() -> str:
    """Hash of this very file, so both sides can prove they ran identical code."""
    return hashlib.sha256(Path(__file__).read_bytes()).hexdigest()[:16]


def list_images(img_dir):
    """Deterministic image order. Both sides index into this same sorted list."""
    exts = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(p for p in Path(img_dir).iterdir() if p.suffix.lower() in exts)


# ------------------------------------------------------------------ YOLOv13-s --
def letterbox(im, new_shape=(640, 640), auto=False, scaleup=True, center=True,
              stride=YOLO_STRIDE):
    """Port of ultralytics LetterBox.__call__ (third_party/yolov13/.../augment.py).

    `auto` is the whole reason this takes a flag. The predictor sets
    `auto = same_shapes and self.model.pt` (predictor.py:158), so evaluating a .pt
    model letterboxes to the nearest stride multiple - 640x512 / 640x384 for this
    dataset - while the exported ONNX is frozen at 640x640. Same content scale
    (both use r = 640/1920 = 0.3333), different amount of grey padding. Keeping the
    flag lets us measure that geometry gap instead of guessing at it.
    """
    import cv2

    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:
        r = min(r, 1.0)

    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))  # (w, h)
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    if center:
        dw /= 2
        dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top = int(round(dh - 0.1)) if center else 0
    bottom = int(round(dh + 0.1))
    left = int(round(dw - 0.1)) if center else 0
    right = int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                            value=(YOLO_PAD_VALUE,) * 3)
    return im, r, (left, top)


def preprocess_yolo(img_path, imgsz=YOLO_IMGSZ, auto=False):
    """-> dict with x [1,3,H,W] float32 in [0,1], plus what postprocess needs."""
    import cv2

    im0 = cv2.imread(str(img_path))  # BGR, as ultralytics reads it
    if im0 is None:
        raise IOError("cv2.imread failed: %s" % img_path)
    orig_hw = im0.shape[:2]
    im, r, pad = letterbox(im0, (imgsz, imgsz), auto=auto)
    # predictor.py:126-133 - BGR2RGB, HWC2CHW, contiguous, then /255
    x = im[..., ::-1].transpose(2, 0, 1)
    x = np.ascontiguousarray(x, dtype=np.float32) / 255.0
    return {"x": x[None], "ratio": float(r), "pad": pad, "orig_hw": orig_hw}


def _nms_xyxy(boxes, scores, iou_thres):
    """torchvision.ops.nms - the same kernel ultralytics itself calls, so NMS is
    not a source of divergence between the two sides. Falls back to a numpy
    implementation only if torchvision is missing."""
    if boxes.shape[0] == 0:
        return np.empty((0,), dtype=np.int64)
    try:
        import torch
        from torchvision.ops import nms

        keep = nms(torch.from_numpy(boxes).float(),
                   torch.from_numpy(scores).float(), iou_thres)
        return keep.numpy().astype(np.int64)
    except ImportError:
        order = scores.argsort()[::-1]
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        keep = []
        while order.size:
            i = order[0]
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
            yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
            xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
            yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
            inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
            iou = inter / (area[i] + area[rest] - inter + 1e-9)
            order = rest[iou <= iou_thres]
        return np.asarray(keep, dtype=np.int64)


def postprocess_yolo(raw, meta, conf=YOLO_CONF_EXPORT, iou=YOLO_IOU,
                     max_det=YOLO_MAX_DET, max_nms=30000):
    """[1,5,8400] -> (xyxy in ORIGINAL image pixels, scores), single class.

    Mirrors ultralytics ops.non_max_suppression + ops.scale_boxes for nc=1.
    Un-letterboxing uses the pad/ratio this module produced rather than
    recomputing them, so the two cannot drift apart.
    """
    p = np.asarray(raw, dtype=np.float32)
    if p.ndim == 3:
        p = p[0]
    box, score = p[:4].T, p[4]  # (8400,4) cxcywh in letterbox px, (8400,)

    m = score > conf
    box, score = box[m], score[m]
    if box.shape[0] > max_nms:  # ops.non_max_suppression max_nms guard
        idx = score.argsort()[::-1][:max_nms]
        box, score = box[idx], score[idx]
    if box.shape[0] == 0:
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    cx, cy, w, h = box.T
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)

    keep = _nms_xyxy(xyxy, score, iou)[:max_det]
    xyxy, score = xyxy[keep], score[keep]

    # ops.scale_boxes: undo pad, undo gain, clip to the original frame
    left, top = meta["pad"]
    xyxy[:, [0, 2]] -= left
    xyxy[:, [1, 3]] -= top
    xyxy /= meta["ratio"]
    h0, w0 = meta["orig_hw"]
    xyxy[:, [0, 2]] = xyxy[:, [0, 2]].clip(0, w0)
    xyxy[:, [1, 3]] = xyxy[:, [1, 3]].clip(0, h0)
    return xyxy.astype(np.float32), score.astype(np.float32)


# --------------------------------------------------------------- RF-DETR-small --
def preprocess_rf(img_path, res=RF_RES):
    """-> dict with x [1,3,512,512] float32, normalised. Mirrors RFDETR.predict().

    Uses torchvision so the resize is the exact op predict() used. The image is
    squashed to a square (no aspect preservation, no padding), which is why RF -
    unlike YOLO - has no eval-vs-engine geometry gap: predict() and the ONNX both
    see 512x512.
    """
    import torch
    import torchvision.transforms.functional as F
    from PIL import Image

    with Image.open(str(img_path)) as im:
        im = im.convert("RGB")
        ow, oh = im.size
        t = F.to_tensor(im)  # CHW float32 in [0,1], RGB
    t = F.resize(t, [res, res], antialias=False)  # detr.py:2278
    t = F.normalize(t, list(RF_MEANS), list(RF_STDS))  # detr.py:2279
    return {"x": t.unsqueeze(0).numpy().astype(np.float32), "orig_wh": (ow, oh)}


def postprocess_rf(dets, labels, meta, conf=RF_CONF_EXPORT):
    """dets [1,300,4] normalised cxcywh + labels [1,300,2] logits -> (xyxy, scores).

    Mirrors rfdetr/export/_onnx/inference.py:219-246: drop the trailing no-object
    logit column, per-class sigmoid (not softmax), then scale normalised corners by
    the ORIGINAL image size - correct precisely because preprocessing squashed the
    whole frame, so there is no padding to undo.
    """
    b = np.asarray(dets, dtype=np.float32)
    lg = np.asarray(labels, dtype=np.float32)
    if b.ndim == 3:
        b = b[0]
    if lg.ndim == 3:
        lg = lg[0]
    lg = lg[:, :-1]  # drop DETR's background/no-object slot

    scores_all = 1.0 / (1.0 + np.exp(-np.clip(lg, -88, 88)))
    score = scores_all.max(axis=-1)
    m = score > conf
    if not m.any():
        return np.zeros((0, 4), np.float32), np.zeros((0,), np.float32)

    cx, cy, bw, bh = b[m].T
    xyxy = np.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    ow, oh = meta["orig_wh"]
    xyxy *= np.array([ow, oh, ow, oh], dtype=np.float32)
    return xyxy.astype(np.float32), score[m].astype(np.float32)


# ------------------------------------------------------------------ drift ------
def drift(a, b):
    """Numerical agreement between a reference tensor and an engine tensor.

    Reported together on purpose:
      max_abs        worst single element - what a spot check would catch
      mean_abs       typical error
      rel_l2         ||b-a||/||a|| - scale-free, comparable across tensors
      cos            direction agreement; drops well before max_abs looks alarming
      n_nonfinite    the decomposed-LayerNorm FP16 overflow signature. Any nonzero
                     value here is a hard fail, no matter how good the rest looks.
    """
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError("shape mismatch %s vs %s" % (a.shape, b.shape))
    n_bad = int((~np.isfinite(b)).sum())
    fin = np.isfinite(a) & np.isfinite(b)
    if not fin.any():
        return {"max_abs": float("nan"), "mean_abs": float("nan"),
                "rel_l2": float("nan"), "cos": float("nan"), "n_nonfinite": n_bad}
    a, b = a[fin], b[fin]
    d = np.abs(b - a)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return {
        "max_abs": float(d.max()),
        "mean_abs": float(d.mean()),
        "rel_l2": float(np.linalg.norm(b - a) / (na + 1e-12)),
        "cos": float(a.dot(b) / (na * nb + 1e-12)),
        "n_nonfinite": n_bad,
    }
