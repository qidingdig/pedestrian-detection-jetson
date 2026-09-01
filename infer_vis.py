#!/usr/bin/env python3
"""Jetson TensorRT FP32 pedestrian infer + vis.

Folder or single image in; vis jpg + predictions.json out.
Uses the board's system TensorRT 8.5.2 / torch 2.1 (do NOT pip install torch).

  python3 infer_vis.py --model yolo --source /path/to/images --out ./out_yolo
  python3 infer_vis.py --model rf   --source /path/to/images --out ./out_rf

YOLO graph has no NMS: decode + NMS here (conf 0.25, iou 0.7).
RF-DETR has no NMS: score threshold only.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import trt_acc_common as C  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

PACK_ENGINE = {
    "yolo": "yolov13s_fp32.engine",
    "rf": "rfdetr-small-gather_fp32.engine",
}


def trt_to_torch_dtype(dt):
    import tensorrt as trt
    import torch

    return {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }[dt]


class Engine(object):
    """Synchronous TRT 8.5 binding-v2 wrapper (same as accuracy_check/run_trt_infer.py)."""

    def __init__(self, path):
        import tensorrt as trt
        import torch

        self.trt, self.torch = trt, torch
        logger = trt.Logger(trt.Logger.WARNING)
        trt.init_libnvinfer_plugins(logger, "")
        with open(path, "rb") as f:
            blob = f.read()
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError(
                "deserialize failed - wrong GPU arch or TRT version? %s" % path
            )
        self.ctx = self.engine.create_execution_context()
        self.inputs, self.outputs, self.buffers = [], [], []
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = tuple(self.engine.get_binding_shape(i))
            dtype = trt_to_torch_dtype(self.engine.get_binding_dtype(i))
            if any(d < 0 for d in shape):
                raise RuntimeError("dynamic shape %s on %s" % (shape, name))
            buf = torch.empty(shape, dtype=dtype, device="cuda")
            self.buffers.append(buf)
            slot = self.inputs if self.engine.binding_is_input(i) else self.outputs
            slot.append({"idx": i, "name": name, "shape": shape, "dtype": dtype})

    def __call__(self, x):
        b = self.inputs[0]
        src = self.torch.from_numpy(np.ascontiguousarray(x))
        self.buffers[b["idx"]].copy_(src.to(self.buffers[b["idx"]].dtype))
        ok = self.ctx.execute_v2([int(t.data_ptr()) for t in self.buffers])
        if not ok:
            raise RuntimeError("execute_v2 returned False")
        self.torch.cuda.synchronize()
        return {
            o["name"]: self.buffers[o["idx"]].float().cpu().numpy()
            for o in self.outputs
        }


def list_images(source):
    source = Path(source)
    if source.is_file():
        return [source]
    if not source.is_dir():
        raise SystemExit("source not found: %s" % source)
    imgs = sorted(p for p in source.iterdir() if p.suffix.lower() in IMG_EXT)
    if not imgs:
        raise SystemExit("no images under %s" % source)
    return imgs


def draw_boxes(im, dets, color=(0, 220, 0)):
    import cv2

    vis = np.ascontiguousarray(im.copy())
    for d in dets:
        x1, y1, x2, y2 = (int(round(v)) for v in d["xyxy"])
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        label = "person %.2f" % d["score"]
        cv2.putText(
            vis, label, (x1, max(0, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA,
        )
    return vis


def to_dets(xyxy, scores):
    out = []
    for box, s in zip(xyxy, scores):
        out.append({
            "xyxy": [float(v) for v in box],
            "score": float(s),
            "cls": "person",
        })
    return out


def main():
    ap = argparse.ArgumentParser(
        description="Jetson TRT FP32 infer + vis (YOLOv13-s / RF-DETR-small Gather)"
    )
    ap.add_argument("--model", required=True, choices=["yolo", "rf"])
    ap.add_argument("--source", required=True, help="image file or directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--engine", default="", help="override engine path")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--iou", type=float, default=0.7, help="YOLO NMS IoU")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    tmp = os.environ.get("TMPDIR", "")
    if tmp.startswith("/tmp") or tmp == "" or tmp == "/var/tmp":
        if not str(HERE).startswith("/ssd"):
            raise SystemExit(
                "unpack this pack under /ssd (root partition is full), "
                "or: export TMPDIR=/ssd/<your-dir>/tmp"
            )
        local_tmp = HERE / "tmp"
        local_tmp.mkdir(parents=True, exist_ok=True)
        os.environ["TMPDIR"] = str(local_tmp)
        print("TMPDIR -> %s" % os.environ["TMPDIR"], flush=True)
    else:
        Path(tmp).mkdir(parents=True, exist_ok=True)

    if args.engine:
        engine_path = args.engine
    else:
        engine_path = str(HERE / "engines" / PACK_ENGINE[args.model])
    if not Path(engine_path).is_file():
        raise SystemExit(
            "engine not found: %s\n"
            "Keep this pack's engines/ next to infer_vis.py, or pass --engine."
            % engine_path
        )

    images = list_images(args.source)
    if args.limit:
        images = images[: args.limit]
    out = Path(args.out)
    vis_dir = out / "vis"
    vis_dir.mkdir(parents=True, exist_ok=True)

    import cv2

    print(
        "model=%s  engine=%s  n=%d  conf=%.2f  iou=%.2f"
        % (args.model, engine_path, len(images), args.conf, args.iou),
        flush=True,
    )
    eng = Engine(engine_path)
    # warmup: first execute pays lazy kernel load
    if args.model == "yolo":
        eng(C.preprocess_yolo(images[0])["x"])
    else:
        eng(C.preprocess_rf(images[0])["x"])

    rows = []
    walls = []
    n_nan = 0
    t0 = time.time()
    for i, p in enumerate(images):
        im0 = cv2.imread(str(p))
        if im0 is None:
            raise SystemExit("failed to read: %s" % p)
        if args.model == "yolo":
            d = C.preprocess_yolo(p)
            t1 = time.time()
            y = eng(d["x"])
            walls.append((time.time() - t1) * 1000.0)
            raw = np.asarray(list(y.values())[0], dtype=np.float32)
            n_nan += int((~np.isfinite(raw)).sum())
            meta = {"ratio": d["ratio"], "pad": d["pad"], "orig_hw": d["orig_hw"]}
            xyxy, sc = C.postprocess_yolo(raw, meta, conf=args.conf, iou=args.iou)
            model_name = "YOLOv13-s"
        else:
            d = C.preprocess_rf(p)
            t1 = time.time()
            y = eng(d["x"])
            walls.append((time.time() - t1) * 1000.0)
            dets_t = np.asarray(y["dets"], dtype=np.float32)
            lab_t = np.asarray(y["labels"], dtype=np.float32)
            n_nan += int((~np.isfinite(dets_t)).sum() + (~np.isfinite(lab_t)).sum())
            meta = {"orig_wh": d["orig_wh"]}
            xyxy, sc = C.postprocess_rf(dets_t, lab_t, meta, conf=args.conf)
            model_name = "RF-DETR-small"
        dets = to_dets(xyxy, sc)
        vis = draw_boxes(im0, dets)
        cv2.imwrite(str(vis_dir / p.name), vis)
        rows.append({"file": p.name, "model": model_name, "n": len(dets), "dets": dets})
        if (i + 1) % 20 == 0 or (i + 1) == len(images):
            print(
                "  %d/%d  wall %.1f ms/img  boxes=%d"
                % (i + 1, len(images), float(np.mean(walls)), len(dets)),
                flush=True,
            )

    (out / "predictions.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    summary = {
        "model": args.model,
        "engine": engine_path,
        "n_images": len(images),
        "conf": args.conf,
        "iou": args.iou if args.model == "yolo" else None,
        "n_nonfinite": n_nan,
        "n_boxes_total": int(sum(r["n"] for r in rows)),
        "n_boxes_mean": float(np.mean([r["n"] for r in rows])),
        "wall_ms_mean": float(np.mean(walls)),
        "elapsed_sec": round(time.time() - t0, 1),
        "vis_dir": str(vis_dir),
        "common_sha256_16": C.self_sha256(),
    }
    (out / "run.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        "DONE nonfinite=%d  boxes_total=%d  wall_mean=%.1f ms  -> %s"
        % (n_nan, summary["n_boxes_total"], summary["wall_ms_mean"], out),
        flush=True,
    )
    if n_nan:
        raise SystemExit("non-finite outputs: engine is not usable")


if __name__ == "__main__":
    main()
