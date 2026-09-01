# Jetson TensorRT FP32 行人推理（CP100 · RF-DETR-small）

AGX Orin / JetPack 5.1.1 / TensorRT 8.5.2.2

## 用法

```bash
cd /ssd/<自己的目录>/rfdetr-cp100-jetson-trt

# 文件夹
python3 infer_vis.py --model rf --source /path/to/images --out ./out_rf

# 单张图
python3 infer_vis.py --model rf --source /path/to/one.jpg --out ./out_one
```

`--conf` 默认 **0.25**

## 可视化

```text
out_rf/
  vis/*.jpg            # 原图上绿框
  predictions.json     # 原图像素 xyxy
  run.json             # 张数、框总数、n_nonfinite
```
