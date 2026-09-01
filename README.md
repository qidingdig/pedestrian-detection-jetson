# Jetson TensorRT FP32 行人推理（CP100 · RF-DETR-small）

AGX Orin / JetPack 5.1.1 / TensorRT 8.5.2.2

`.engine` 走 Git LFS，**clone 不会自动安装**。没装的话文件只有一百多字节，推理会失败。

```bash
sudo apt install git-lfs
git lfs install
git clone -b cp100 https://github.com/qidingdig/pedestrian-detection-jetson.git
cd /ssd/<自己的目录>/pedestrian-detection-jetson
ls -lh engines/rfdetr-small-gather_fp32.engine   # 应约 111M；若只有一百多字节则 git lfs pull
```

## 用法

```bash
cd /ssd/<自己的目录>/pedestrian-detection-jetson

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
