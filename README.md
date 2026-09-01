# Jetson TensorRT FP32 行人推理（CP100 · RF-DETR-small）

clone 下来即可在 **AGX Orin / JetPack 5.1.1 / TensorRT 8.5.2.2** 上跑。文件夹或单图进，绿框画在图上，json 出。

本仓库是给测试同事的**最小可跑包**（脚本 + 已编好的 Gather FP32 engine），不是训练仓。不要 `pip install torch`。用板上 **Python 3.8** + 系统 TensorRT 8.5.2 + JetPack 自带 torch（只当 CUDA buffer）。不要动系统 `site-packages`。

engine 约 111MB，走 **Git LFS**。clone 前先：

```bash
git lfs install
git clone https://github.com/qidingdig/rfdetr-cp100-jetson-trt.git
# 必须放到 /ssd 下（根分区已满；脚本未设 TMPDIR 时用本目录 tmp/）
cd /ssd/<自己的目录>/rfdetr-cp100-jetson-trt
```

若 `engines/*.engine` 只有几十字节的指针文件，说明 LFS 没拉下来，执行 `git lfs pull`。

## 在哪台板上跑

engine 在共享板 **miivii-tegra**（AGX Orin 32GB，JetPack 5.1.1，TRT 8.5.2.2）上编过，并做过 test_id 可用性检查（md5 `47c9e419f3632fb72bd33836cf552bf0`）。

| 对方环境 | 怎么用 |
|---|---|
| 同一块共享板 | clone 到 `/ssd/<自己的目录>/`。只用仓内 `engines/`。 |
| 另一块 Jetson | 必须同样是 **AGX Orin + JetPack 5.1.1 + TRT 8.5.2.2**。换板 / 换 JetPack / 换 TRT 都不能直接用这份 `.engine`。 |

不要把 A800 的 `.trt` / `.engine` 拷到板子。

## 已验证的 FP32 engine（不要换）

校验和见 `engines/MD5SUMS.txt`。

| 模型 | 文件 | 输入 | 后处理 |
|---|---|---|---|
| RF-DETR-small CP100 | `engines/rfdetr-small-gather_fp32.engine` | 512² 拉伸 | 无 NMS，只做分数阈值 |

RF **必须**用 Gather 改图后编的这份。未 Gather 的 ONNX 在 TRT 8.5.2 上会出 NaN/垃圾框。不要用 FP16。本仓只有 RF，不要传 `--model yolo`。

## 用法

```bash
python3 infer_vis.py --model rf --source /path/to/images --out ./out_rf
python3 infer_vis.py --model rf --source /path/to/one.jpg --out ./out_one
```

`--conf` 默认 **0.25**（不要改成 0.4 当交付档）。`--engine` 可改路径。

## 可视化怎么看

```text
out_rf/
  vis/*.jpg            # 原图上画绿框
  predictions.json     # 原图像素 xyxy
  run.json             # 张数、框总数、n_nonfinite
```

- 绿色矩形：行人框，已映射回原图像素。
- 框上方：`person 0.81`。
- 没有框 = 当前 `--conf` 下无人，正常。
- RF 无 NMS，近处同一人可能叠多个框。
- 若满屏乱框，或 `run.json` 里 `n_nonfinite` 不为 0：用错了 engine。

## 不要

- 不要 `pip install torch`
- 不要从 A800 拷 `.engine` / `.trt`
- 不要用未 Gather 的 ONNX 重编
- 不要覆盖板上已有的旧 +2000 engine
