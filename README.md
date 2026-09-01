# 行人检测（Jetson）

AGX Orin / JetPack 5.1.1 / TensorRT 8.5.2.2

按模型分支拉取，后续新模型会另开分支。

| 分支 | 模型 |
|---|---|
| `cp100` | RF-DETR-small CP100 |

```bash
git clone -b cp100 <本仓 URL>
cd /ssd/<自己的目录>/pedestrian-detection-jetson
python3 infer_vis.py --model rf --source /path/to/images --out ./out_rf
```
