# 钱币定位测试数据

| 来源 | 本地目录 | 图片 | 圆形真值 | 授权信息 |
| --- | --- | ---: | --- | --- |
| [Eurocoin Vision Dataset](https://huggingface.co/datasets/3v3r51nc3/eurocoin-vision-dataset/tree/b13452e31062e0bf81320e22fc07ce7bb3eba047) | `real/eurocoin-vision/data/train/` | 96 | 无。原始 `metadata.jsonl` 有 911 个矩形框 | 上游数据卡标为 MIT |
| [EuroCoinRecognition](https://github.com/Fraf48/EuroCoinRecognition/tree/ae410d8fd95dcc7ad1a488f5b04f6326e56a2f26) | `real/eurocoin-recognition/` | 28 | 有。68 个上游圆形标注 | 上游 README 说明项目用于教育，未找到明确的数据许可证。本地保留用于评估 |
| [scikit-image `data.coins()`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coins) | `coins.png` | 1 | 无 | scikit-image 文档注明无已知版权限制，原图来自布鲁克林博物馆 |
| [Coin Detection Challenge](https://github.com/carlos-collado/coin-detection-challenge#quick-start) | — | 0 | — | README 说明比赛 `train/` 与 `test/` 图片未随仓库公开。仓库里的两张校准参考照片未找到明确的数据许可，未纳入基准 |
| 程序合成示例 | `synthetic.png`、`synthetic_noisy.png` | 2 | 有。10 个精确圆 | 程序生成，无第三方图像素材 |

EuroCoinRecognition 保留原始 JPEG 及原始 TXT，均按上游 Git blob 校验。`dataset/` 包含 24 张单币照片，`test/` 包含 4 张多币照片。标注为原图像素坐标下的 `EUR_面值 (x, y) r`，多币标注文件末尾可能附有 `SUM`。基准只读取坐标和半径。仓库的检测结果图、视频及其自动检测快照不在本地测试集内。

两张合成示例由 `demo_data.py` 绘制，圆心和半径见 `synthetic_truth.json`。

## 复现基准

在项目根目录运行合成示例基准。

```powershell
uv run --locked python benchmark_datasets.py --synthetic-only --output .\outputs\synthetic-benchmark
```

全量基准需要按上表准备 `data/real/` 下的实拍图和原始标注。在准备齐全后运行 `uv run --locked python benchmark_datasets.py`。

批量测试复用 `coin_detection.py` 和默认 `config.toml`。长边超过 640 像素的图在内存中等比缩小。`read_resize_ms` 记录一次读取和缩放的耗时，`detect_ms` 是 OpenCV 单线程预热后运行三次检测的耗时中位数。两者均以毫秒计，不含结果写盘和绘图。

圆心欧氏距离与半径绝对误差均在检测分辨率的 5 像素以内，才进入现有的一对一匹配。误差均值只统计匹配圆。未匹配的检测和标注分别计为误检和漏检。实拍误检按上游圆标注口径统计。Eurocoin Vision 的原始矩形框只统计数量，scikit-image 只记录检出数和耗时。

默认输出目录为 `outputs/dataset-benchmark/`，可用 `--output` 指定其他位置。`per_image.csv` 汇总逐图计数、时间和误差，`results.json` 额外记录圆坐标、匹配详情及换算回原图坐标的误差。
