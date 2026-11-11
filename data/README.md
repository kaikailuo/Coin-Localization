# 钱币定位测试数据

| 来源 | 本地目录 | 图片 | 圆形真值 | 授权信息 |
| --- | --- | ---: | --- | --- |
| [Eurocoin Vision Dataset](https://huggingface.co/datasets/3v3r51nc3/eurocoin-vision-dataset/tree/b13452e31062e0bf81320e22fc07ce7bb3eba047) | `real/eurocoin-vision/data/train/` | 96 | 无。原始 `metadata.jsonl` 有 911 个矩形框 | 上游数据卡标为 MIT |
| [EuroCoinRecognition](https://github.com/Fraf48/EuroCoinRecognition/tree/ae410d8fd95dcc7ad1a488f5b04f6326e56a2f26) | `real/eurocoin-recognition/` | 28 | 有。68 个上游圆形标注 | 上游 README 说明项目用于教育，未找到明确的数据许可证。本地保留用于评估 |
| [scikit-image `data.coins()`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coins) | `coins.png` | 1 | 无 | scikit-image 文档注明无已知版权限制，原图来自布鲁克林博物馆 |
| [Coin Detection Challenge](https://github.com/carlos-collado/coin-detection-challenge#quick-start) | — | 0 | — | README 说明比赛 `train/` 与 `test/` 图片未随仓库公开。仓库里的两张校准参考照片未找到明确的数据许可，未纳入基准 |
| [Brazilian Coin Detection Dataset](https://ieee-dataport.org/open-access/brazilian-coin-detection-dataset) | `real/brazilian-coins/` | 36 | 有。243 个上游圆形标注 | [Kaggle v2](https://www.kaggle.com/datasets/lgmoneda/br-coins) 标为 CC BY-NC-SA 4.0 |
| Wikimedia Commons | `real/commons-coins/` | 3 | 无 | 逐张许可见下表 |
| 程序合成示例 | `synthetic.png`、`synthetic_noisy.png` | 2 | 有。10 个精确圆 | 程序生成，无第三方图像素材 |

EuroCoinRecognition 保留原始 JPEG 及原始 TXT，均按上游 Git blob 校验。`dataset/` 包含 24 张单币照片，`test/` 包含 4 张多币照片。标注为原图像素坐标下的 `EUR_面值 (x, y) r`，多币标注文件末尾可能附有 `SUM`。基准只读取坐标和半径。仓库的检测结果图、视频及其自动检测快照不在本地测试集内。

两张合成示例由 `demo_data.py` 绘制，圆心和半径见 `synthetic_truth.json`。

## 泛化评估子集

`generalization.json` 固定评估图片、场景和参考标注。Eurocoin Vision 的子集复用上表中的原图。巴西数据和 Wikimedia Commons 原图放在 Git 忽略的 `data/real/` 下。

- Eurocoin Vision 选了 20 张，涵盖普通背景、纹理背景、光照变化和密集或相邻排列，每类 5 张。19 张共 132 个近似圆，用原有人工矩形框限定范围，沿框中心发出的射线在框边附近找灰度变化，再对轮廓点做剔除离群点的最小二乘圆拟合。检查了圆叠加图。遮挡图 `gp2__3.jpeg` 用于目视分析。将 `box-guided-fit` 的误差作为近似评估，单独报告。
- 巴西子集取自 Kaggle v2 的 `COCO_labelme_regression/regression/`，排除了同时出现在单币子集中的文件。按每图币数和标注半径筛选，保留每张所有目标均有圆形标注的 36 张图。2 枚、3 枚、4–7 枚、8 枚及以上分别有 4、4、14、14 张。圆心取 LabelMe 第一个点，半径取两个点的欧氏距离。图像和原始 LabelMe JSON 的归档路径、SHA-256 均列于清单，JSON 保存在本地 `labels/` 子目录。
- Wikimedia Commons 选了 3 张用于目视检查，覆盖钱币册及杂物、重叠币堆和侧视叠币。原图的本地文件名和署名如下。

| 文件（相对于 `data/real/commons-coins/`） | 作者 | 许可与来源 |
| --- | --- | --- |
| `coin-collection.jpg` | Anemonemma、Generalissima、3df | [CC BY-SA 4.0](https://commons.wikimedia.org/wiki/File:Coin_collection.jpg) |
| `ukrainian-coin-pile.jpg` | Роман Рябенко | [CC0 1.0](https://commons.wikimedia.org/wiki/File:A_pile_of_Ukrainian_currency_coins.jpg) |
| `wooden-blocks-and-coins.jpg` | Jernej Furman | [CC BY 2.0](https://commons.wikimedia.org/wiki/File:Wooden_blocks_with_text_LOSS_and_coins_(53156710314).jpg) |

复现时，从 Kaggle v2 ZIP 按清单里的 `archive_image`、`archive_label` 提取指定成员。图片保存到 `image` 指定的位置（相对于 `data/`），标注保存到图片目录下的 `labels/`。从上表下载 Wikimedia Commons 原图并使用列出的本地文件名。

调参时留出巴西子集及清单中的 Eurocoin Vision `capture_group` 整组（文件名前缀）。固定配置后运行这个子集。输出按来源和场景汇总，并记录参考类型及实际参与圆匹配的图片数。

```powershell
uv run --locked python benchmark_datasets.py --generalization-only --output .\outputs\generalization-benchmark
```

## 复现基准

在项目根目录运行合成示例基准。

```powershell
uv run --locked python benchmark_datasets.py --synthetic-only --output .\outputs\synthetic-benchmark
```

原有基准需要准备 Eurocoin Vision 和 EuroCoinRecognition 的实拍图和原始标注，运行 `uv run --locked python benchmark_datasets.py`。泛化子集使用上面的 `--generalization-only` 入口。

批量测试复用 `coin_detection.py` 和默认 `config.toml`。长边超过 640 像素的图在内存中等比缩小。`read_resize_ms` 记录一次读取和缩放的耗时，`detect_ms` 是 OpenCV 单线程预热后运行三次检测的耗时中位数。两者均以毫秒计，不含结果写盘和绘图。

圆心欧氏距离与半径绝对误差均在检测分辨率的 5 像素以内，才进入现有的一对一匹配。误差均值只统计匹配圆。未匹配的检测和标注分别计为误检和漏检。实拍误检按上游圆标注口径统计。Eurocoin Vision 的原始矩形框只统计数量，scikit-image 只记录检出数和耗时。

默认输出目录为 `outputs/dataset-benchmark/`，可用 `--output` 指定其他位置。`per_image.csv` 汇总逐图计数、时间和误差，`results.json` 额外记录圆坐标、匹配详情及换算回原图坐标的误差。
