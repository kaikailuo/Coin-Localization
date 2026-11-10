# 钱币定位系统 · OpenCV 基础版

输入一张图片或一个目录，输出钱币的圆心、半径以及六个阶段的可视化。程序按输入图片的像素尺寸处理，编号按圆心纵坐标排序，纵坐标相同时从左到右排序。坐标原点在左上角，`x` 向右，`y` 向下，`x`、`y` 和 `r` 的单位均为像素。JSON 中的坐标和半径保留到小数点后两位，绘图时将其取整。

## 安装和运行

在 Windows PowerShell 中进入项目目录。当前交付使用已验证的 Python 3.13 和 uv，依赖安装在项目的 `.venv`。

```powershell
Set-Location 'D:\Projects\Coin Localization'
uv sync --locked --python 3.13

# 生成并检测三个示例
uv run --locked python coin_detection.py

# 单张图片
uv run --locked python coin_detection.py '.\data\coins.png' --output '.\outputs\single'

# 批量检测目录中的 JPG、JPEG 和 PNG（不递归）
uv run --locked python coin_detection.py 'D:\我的图片' --output '.\outputs\batch'
```

读写图片采用字节缓冲区，支持中文路径。独立阶段图保存为原尺寸 PNG。原图阶段展示解码后的 8 位彩色图，读取 JPEG 时按文件的像素排列保留坐标，不应用 EXIF 旋转。未找到圆时，该图片的 `count` 为 0，`circles` 为空，程序仍保存六阶段结果并提示。目录中有损坏的图片时，程序会继续处理其余图片，将失败项记入 `errors`，最后返回非零退出码。

## 目录与输出

```text
.
├── coin_detection.py           # 命令行、图像处理、可视化及 JSON 输出
├── demo_data.py                # 生成示例、保存真值及计算误差
├── config.toml                 # 三个示例共用的默认参数
├── pyproject.toml
├── uv.lock
├── data/
│   ├── coins.png               # scikit-image 内置真实钱币图
│   ├── synthetic.png           # 五个程序绘制的实心圆
│   ├── synthetic_noisy.png     # 对同一合成图加入高斯噪声
│   └── synthetic_truth.json
└── outputs/
    ├── detections.json          # 参数、数量、圆编号、x/y/r 和合成图评估
    ├── coins.png/
    ├── synthetic.png/
    └── synthetic_noisy.png/
```

每个图片子目录都包含 `01_original.png`、`02_grayscale.png`、`03_gaussian.png`、`04_canny.png`、`05_hough.png`、`06_localization.png` 以及 `overview.png`。第 5 张以黄色标出霍夫圆，第 6 张以绿色画圆周，橙红色标圆心，并给出编号和半径。同一次运行的 `detections.json` 记录实际使用的配置。用不同的 `--output` 目录可并排保留多次实验。

## 从灰度图到圆参数

1. `cv2.cvtColor` 将彩色图转换为单通道灰度图。`cv2.GaussianBlur` 抑制细小的亮度波动，同时会让边缘变得更宽。
2. 独立的 `cv2.Canny` 在平滑灰度图上计算梯度，用非极大值抑制将边缘细化，再以高低双阈值保留强边缘及与强边缘相连的弱边缘，结果保存为第 4 张图。
3. `cv2.HoughCircles` 的 `HOUGH_GRADIENT` 同样读取平滑灰度图。它内部提取边缘，利用边缘梯度方向为可能的圆心投票，再为候选圆心估计半径。OpenCV 用 `minDist` 抑制过近的候选圆心，本示例直接保留 OpenCV 返回的圆。
4. `cv2.circle` 将检测圆叠加到原图，最终图增加圆心、编号和半径。总览图按从左到右、从上到下的顺序排列六个阶段。

接口参考见 [OpenCV 霍夫圆教程](https://docs.opencv.org/4.x/d4/d70/tutorial_hough_circle.html) 和 [OpenCV HoughCircles 参数文档](https://docs.opencv.org/4.x/dd/d1a/group__imgproc__feature.html)。

## 参数

`config.toml` 是默认配置。可复制为 `experiment.toml`，调整后通过 `--config` 指定。

```powershell
Copy-Item .\config.toml .\experiment.toml
# 编辑 experiment.toml 后运行
uv run --locked python coin_detection.py --config .\experiment.toml --output .\outputs\experiment
```

| 配置组 | 参数 | 默认 | 含义 |
| --- | --- | ---: | --- |
| `gaussian` | `kernel_size` | 5 | 高斯核边长，必须为正奇数。值越大，滤波涉及的邻域越大。 |
| `gaussian` | `sigma` | 1.2 | 高斯标准差，控制权重扩散程度。设为 0 时由 OpenCV 根据核大小推算。 |
| `canny` | `threshold1` | 50 | 独立 Canny 的低阈值，参与弱边缘连接判断。只影响第 4 张图及总览中的对应面板。 |
| `canny` | `threshold2` | 120 | 独立 Canny 的高阈值，决定强边缘。只影响第 4 张图及总览中的对应面板。 |
| `hough` | `dp` | 1.2 | 输入图尺寸与霍夫累加器尺寸的比值。1 表示同分辨率，值越大，累加器采样越粗。 |
| `hough` | `minDist` | 32 | 候选圆心之间的最小距离，单位为像素。过小容易重复，过大容易漏掉相邻圆。 |
| `hough` | `param1` | 110 | `HOUGH_GRADIENT` 内部 Canny 的高阈值，内部低阈值约为其一半。 |
| `hough` | `param2` | 27 | `HOUGH_GRADIENT` 接受圆心候选的累加器阈值。调低通常会得到更多候选。 |
| `hough` | `minRadius` | 12 | 搜索的最小半径，单位为像素。 |
| `hough` | `maxRadius` | 37 | 搜索的最大半径，单位为像素。设为 0 时由 OpenCV 根据图像尺寸确定。 |

可以先单独调整 `canny.threshold2`，观察第 4 张边缘图变化，再单独调整 `hough.param2`，观察第 5 张图里候选圆的变化。高斯参数则同时影响两条处理路径。

## 实测结果

在 Windows 上使用 Python 3.13.18、OpenCV 4.14.0、NumPy 2.5.4、Matplotlib 3.11.2、scikit-image 0.26.0 运行，三张示例图使用同一份 `config.toml`。

| 示例 | 检出 | 匹配对象 | 误检 / 漏检 | 圆心误差均值 / 最大值（px） | 半径误差均值 / 最大值（px） |
| --- | ---: | ---: | ---: | ---: | ---: |
| [真实钱币](outputs/coins.png/overview.png) | 24 | 24（逐枚目视） | 0 / 0 | — | — |
| [理想合成圆](outputs/synthetic.png/overview.png) | 5 | 5 | 0 / 0 | 1.077 / 2.088 | 0.552 / 2.080 |
| [含噪合成圆](outputs/synthetic_noisy.png/overview.png) | 5 | 5 | 0 / 0 | 0.661 / 1.000 | 0.384 / 0.520 |

- 真实钱币图来自 [scikit-image `data.coins()`](https://scikit-image.org/docs/stable/api/skimage.data.html#skimage.data.coins)。目视将圆与可见钱币逐枚对应。第 15 号 `(279.0, 195.0, 16.56)` 在第三排偏右，圆周主要落在币面内部。第 23 号 `(108.6, 265.8, 16.92)` 在最下排靠左，圆偏小且偏左。真实图没有像素级圆参数真值。
- 合成评估要求圆心欧氏距离和半径绝对误差均不超过 5 px，并做一对一匹配。每个真值对应的检测编号和误差可在 [detections.json](outputs/detections.json) 中查看。含噪示例使用固定随机种子 2026，叠加标准差为 20 灰度级的零均值高斯噪声后裁剪到 8 位范围。两组误差只代表本次样本。

## 局限与下一步

当前默认半径范围针对这三张示例的像素尺度。钱币较大或较小时需调整半径范围和最小圆心间距。倾斜拍摄会使钱币投影偏椭圆，遮挡、强反光和复杂纹理也可能使外缘投票不足，或使内部纹理得到更强的投票。下一步可先用不同分辨率、光照和遮挡程度的实拍图验证，再用候选圆周附近的边缘支持度筛选假圆，或沿外缘重新拟合中心和半径。
