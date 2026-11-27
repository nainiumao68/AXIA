# AXIA

**AXIA：面向 RGB-X 遥感语义分割的非对称跨模态注入架构与参数高效适配器**的官方 PyTorch 实现。

Official PyTorch implementation of **AXIA: An Asymmetric X-modal Injection Architecture with Parameter-Efficient Adapters for RGB-X Remote Sensing Semantic Segmentation**.

## 简介 / Introduction

RGB-X 遥感语义分割利用 SAR、DSM 或热红外数据补充光学信息。AXIA 以 RGB 为主分支、X 为辅助分支，通过轻量适配器将互补的物理信息注入 RGB 特征。

- **非对称双编码器**：冻结 ImageNet-22K 预训练的 Swin-Large RGB 主干，使用紧凑的 Swin-Tiny X 主干，仅训练 X 主干前两个阶段。可训练参数量取决于解码器宽度和类别数，PIE 默认配置约为 19.0M。
- **块级跨模态注入**：在 24 个冻结的 RGB Transformer 块中插入 DGA 和 SSA 适配器。DGA 在注意力后根据跨模态余弦差异控制空间信息注入；SSA 在 FFN 后通过共享通道子空间进行门控反对称信息交换。
- **场景感知重组解码器 SARD**：SGCR 根据场景信息进行低秩通道组混合，CMKP 通过多感受野深度卷积将深层语义逐级传播到浅层。

RGB-X remote sensing segmentation complements optical imagery with SAR, DSM, or thermal infrared data. AXIA uses RGB as the primary stream and X as the auxiliary stream, injecting complementary physical cues through lightweight adapters.

- **Asymmetric dual encoder**: a frozen ImageNet-22K pretrained Swin-Large RGB backbone is paired with a compact Swin-Tiny X backbone, with only its first two stages trainable. The trainable parameter count depends on decoder width and class count; the default PIE configuration has approximately 19.0M trainable parameters.
- **Block-level X-modal injection**: DGA and SSA adapters are inserted into the 24 frozen RGB Transformer blocks. After attention, DGA gates spatial injection using cross-modal cosine discrepancy; after the FFN, SSA performs gated anti-symmetric exchange within shared channel subspaces.
- **Scene-Aware Reorganization Decoder (SARD)**: Scene-Guided Channel Reassembly (SGCR) performs scene-conditioned low-rank channel-group mixing, while Cascaded Multi-Kernel Perception (CMKP) propagates deep semantics toward shallow stages through depthwise convolutions with multiple receptive fields.

## 数据准备 / Data preparation

各数据集按以下方式组织，样本列表每行填写一个不含扩展名的样本名称。

```text
datasets/<DATASET>/
├── RGB/                              # RGB 影像
├── SAR/ (T/, DSM/, ...)              # 辅助模态
├── Label/                            # 标签，具体目录见下表
├── train.txt                         # 训练样本
└── val.txt                           # 验证样本
```

修改对应配置文件中的 `C.dataset_path`，使其指向本地数据集。各数据集的标签目录和验证列表如下。

| 配置 | 数据集 | 模态 | 输入尺寸 | 标签目录 | 验证列表 |
| --- | --- | --- | --- | --- | --- |
| `config_pie.py` | PIE-RGB-SAR | RGB + SAR | 256 × 256 | `Label` | `val.txt` |
| `config_yeseg.py` | YESeg-OPT-SAR | RGB + SAR | 256 × 256 | `Label_normalized` | `test.txt` |
| `config_pot.py` | ISPRS Potsdam | RGB + DSM | 300 × 300 | `Label` | `test.txt` |
| `config_cart.py` | CART | RGB + 热红外 | 600 × 960 | `Labels` | `test.txt` |

Potsdam 使用 6 类标签，mIoU 对前 5 类求平均，不包含 clutter 类（`C.num_eval_classes = 5`）；CART 的原始标签通过 `C.label_mapping` 映射为训练类别。

Organize each dataset as follows, with one sample name without its extension per line in each split file.

```text
datasets/<DATASET>/
├── RGB/                              # RGB images
├── SAR/ (T/, DSM/, ...)              # auxiliary modality
├── Label/                            # labels; see table below
├── train.txt                         # training samples
└── val.txt                           # validation samples
```

Set `C.dataset_path` in the corresponding config to your local dataset. Label folders and validation split files are listed below.

| Config | Dataset | Modality | Input size | Label folder | Validation split |
| --- | --- | --- | --- | --- | --- |
| `config_pie.py` | PIE-RGB-SAR | RGB + SAR | 256 × 256 | `Label` | `val.txt` |
| `config_yeseg.py` | YESeg-OPT-SAR | RGB + SAR | 256 × 256 | `Label_normalized` | `test.txt` |
| `config_pot.py` | ISPRS Potsdam | RGB + DSM | 300 × 300 | `Label` | `test.txt` |
| `config_cart.py` | CART | RGB + Thermal | 600 × 960 | `Labels` | `test.txt` |

Potsdam uses six label classes and averages mIoU over the first five, excluding clutter (`C.num_eval_classes = 5`). CART maps raw labels to training classes through `C.label_mapping`.

## 预训练权重 / Pretrained weights

将 Swin 预训练权重放入 `pretrained/`，或直接修改配置中的权重路径。

- RGB 分支: `swin_large_patch4_window7_224_22k.pth`
- X 分支: `swin_tiny_patch4_window7_224.pth`

```python
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
```

Place the pretrained Swin checkpoints under `pretrained/`, or edit their paths in the config.

- RGB branch: `swin_large_patch4_window7_224_22k.pth`
- X branch: `swin_tiny_patch4_window7_224.pth`

```python
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
```

## 训练 / Training

使用配置文件选择数据集并启动训练。

```bash
python train.py --cfg config_pie.py
```

可通过以下参数选择 GPU 或从检查点恢复训练。

- 选择 GPU: `--gpu-id 0` 或 `--gpu-devices 0`
- 恢复训练: `-c path/to/checkpoint.pth`

### 训练流程

1. **模型构建**：加载 RGB/X 预训练权重，冻结 RGB 主干和 X 主干的第 3、4 阶段，训练适配器、融合模块、解码器及辅助头。

2. **数据增强**：随机水平翻转、按 `C.train_scale_array` 随机缩放，随后进行归一化和随机裁剪。

3. **优化**：默认使用 AdamW，对 X 主干、适配器和解码器设置分组学习率；线性预热后使用多项式衰减，并按 `C.grad_clip` 裁剪梯度。损失为像素级交叉熵与辅助监督损失之和。

4. **验证与保存**：四个数据集均设置 `C.val_frequency = 5`，从第 5 轮开始，在第 5、10、15……轮验证；全程保持固定间隔。最后一轮也遵循该间隔。验证 mIoU 最佳的模型保存到 `log_<DATASET>/checkpoint/best_model.pth`。

例如，PIE-RGB-SAR 默认训练 300 轮，批量大小为 8，主干、适配器和解码器学习率分别为 6e-5、3e-4 和 1e-4，解码器通道数为 328，dropout 为 0.05。

当前训练实现使用 AdamW 和多项式衰减。训练验证每五轮只计算指标并按最佳 mIoU 保存检查点。

Select a dataset through its config file and start training.

```bash
python train.py --cfg config_pie.py
```

Use the following flags to select a GPU or resume from a checkpoint.

- Select GPU: `--gpu-id 0` or `--gpu-devices 0`
- Resume training: `-c path/to/checkpoint.pth`

### Training pipeline

1. **Model construction**: load RGB/X pretrained weights, freeze the RGB backbone and X stages 3–4, and train adapters, fusion modules, the decoder, and the auxiliary head.

2. **Data augmentation**: apply random horizontal flipping and scaling according to `C.train_scale_array`, followed by normalization and random cropping.

3. **Optimization**: use AdamW by default with grouped learning rates for the X backbone, adapters, and decoder. Linear warmup is followed by polynomial decay, with gradient clipping controlled by `C.grad_clip`. The loss combines pixel-wise cross-entropy and auxiliary supervision.

4. **Validation and checkpointing**: all four datasets set `C.val_frequency = 5` and validate at epochs 5, 10, 15, and so on throughout training. The final epoch follows the same interval. The model with the best validation mIoU is saved to `log_<DATASET>/checkpoint/best_model.pth`.

For PIE-RGB-SAR, the defaults are 300 epochs, batch size 8, backbone/adapter/decoder learning rates of 6e-5/3e-4/1e-4, and a 328-channel decoder with dropout 0.05.

The current training implementation uses AdamW with polynomial decay. Training validation computes metrics every five epochs and saves the best-mIoU checkpoint without generating visualizations.

## 评估 / Evaluation

通过配置文件和模型权重执行独立评估。

```bash
python test.py --config config_pie.py --weights log_PIE-RGB-SAR/checkpoint/best_model.pth
```

默认使用单尺度整图推理。常用参数如下。

- 批量大小与加载进程数: `--batch-size 4 --workers 4`
- 保存 RGB、X、真值和预测四联图: `--vis`
- 仅保存彩色预测图: `--vis-seg-only`
- 导出 JSON 指标: `--results-json results.json`

Run standalone evaluation with a config file and model checkpoint.

```bash
python test.py --config config_pie.py --weights log_PIE-RGB-SAR/checkpoint/best_model.pth
```

Evaluation uses single-scale whole-image inference by default. Useful flags are listed below.

- Batch size and loader workers: `--batch-size 4 --workers 4`
- Save RGB, X, ground truth, and prediction panels: `--vis`
- Save only colorized predictions: `--vis-seg-only`
- Export metrics as JSON: `--results-json results.json`

## 代码结构 / Repository structure

主要训练、评估与模型组件的职责如下。

```text
AXIA/
├── config_*.py                       # 数据集配置
├── train.py                          # 训练入口
├── test.py                           # 评估入口
├── dataloader/
│   ├── unified_loader.py             # 数据增强与加载
│   └── RGBXDataset.py                # RGB-X 数据集
├── engine/                           # 状态管理与检查点
├── models/
│   ├── builder_AXIA.py               # 模型构建与损失
│   ├── encoders/dual_encoder.py      # 双编码器
│   └── decoders/SARD.py              # 场景感知重组解码器
├── modules/
│   ├── DiscrepancyGuidedAdapter.py   # 差异引导适配器
│   └── SharedSubspaceAdapter.py      # 共享子空间适配器
└── utils/                            # 验证、指标及工具
```

The main training, evaluation, and model components are organized as follows.

```text
AXIA/
├── config_*.py                       # dataset configs
├── train.py                          # training entry
├── test.py                           # evaluation entry
├── dataloader/
│   ├── unified_loader.py             # augmentation and loading
│   └── RGBXDataset.py                # RGB-X dataset
├── engine/                           # state and checkpoints
├── models/
│   ├── builder_AXIA.py               # model assembly and losses
│   ├── encoders/dual_encoder.py      # dual encoder and LSGF
│   └── decoders/SARD.py              # scene-aware decoder
├── modules/
│   ├── DiscrepancyGuidedAdapter.py   # DGA
│   └── SharedSubspaceAdapter.py      # SSA
└── utils/                            # validation, metrics, utilities
```

## 引用 / Citation

如果本项目对您的研究有帮助，请考虑引用以下工作。

```text
AXIA: An Asymmetric X-modal Injection Architecture with Parameter-Efficient
Adapters for RGB-X Remote Sensing Semantic Segmentation.
```

If you find this work useful, please consider citing the following work.

```text
AXIA: An Asymmetric X-modal Injection Architecture with Parameter-Efficient
Adapters for RGB-X Remote Sensing Semantic Segmentation.
```

## 许可证 / License

AXIA 自有代码使用 [Apache License 2.0](LICENSE)。源自 CMX 和 Swin Transformer 的代码保留其原有 MIT 许可证，详见[第三方声明](THIRD_PARTY_NOTICES.md)。

AXIA's original code is licensed under the [Apache License 2.0](LICENSE). Code derived from CMX and Swin Transformer retains its MIT license. See the [third-party notices](THIRD_PARTY_NOTICES.md).

## 致谢 / Acknowledgements

感谢 [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) 和 [CFFormer](https://github.com/masurq/CFFormer) 两个仓库的作者公开代码，为本项目提供了宝贵参考。

We thank the authors of [CMX](https://github.com/huaaaliu/RGBX_Semantic_Segmentation) and [CFFormer](https://github.com/masurq/CFFormer) for sharing their code and providing valuable references for this project.
