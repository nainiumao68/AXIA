# AXIA

Official PyTorch implementation of **AXIA: An Asymmetric X-modal Injection Architecture with Parameter-Efficient Adapters for RGB-X Remote Sensing Semantic Segmentation**.

## Introduction

RGB-X multimodal remote sensing segmentation improves scene parsing with auxiliary SAR, DSM, or thermal infrared data. Existing methods usually adopt symmetric dual backbones with stage-wise fusion, implicitly assuming comparable RGB/X information capacity and task adaptability. Based on the observed role asymmetry between modalities, AXIA instead treats RGB as the primary stream and X as an auxiliary stream that injects complementary physical cues (backscatter, elevation, thermal radiation):

- **Asymmetric dual encoder**: a frozen ImageNet-22K pretrained **Swin-Large** RGB backbone preserves transferable optical priors, while a compact **Swin-Tiny** X backbone (ImageNet-22K pretrained, only the first two stages trainable) adapts to modality-specific statistics. Only **19.4M** parameters are trainable.
- **Block-level X-modal injection**: instead of interacting only between encoder stages, two lightweight adapters are inserted into every one of the 24 frozen RGB blocks:
  - **DGA (Discrepancy-Guided Adapter)**, after (S)W-MSA: injects spatial X-modal cues into RGB in a gated manner, with injection strength proportional to per-pixel cross-modal cosine discrepancy.
  - **SSA (Shared-Subspace Adapter)**, after the FFN: decomposes both modalities into private/shared channel subspaces and performs gated anti-symmetric exchange only within the shared subspace, aligning semantics without overwriting modality-specific cues.
- **Lightweight stage fusion**: the two streams are merged at each stage boundary by a zero-initialized confidence-gated additive fusion (LSGF, ~0.15M parameters in total).
- **SARD (Scene-Aware Reorganization Decoder)**: designed for remote sensing scenes and the asymmetric fused representation. It consists of **SGCR (Scene-Guided Channel Reassembly)**, which reorganizes the deepest semantics via scene-conditioned low-rank channel-group mixing, and **CMKP (Cascaded Multi-Kernel Perception)**, which progressively propagates deep semantics to shallow stages through parallel depthwise convolutions with different receptive fields.

## Results

AXIA achieves state-of-the-art mIoU on four RGB-X benchmarks with only 19.4M trainable parameters:

| Dataset | Modality | Classes | mIoU (%) | mAcc (%) |
| --- | --- | --- | --- | --- |
| PIE-RGB-SAR (cloud-free) | RGB-SAR | 6 | **80.2** | **87.6** |
| PIE-RGB-SAR (cloudy) | RGB-SAR | 6 | **78.3** | **86.7** |
| YESeg-OPT-SAR | RGB-SAR | 8 | **65.0** | **74.0** |
| ISPRS Potsdam | RGB-DSM | 6 (5 evaluated) | **86.2** | **92.9** |
| CART | RGB-T | 10 | **76.8** | 84.2 |

## Installation

```bash
conda create -n axia python=3.10 -y
conda activate axia
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install timm easydict opencv-python tensorboardX tabulate matplotlib tqdm
```

Reference environment: a single NVIDIA A40 GPU, PyTorch 2.3.1, CUDA 12.1.

## Data preparation

Each dataset should be organized as:

```
datasets/<DATASET>/
├── RGB/                          # RGB images
├── SAR/  (or T/, DSM/, ...)      # auxiliary modality images
├── Label/                        # label maps
├── train.txt                     # one sample name per line
└── test.txt  (or val.txt)
```

Then edit the corresponding config file (`config_pie.py`, `config_yeseg.py`, `config_pot.py`, `config_cart.py`) and point `C.dataset_path` to your local copy.

| Config | Dataset | Modality | Input size |
| --- | --- | --- | --- |
| `config_pie.py` | PIE-RGB-SAR | RGB + SAR | 256 x 256 |
| `config_yeseg.py` | YESeg-OPT-SAR | RGB + SAR | 256 x 256 |
| `config_pot.py` | ISPRS Potsdam | RGB + DSM | 300 x 300 |
| `config_cart.py` | CART | RGB + Thermal | 600 x 960 |

## Pretrained weights

Download the ImageNet-22K pretrained Swin checkpoints and place them under `pretrained/`:

- `swin_large_patch4_window7_224_22k.pth` (RGB branch)
- `swin_tiny_patch4_window7_224.pth` (X branch)

The default config paths are:

```python
C.pretrained_model = './pretrained/swin_large_patch4_window7_224_22k.pth'
C.sar_pretrained_model = './pretrained/swin_tiny_patch4_window7_224.pth'
```

## Training

```bash
python train.py --cfg config_pie.py
```

Useful flags:

- `--gpu-id 0` / `--gpu-devices 0` to select the GPU
- `-c path/to/checkpoint.pth` to resume training from a checkpoint

### Training pipeline

1. **Model construction** (`models/builder_AXIA.py`): the AXIA encoder is built, the ImageNet-22K Swin-Large/Swin-Tiny weights are loaded into the RGB/X branches, the RGB backbone is fully frozen, and X stages 3-4 are frozen (only X stages 1-2, the DGA/SSA adapters, the LSGF fusion, the decoder, and the auxiliary head are trained).
2. **Data augmentation**: random horizontal flipping and random scaling with ratios {0.75, 1.0, 1.25}, followed by normalization and random cropping to the dataset input size.
3. **Optimization**: AdamW (betas=(0.9, 0.999), eps=1e-8, weight decay 0.01) with grouped learning rates - base LR for the trainable X backbone, a higher `adapter_lr` for DGA/SSA, and `decoder_lr` for the decoder and auxiliary head. Linear warm-up for `warm_up_epoch` epochs followed by polynomial decay (power 0.9). Gradient clipping is applied via `grad_clip`. The loss is pixel-wise cross-entropy plus the auxiliary-head loss.
4. **Validation and checkpointing**: validation runs every `val_early_frequency` epochs for the first `val_early_epochs` epochs, then every `val_late_frequency` epochs. The checkpoint with the best validation mIoU is saved to `log_<DATASET>/checkpoint/best_model.pth`.

The default hyperparameters of each dataset config follow the paper settings (e.g. for PIE-RGB-SAR: base LR 6e-5, adapter LR 3e-4, decoder LR 1e-4, 300 epochs, batch size 8, SARD with 328 channels and dropout 0.05).

## Evaluation

```bash
python test.py --config config_pie.py --weights log_PIE-RGB-SAR/checkpoint/best_model.pth
```

Testing uses single-scale whole-image inference by default. Useful flags:

- `--batch-size 4` / `--workers 4` for faster evaluation
- `--vis` to save 4-panel visualizations (RGB / X / GT / Pred)
- `--vis-seg-only` to save only the colorized prediction maps
- `--results-json results.json` to dump metrics as JSON

## Ablations

Switch the decoder via `C.decoder`:

```python
C.decoder = 'SARD'        # default
# C.decoder = 'UPerHead'
# C.decoder = 'MLPDecoder'
# C.decoder = 'FCNHead'
```

Switch the auxiliary head via `C.aux_head`:

```python
C.aux_head = 'FCNHead'    # default
# C.aux_head = 'AFDAH'
# C.aux_head = None       # disable auxiliary supervision
```

## Repository structure

```
AXIA/
├── config_cart.py                # CART (RGB-T)
├── config_pie.py                 # PIE-RGB-SAR
├── config_pot.py                 # ISPRS Potsdam (RGB-DSM)
├── config_yeseg.py               # YESeg-OPT-SAR
├── train.py                      # training entry
├── test.py                       # evaluation entry
├── dataloader/
│   ├── loader.py                 # train/val preprocessing and loaders
│   └── RGBXDataset.py            # generic RGB-X dataset
├── engine/
│   ├── engine.py                 # state registry and checkpointing
│   └── logger.py
├── models/
│   ├── builder_AXIA.py           # model assembly, weight loading, freezing
│   ├── encoders/
│   │   └── axia_encoder.py       # AXIAEncoder (DGA/SSA adapters + LSGF fusion)
│   └── decoders/
│       ├── SARD.py               # SGCR + CMKP decoder
│       ├── AFDAH.py              # frequency-domain auxiliary head
│       ├── fcnhead.py
│       ├── MLPDecoder.py
│       └── UPerNet.py
├── modules/
│   ├── DiscrepancyGuidedAdapter.py   # DGA
│   └── SharedSubspaceAdapter.py      # SSA
└── utils/
    ├── ema.py                    # exponential moving average
    ├── lr_policy.py              # warmup + poly/cosine schedules
    ├── metric.py                 # confusion-matrix metrics
    ├── pyt_utils.py              # config loading, checkpoint IO, misc
    ├── transforms.py             # crop / pad / normalize
    ├── validation.py             # validation during training
    └── visualize.py              # palettes and mask colorization
```

## Citation

If you find this work useful, please consider citing:

```
AXIA: An Asymmetric X-modal Injection Architecture with Parameter-Efficient
Adapters for RGB-X Remote Sensing Semantic Segmentation.
```
