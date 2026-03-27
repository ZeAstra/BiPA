# 🌊 BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation
<p align="center">
    <a href="#-paper-acceptance">Paper Acceptance</a> •
    <a href="#-key-features">Key Features</a> •
    <a href="#-setup-guide">Setup Guide</a> •
    <a href="#-dataset-access">Dataset Access</a> •
    <a href="#-pretrained-models">Pretrained Models</a> •
    <a href="#-training-pipeline">Training Pipeline</a> •
    <a href="#-citation">Citation</a>
</p>



## 🎉 Paper Acceptance

 **Great News!** (March 2026) We are thrilled to announce that our paper **BiPA: Bilevel Prompt Adaptation for Underwater Instance Segmentation** has been accepted for publication at [**CVPR 2026**](https://cvpr.thecvf.com/)!



## ✨ Key Features

- **BiPA Framework**: We introduce **BiPA: Bilevel Prompt Adaptation**,, a bilevel optimization method specifically designed for underwater instance segmentation tasks.
- **Progressive Training Strategy**: BiPA employs a two-stage training methodology:
  - **Stage 1**: Initial weight training using standard optimization techniques
  - **Stage 2**: Advanced fine-tuning with Bayesian optimization for optimal model weights and dense embeddings.
- **Comprehensive Dataset Support**: We provide implementations and pre-trained models for two prominent underwater datasets.


<p align="center">
    <img src="figs/flow.png" width="800">
</p>


## 🛠️ Setup Guide

### System Requirements

- Python 3.7 or higher
- PyTorch 2.0+ (we use PyTorch 2.1.0 in our experiments)
- CUDA 11.8 or compatible version
- mmengine library
- mmcv version 2.0.0 or higher
- transformers library version <= 4.50.3
- [MMDetection](https://mmdetection.readthedocs.io/en/latest/get_started.html) 3.0+ framework

### Step-by-Step Installation

```bash
# Step 1: Create and activate a new conda environment
conda create -n bipa python=3.10 -y
conda activate bipa

# Step 2: Install PyTorch
# For detailed PyTorch installation, please visit: https://pytorch.org/get-started/previous-versions/#v212

# Step 3: Install MMEngine, MMCV, and MMDetection using MIM
pip install -U openmim
mim install mmengine
mim install "mmcv>=2.0.0"
mim install mmdet

# Step 4: Install additional dependencies
pip install -r requirements.txt

# Step 5: Download pre-trained weights (SAM and ResNet50)
cd pretrain
bash download_huggingface.sh facebook/sam-vit-huge sam-vit-huge
cd ..
```


## 📦 Dataset Access
You can obtain both datasets from their official repositories:
| Dataset | Official Repository |
|:---:|:---:|
| **UIIS** | [WaterMask Repository](https://github.com/LiamLian0727/WaterMask) |
| **USIS10K** | [USIS10K Repository](https://github.com/LiamLian0727/USIS10K) |

### Data Organization

After downloading, please organize your datasets in the following structure:

```
d:\BiPA/
├── data/
│   ├── UIIS/
│   │   ├── annotations/
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── USIS10K/
│       ├── foreground_annotations/
│       ├── multi_class_annotations/
│       ├── train/
│       ├── val/
│       └── test/
└── ...
```


## 🎯 Pretrained Models

### Results on UIIS Dataset

| Model | Dataset | mAP | AP50 | AP75 | Download |
|:---:|:---:|:---:|:---:|:---:|:---:|
| BiPA | UIIS | 32.1 | 48.7 | 35.2 | [Google]() |
| BiPA | USIS10K-MultiClass | 45.2 | 60.5 | 52.5 | [Google]() |
| BiPA | USIS10K-ClassAgnostic | 64.2 |85.1 | 74.0 | [Google]() |


## 🚀 Training Pipeline

### Two-Stage Training Approach

BiPA implements a two-stage training workflow with Bayesian optimization integration:

#### Stage 1: Initial Weight Training

Begin with standard training to establish initial model weights:

```bash
# Single GPU training
python tools/train.py project/our/configs/stage1_train_.py
```

#### Stage 2: Bayesian Optimization Fine-tuning

Load the weights from Stage 1 and proceed with fine-tuning using Bayesian optimization:

```bash
# Single GPU training
python tools/train.py project/our/configs/stage2_finetune.py
```

### Model Evaluation

To evaluate your trained models, use the Stage 1 configuration:

```bash
python tools/test.py project/our/configs/stage1_train_.py work_dirs/stage2_finetune/epoch_*.pth
```
> **Important Note**: UIIS and USIS10K datasets require different configuration parameters. Please refer to the comments in the configuration files to switch between different dataset configurations.
> **Pro Tip**: For additional training and evaluation options, please refer to the comprehensive [MMDetection User Guides](https://mmdetection.readthedocs.io/en/latest/user_guides/index.html#useful-tools).
---

## 🙏 Acknowledgement

This project builds upon several excellent open-source works:
- [MMDetection](https://github.com/open-mmlab/mmdetection)
- [Segment Anything Model](https://huggingface.co/facebook/sam-vit-huge)
- [RSPrompter](https://github.com/KyanChen/RSPrompter/tree/lightning)
- [USIS-SAM](https://github.com/LiamLian0727/USIS10K)
- [WaterMask](https://github.com/LiamLian0727/WaterMask)
- [Optuna](https://github.com/optuna/optuna)

We are grateful to all contributors for their outstanding work and contributions to the community!
