# SRPR-Net

This repository contains the implementation of the following paper:

> **SRPR-Net: Semantic and Relational Prompt Refinement for Automated SAM-based Instance Segmentation**

## Overview

<p align="center">
  <img src="pipline.png" alt="Overview of the SRPR-Net framework" width="100%">
</p>

we propose a novel architecture, named Semantic Relational Prompt Refinement Network (SRPR-Net), for automated SAM-based instance segmentation. Specifically, SRPR-Net enriches detector-generated prompts with frozen CLIP features and conducts relational prompt refinement via a Transformer-based interaction module, which models inter-instance dependencies among semantically enhanced prompts to yield refined box prompts for SAM segmentation. Experiments on multiple standard benchmarks demonstrate that SRPR-Net consistently improves segmentation performance over existing approaches and achieves consistent improvements across three heterogeneous instance segmentation benchmarks. 

## Prerequisites
- Linux (We tested our codes on Ubuntu 24.04)
- Anaconda

For a CUDA-capable computer, build the original environment:
```
conda env create -f environment.yml
conda activate blo-inst
```

### CPU-only environment（无 GPU）

训练入口未将 GPU 编号写死，计算设备由 `--device` 参数选择。当前
`train.sh` 已显式设置 `--device cpu`，因此不会自动占用 CUDA 设备。CPU
模式下自动混合精度会被关闭，代码中的 CUDA 缓存清理调用不会使训练切换到
GPU。

由于 YOLO 与 SAM 会在同一训练循环中被优化，CPU 训练速度将明显低于 GPU
训练速度，完整执行 20 个 epoch 可能需要较长时间。Windows 环境建议使用
WSL2/Ubuntu 执行本项目；若使用原生 Windows，应在 PowerShell 中直接执行
下文给出的 Python 单行命令，而不是执行 Bash 脚本。

在项目根目录创建 CPU 专用环境：

```bash
conda env create -f environment-cpu.yml
conda activate blo-inst-cpu
```

`environment-cpu.yml` 将安装 PyTorch 与 torchvision 的 CPU 版本，不会安装
`environment.yml` 中的 NVIDIA CUDA 运行库。训练前可执行以下命令检查环境：

```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

第二行应输出 `False`。

## Training
You can try our code on one of the public datasets we used in our experiments. Here are the instructions: 

1. We provide the [Penn-Fudan Database](./PennFudanPed/)
2. Pretrain the YOLO model on this dataset. We also provide the [checkpoint](./yolo-pretrained/ped.pt) for your quick try.
3. Download the
   [official SAM ViT-B checkpoint](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth)
   and place it at `weights/sam_vit_b_01ec64.pth`. This repository copy
   already contains that file. If it is missing, run:
   ```bash
   mkdir -p weights
   wget -O weights/sam_vit_b_01ec64.pth https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
   ```
4. Run this command from the repository root:
```bash
bash train.sh
```

当前 `train.sh` 已配置为使用项目内的 Penn-Fudan 数据、YOLO 权重和 SAM
权重，可直接执行。CPU 相关设置如下：

```bash
--device cpu
--batch-size 1
--workers 0
```

其中，`--workers 0` 用于减少本地 Windows 数据加载的多进程兼容问题。训练
生成的权重位于 `runs/train-seg/blo-inst*/weights/`。

在原生 Windows PowerShell 中，可执行以下等价命令：

```powershell
python -W ignore segment/train.py --data data/PennFudanPed.yaml --batch-size 1 --weights yolo-pretrained/ped.pt --cfg models/segment/yolov7-seg.yaml --epochs 20 --name blo-inst --imgsz 256 --hyp data/hyp.scratch.custom.yaml --sam_ckpt weights/sam_vit_b_01ec64.pth --device cpu --workers 0 --wandb_mode disabled
```

正式训练前，可将命令中的 `--epochs 20` 临时改为 `--epochs 1`，用于检查环境、
数据路径及权重加载；检查通过后应恢复为 `20`。


## Cross-dataset experiments

The leakage-free WheatIns/RWCellIns pipeline, target-domain detector pretraining,
Mask-IoU score calibration, boundary loss, inference tuning, TTA, and three-seed
experiment matrix are documented in
[docs/10_跨数据集实验运行说明.md](docs/10_跨数据集实验运行说明.md).

## License
This work is licensed under MIT license. See the [LICENSE](LICENSE) for details.


## Acknowledgement
The code of SRPR-Net is built upon [YOLO](https://github.com/RizwanMunawar/yolov7-segmentation) and [Segment Anything Model](https://github.com/facebookresearch/segment-anything), and we express our gratitude to these awesome projects.
