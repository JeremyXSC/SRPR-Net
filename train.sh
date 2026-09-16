#!/usr/bin/env bash
set -euo pipefail

python -W ignore segment/train.py \
    --data data/PennFudanPed.yaml \
    --batch-size 1 \
    --weights yolo-pretrained/ped.pt \
    --cfg models/segment/yolov7-seg.yaml \
    --epochs 20 \
    --name blo-inst \
    --imgsz 256 \
    --hyp data/hyp.scratch.custom.yaml \
    --sam_ckpt weights/sam_vit_b_01ec64.pth \
    --device cpu \
    --workers 0 \
    --wandb_mode disabled
