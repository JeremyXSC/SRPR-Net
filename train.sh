#!/usr/bin/env bash
set -euo pipefail

python segment/train_generalized.py \
    --config configs/generalized_blo_pennfudan.yaml \
    --set training.device=cpu \
    --set training.batch_size=1 \
    --set training.workers=0 \
    --set training.amp=false \
    --set foundation.enabled=true \
    --set refiner.enabled=true \
    --set refiner.use_attention=true \
    --set loss.lambda_constraint=0.0 \
    --set semantic_prior.enabled=false \
    --set loss.lambda_semantic=0.0
