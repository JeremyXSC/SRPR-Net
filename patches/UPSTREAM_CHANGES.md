# Upstream integration note

This delivery is an additive overlay for `importZL/BLO-Inst`.

- No upstream source file is deleted.
- `segment/train.py` and `segment/val.py` remain available as reproduction baselines.
- New entry points are `segment/train_generalized.py` and `segment/val_generalized.py`.
- The corrected D1/D2 update order, persistent outer iterator, multimodal prompt refiner, semantic prior, prompt constraints, and result export are implemented only in the new files.
- Use `tools/install_overlay.py` to copy the extension into a clean upstream clone.
