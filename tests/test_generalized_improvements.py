from __future__ import annotations

import random
import unittest

import numpy as np
import torch

from generalized.losses import (
    binary_mask_iou,
    geometric_score_fusion,
    sobel_boundary_loss,
)
from generalized.training_utils import capture_rng_state, restore_rng_state
from segment.generalized_runtime import descale_tta_prediction, resolve_max_det


class ImprovementLossTests(unittest.TestCase):
    def test_binary_mask_iou_is_one_for_perfect_mask(self):
        targets = torch.tensor([[[1.0, 0.0], [0.0, 1.0]]])
        logits = torch.tensor([[[10.0, -10.0], [-10.0, 10.0]]])
        self.assertTrue(torch.allclose(binary_mask_iou(logits, targets), torch.ones(1)))

    def test_score_fusion_endpoints(self):
        detector = torch.tensor([0.2, 0.8])
        quality = torch.tensor([0.9, 0.3])
        self.assertTrue(
            torch.allclose(geometric_score_fusion(detector, quality, 0.0), detector)
        )
        self.assertTrue(
            torch.allclose(geometric_score_fusion(detector, quality, 1.0), quality)
        )

    def test_boundary_loss_prefers_matching_mask(self):
        targets = torch.zeros((1, 8, 8))
        targets[:, 2:6, 2:6] = 1.0
        matching = torch.where(targets > 0, torch.tensor(10.0), torch.tensor(-10.0))
        shifted = torch.roll(matching, shifts=2, dims=-1)
        self.assertLess(
            float(sobel_boundary_loss(matching, targets)),
            float(sobel_boundary_loss(shifted, targets)),
        )


class RuntimeTests(unittest.TestCase):
    def test_auto_max_det_uses_annotation_density_and_caps(self):
        config = {"inference": {"max_det": "auto"}}
        labels = [np.zeros((12, 6)), np.zeros((400, 6))]
        self.assertEqual(resolve_max_det(config, labels), 500)
        self.assertEqual(config["inference"]["_resolved_max_det"], 500)

    def test_tta_inverse_scale_and_horizontal_flip(self):
        prediction = torch.tensor([[[40.0, 20.0, 10.0, 8.0, 0.5]]])
        restored = descale_tta_prediction(
            prediction,
            flip_dimension=3,
            scale=2.0,
            image_size=(50, 100),
        )
        expected = torch.tensor([[[80.0, 10.0, 5.0, 4.0, 0.5]]])
        self.assertTrue(torch.allclose(restored, expected))

    def test_rng_state_round_trip(self):
        random.seed(7)
        np.random.seed(7)
        torch.manual_seed(7)
        state = capture_rng_state()
        expected = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        if torch.cuda.is_available():
            # Mirrors loading a whole checkpoint with map_location="cuda".
            state["torch"] = state["torch"].cuda()
            state["cuda"] = [cuda_state.cuda() for cuda_state in state["cuda"]]
        restore_rng_state(state)
        actual = (random.random(), float(np.random.rand()), float(torch.rand(1)))
        self.assertEqual(expected, actual)


if __name__ == "__main__":
    unittest.main()
