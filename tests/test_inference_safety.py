from __future__ import annotations

import math

import numpy as np
import pytest

from src.inference_safety import (
    assess_capture_support,
    conformal_quantile,
    decide,
    group_coverage_ci,
    prediction_set,
)


def test_finite_sample_conformal_quantile_uses_requested_order_statistic():
    scores = np.arange(1, 11, dtype=float) / 10
    assert conformal_quantile(scores, alpha=0.2) == pytest.approx(0.9)
    assert math.isinf(conformal_quantile(scores, alpha=0.01))


def test_prediction_set_and_capture_support_are_fail_closed():
    assert prediction_set(0.9, 0.2) == (1,)
    assert prediction_set(0.5, 0.2) == ()
    support = {
        "cameras": {
            "Known Phone": {
                "iso_by_class": {
                    "0": {"min": 100, "max": 400},
                    "1": {"min": 80, "max": 200},
                },
                "flash_by_class": {"0": ["off"], "1": ["off"]},
            }
        }
    }
    assert assess_capture_support(
        {"camera": "Other Phone", "iso": 100, "flash_status": "off"}, support
    )
    assert assess_capture_support(
        {"camera": "Known Phone", "iso": 500, "flash_status": "off"},
        support, class_candidate=1,
    )
    assert assess_capture_support(
        {"camera": "Known Phone", "iso": 150, "flash_status": "on"}, support
    )


def test_decision_abstains_for_each_missing_or_unsupported_signal():
    args = dict(
        p_fake=0.9, quantile=0.2,
        fold_probabilities=[0.9, 0.9], fold_thresholds=[0.5, 0.5],
        mc_uncertainty=0.01, max_mc_uncertainty=0.1,
        ood_distance=0.5, max_ood_distance=1.0,
        capture_reason=None, shot_count=3,
    )
    assert decide(**args)["label"] == "FAKE"
    for changes, expected in (
        ({"shot_count": 2}, "fewer than 3 shots"),
        ({"fold_probabilities": [0.9, 0.1]}, "fold models disagree"),
        ({"mc_uncertainty": 1.0}, "MC uncertainty"),
        ({"ood_distance": None}, "embedding OOD"),
        ({"capture_reason": "unsupported camera"}, "unsupported camera"),
        ({"quantile": 0.9}, "conformal prediction set"),
    ):
        result = decide(**(args | changes))
        assert result["label"] == "INCONCLUSIVE"
        assert any(expected in reason for reason in result["reasons"])


def test_group_coverage_interval_resamples_physical_groups():
    interval = group_coverage_ci(
        [True, True, False, False], ["a", "a", "b", "b"],
        n_bootstrap=200, seed=7,
    )
    assert interval == (0.0, 1.0)
