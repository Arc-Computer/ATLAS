"""Unit tests for GKD baseline evaluation metrics."""

import pytest

from trainers.gkd_evaluator import BaselineMetricsCallback, compute_baseline_summary


def test_baseline_callback_initialization():
    """Test callback initializes with baseline metrics."""
    callback = BaselineMetricsCallback(
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    assert callback.baseline_success == 0.75
    assert callback.baseline_tokens == 1200


def test_baseline_callback_computes_success_delta():
    """Test success delta calculation."""
    callback = BaselineMetricsCallback(baseline_success=0.75)

    metrics = {"eval_success_rate": 0.87}

    # Simulate on_evaluate call
    callback.on_evaluate(
        args=None,
        state=None,
        control=None,
        metrics=metrics,
    )

    # Check success delta
    assert "metrics/success_delta" in metrics
    assert pytest.approx(metrics["metrics/success_delta"], rel=1e-6) == 0.12

    # Check meets target flag
    assert "metrics/meets_target" in metrics
    assert metrics["metrics/meets_target"] is True  # 12pp >= 10pp


def test_baseline_callback_computes_token_reduction():
    """Test token reduction calculation."""
    callback = BaselineMetricsCallback(
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    metrics = {
        "eval_success_rate": 0.85,
        "eval_avg_tokens": 780,
    }

    callback.on_evaluate(
        args=None,
        state=None,
        control=None,
        metrics=metrics,
    )

    # Check token reduction
    assert "metrics/token_reduction_pct" in metrics
    expected_reduction = (1200 - 780) / 1200 * 100  # 35%
    assert pytest.approx(metrics["metrics/token_reduction_pct"], rel=1e-6) == expected_reduction

    # Check meets token target
    assert "metrics/token_target_met" in metrics
    assert metrics["metrics/token_target_met"] is True  # 35% >= 30%


def test_baseline_callback_target_not_met():
    """Test when targets are not met."""
    callback = BaselineMetricsCallback(
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    metrics = {
        "eval_success_rate": 0.78,  # Only 3pp improvement
        "eval_avg_tokens": 900,  # Only 25% reduction
    }

    callback.on_evaluate(
        args=None,
        state=None,
        control=None,
        metrics=metrics,
    )

    # Success target not met
    assert pytest.approx(metrics["metrics/success_delta"], rel=1e-6) == 0.03
    assert metrics["metrics/meets_target"] is False

    # Token target not met
    assert pytest.approx(metrics["metrics/token_reduction_pct"], rel=1e-6) == 25.0
    assert metrics["metrics/token_target_met"] is False


def test_baseline_callback_passes_through_learning_metrics():
    """Test that Atlas learning metrics are passed through."""
    callback = BaselineMetricsCallback(baseline_success=0.75)

    metrics = {
        "eval_success_rate": 0.85,
        "eval_cue_hit_rate": 0.92,
        "eval_adoption_rate": 0.88,
        "eval_reward_delta": 0.15,
        "eval_token_delta": -0.35,
        "eval_transfer_success": 0.78,
    }

    callback.on_evaluate(
        args=None,
        state=None,
        control=None,
        metrics=metrics,
    )

    # Check learning metrics are passed through
    assert metrics["metrics/cue_hit_rate"] == 0.92
    assert metrics["metrics/adoption_rate"] == 0.88
    assert metrics["metrics/reward_delta"] == 0.15
    assert metrics["metrics/token_delta"] == -0.35
    assert metrics["metrics/transfer_success"] == 0.78


def test_baseline_callback_handles_missing_baselines():
    """Test callback handles missing baseline metrics gracefully."""
    callback = BaselineMetricsCallback(
        baseline_success=0.0,
        baseline_tokens=None,
    )

    metrics = {
        "eval_success_rate": 0.85,
        "eval_avg_tokens": 800,
    }

    callback.on_evaluate(
        args=None,
        state=None,
        control=None,
        metrics=metrics,
    )

    # Success delta computed even with 0 baseline
    assert "metrics/success_delta" in metrics
    assert metrics["metrics/success_delta"] == 0.85

    # Token reduction not computed (baseline missing)
    assert "metrics/token_reduction_pct" not in metrics
    assert "metrics/token_target_met" not in metrics


def test_compute_baseline_summary_basic():
    """Test stage 2 summary computation."""
    eval_results = {
        "eval_success_rate": 0.87,
        "eval_avg_tokens": 780,
    }

    summary = compute_baseline_summary(
        eval_results,
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    # Check summary fields
    assert "success_delta" in summary
    assert "success_relative" in summary
    assert "token_reduction_pct" in summary
    assert "meets_success_target" in summary
    assert "meets_token_target" in summary
    assert "meets_all_targets" in summary

    # Check values
    assert pytest.approx(summary["success_delta"], rel=1e-6) == 0.12
    assert pytest.approx(summary["success_relative"], rel=1e-6) == 0.16  # 16% relative
    assert pytest.approx(summary["token_reduction_pct"], rel=1e-6) == 35.0
    assert summary["meets_success_target"] is True
    assert summary["meets_token_target"] is True
    assert summary["meets_all_targets"] is True


def test_compute_baseline_summary_partial_success():
    """Test summary when only some targets are met."""
    eval_results = {
        "eval_success_rate": 0.88,  # 13pp improvement (meets target)
        "eval_avg_tokens": 900,  # 25% reduction (does not meet target)
    }

    summary = compute_baseline_summary(
        eval_results,
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    assert summary["meets_success_target"] is True
    assert summary["meets_token_target"] is False
    assert summary["meets_all_targets"] is False


def test_compute_baseline_summary_without_token_baseline():
    """Test summary computation without token baseline."""
    eval_results = {
        "eval_success_rate": 0.87,
        "eval_avg_tokens": 780,
    }

    summary = compute_baseline_summary(
        eval_results,
        baseline_success=0.75,
        baseline_tokens=None,
    )

    # Success metrics present
    assert summary["success_delta"] == 0.12
    assert summary["meets_success_target"] is True

    # Token metrics None (baseline missing)
    assert summary["token_reduction_pct"] is None
    assert summary["meets_token_target"] is None

    # All targets met if success target met (token target not applicable)
    assert summary["meets_all_targets"] is True


def test_compute_baseline_summary_zero_baseline():
    """Test handling of zero baseline (edge case)."""
    eval_results = {"eval_success_rate": 0.85}

    summary = compute_baseline_summary(
        eval_results,
        baseline_success=0.0,
    )

    # Success delta calculated
    assert summary["success_delta"] == 0.85

    # Relative improvement is 0 (can't divide by zero baseline)
    assert summary["success_relative"] == 0.0


def test_compute_baseline_summary_negative_delta():
    """Test handling of negative improvement (regression)."""
    eval_results = {
        "eval_success_rate": 0.70,  # Regression
        "eval_avg_tokens": 1500,  # More tokens than baseline
    }

    summary = compute_baseline_summary(
        eval_results,
        baseline_success=0.75,
        baseline_tokens=1200,
    )

    # Negative deltas
    assert pytest.approx(summary["success_delta"], rel=1e-6) == -0.05
    assert summary["token_reduction_pct"] == -25.0  # Negative = increase

    # Targets not met
    assert summary["meets_success_target"] is False
    assert summary["meets_token_target"] is False
    assert summary["meets_all_targets"] is False
