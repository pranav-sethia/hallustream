import numpy as np

from hallustream.research.evaluate import (
    apply_platt,
    ece,
    fit_platt,
    operating_point,
    threshold_at_fpr,
)


def test_threshold_keeps_false_alarms_at_or_below_target():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    s = rng.random(5000) + 0.3 * y
    thr = threshold_at_fpr(s, y, 0.10)
    assert operating_point(s, y, thr)["fpr"] <= 0.10


def test_platt_is_monotonic_and_improves_calibration():
    rng = np.random.default_rng(0)
    p_true = rng.random(20000)
    y = (rng.random(20000) < p_true).astype(int)
    overconfident = 1 / (1 + np.exp(-3 * np.log(p_true / (1 - p_true))))
    cal = apply_platt(overconfident, fit_platt(overconfident, y))
    assert (np.diff(cal[np.argsort(overconfident)]) >= -1e-12).all()
    assert ece(cal, y) < ece(overconfident, y)
