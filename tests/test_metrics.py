import numpy as np
from agentic_rob.metrics import attack_summary, robustness_gap, wilson_ci, paired_bootstrap_diff, per_class_table


def test_asr_uses_true_labels_and_clean_correct_denominator():
    y = np.array([0, 0, 1, 1])
    pc = np.array([0, 0, 1, 0])            # 3 clean-correct
    pa = np.array([1, 0, 0, 0])            # of those: sample0 and sample2 now wrong
    s = attack_summary(y, pc, np.ones(4), pa, np.ones(4) * .5)
    assert abs(s["asr"] - 100 * 2 / 3) < 1e-6
    assert abs(s["acc"] - 25.0) < 1e-6
    assert abs(s["flip_rate"] - 50.0) < 1e-6


def test_gap_and_ci_and_bootstrap():
    gap, worst = robustness_gap(99.5, {"fgsm": 40.0, "pgd": 28.0, "cw": 35.0})
    assert worst == "pgd" and abs(gap - 71.5) < 1e-9
    lo, hi = wilson_ci(50, 100)
    assert lo < 50 < hi
    a = np.array([1, 0, 0, 0] * 50); b = np.array([1, 1, 1, 0] * 50)
    m, lo, hi = paired_bootstrap_diff(a, b, 300)
    assert abs(m - 50) < 1e-6 and lo < m < hi


def test_per_class_has_macro_rows():
    rows = per_class_table(np.array([0, 1, 2, 2]), np.array([0, 1, 2, 1]), ["a", "b", "c"])
    assert [r["cls"] for r in rows][-2:] == ["macro", "weighted"]
