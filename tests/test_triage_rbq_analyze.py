import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "tools" / "triage_rbq_analyze.py"
SPEC = importlib.util.spec_from_file_location("triage_rbq_analyze", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_sign_test_uses_only_nonzero_pairs():
    result = MODULE.sign_test_two_sided([1, 1, 1, 1, 1, -1, 0])
    assert result == {"nonzero_pairs": 6, "positive_pairs": 5, "two_sided_p": 0.21875}


def test_paired_aligns_by_case_id():
    baseline = [{"case_id": "a", "x": 1}, {"case_id": "b", "x": 3}]
    treatment = [{"case_id": "b", "x": 4}, {"case_id": "a", "x": 3}]
    result = MODULE.paired(baseline, treatment, "x")
    assert result["paired_differences"] == [2, 1]
    assert result["mean_difference"] == 1.5


@pytest.mark.parametrize("ids", [[], ["a", "a"], ["b"]])
def test_paired_requires_complete_unique_cases(ids):
    with pytest.raises(ValueError):
        MODULE.paired([{"case_id": "a", "x": 1}], [{"case_id": i, "x": 2} for i in ids], "x")
