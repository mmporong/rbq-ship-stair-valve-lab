import importlib.util
import sys
from pathlib import Path

import pytest

PATH = Path(__file__).parents[1] / "tools" / "triage_rbq_select.py"
SPEC = importlib.util.spec_from_file_location("triage_rbq_select", PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = MODULE
SPEC.loader.exec_module(MODULE)


def test_validate_materializes_exact_claim_values():
    facts = {"a": 1, "b": False}
    result = MODULE.validate(
        {"candidate_id": "phase_25", "claims": ["a", "b"], "reason": "best tread"}, facts
    )
    assert result["claims"] == [{"id": "a", "value": 1}, {"id": "b", "value": False}]


def test_validate_rejects_unseen_claim():
    with pytest.raises(ValueError, match="claims"):
        MODULE.validate(
            {"candidate_id": "phase_25", "claims": ["a", "hidden"], "reason": "x"},
            {"a": 1},
        )
