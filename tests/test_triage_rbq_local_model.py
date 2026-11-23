import importlib.util
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).parents[1] / "tools/triage_rbq_local_model.py"
SPEC = importlib.util.spec_from_file_location("triage_rbq_local_model", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


PROTOCOL = {
    "allowed_actions": ["runtime_contact_window", "HOLD"],
    "allowed_citations": ["a", "b"],
}


def test_validate_accepts_bounded_hold():
    value = {"action": "HOLD", "cited_facts": ["a", "b"], "reason": "out of scope"}
    assert MODULE.validate(value, PROTOCOL) == value


@pytest.mark.parametrize(
    "value",
    [
        {"action": "train", "cited_facts": ["a", "b"], "reason": "x"},
        {"action": "HOLD", "cited_facts": ["a", "a"], "reason": "x"},
        {"action": "HOLD", "cited_facts": ["a", "unknown"], "reason": "x"},
    ],
)
def test_validate_rejects_unregistered_or_unbound_output(value):
    with pytest.raises(ValueError):
        MODULE.validate(value, PROTOCOL)
