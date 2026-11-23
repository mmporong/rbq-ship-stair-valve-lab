import doctest
import importlib

import pytest

MODULES = ["robot", "transition", "lever", "loads"]


@pytest.mark.parametrize("name", MODULES)
def test_module_examples(name):
    module = importlib.import_module(f"go2_wall.{name}")
    result = doctest.testmod(module)
    assert result.failed == 0
