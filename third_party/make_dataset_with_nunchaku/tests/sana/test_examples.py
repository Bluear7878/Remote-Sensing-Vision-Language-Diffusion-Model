import os
import subprocess

import pytest

from nunchaku.utils import get_precision, is_turing

EXAMPLES_DIR = "./examples"

example_scripts = [f for f in os.listdir(EXAMPLES_DIR) if f.endswith(".py") and f.startswith("sana")]


@pytest.mark.skipif(
    is_turing() or get_precision() == "fp4", reason="SANA does not support Turing GPUs or FP4 precision"
)
@pytest.mark.parametrize("script_name", example_scripts)
def test_example_script_runs(script_name):
    script_path = os.path.join(EXAMPLES_DIR, script_name)
    result = subprocess.run(["python", script_path], capture_output=True, text=True)
    print(f"Running {script_path} -> Return code: {result.returncode}")
    assert result.returncode == 0, f"{script_path} failed with code {result.returncode}"
