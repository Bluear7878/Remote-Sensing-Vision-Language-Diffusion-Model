import os
import json
import glob
import pandas as pd
import pytest
import itertools

from nunchaku.utils import get_precision, is_turing
from .utils import run_test

residual_diff_threshold_multi_list = [0.08, 0.09, 0.10, 0.11, 0.12]
residual_diff_threshold_single_list = [
    0.05, 0.06, 0.07, 0.08, 0.09,
    0.10, 0.11, 0.12, 0.13, 0.14,
    0.15, 0.16, 0.17, 0.18, 0.19,
    0.20, 0.21, 0.22, 0.23, 0.24,
    0.25, 0.26, 0.27, 0.28, 0.29, 0.30
]
Double_FBCache = [True]

height = 1024
width = 1024
num_inference_steps = 50
lora_name = None
lora_strength = 1
expected_lpips = 0.26

params = []
for doubleFB, single_thr, multi_thr in itertools.product(
    Double_FBCache,
    residual_diff_threshold_single_list,
    residual_diff_threshold_multi_list,
):
    params.append((
        doubleFB,
        single_thr,
        multi_thr,
        height,
        width,
        num_inference_steps,
        lora_name,
        lora_strength,
        expected_lpips,
    ))
    params.append((
        doubleFB,
        single_thr,
        multi_thr,
        height,
        width,
        num_inference_steps,
        lora_name,
        lora_strength,
        expected_lpips,
    ))


@pytest.mark.skipif(is_turing(), reason="Skip tests for Turing GPUs")
@pytest.mark.parametrize(
    (
        "Double_FBCache,residual_diff_threshold_single,residual_diff_threshold_multi,"
        "height,width,num_inference_steps,lora_name,lora_strength,expected_lpips"
    ),
    params,
)
def test_flux_dev_loras(
    Double_FBCache: bool,
    residual_diff_threshold_single: float,
    residual_diff_threshold_multi: float,
    height: int,
    width: int,
    num_inference_steps: int,
    lora_name: str,
    lora_strength: float,
    expected_lpips: float,
):
    run_test(
        precision=get_precision(),
        model_name="flux.1-dev",
        dataset_name="MJHQ" if lora_name is None else lora_name,
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=3.5,
        use_qencoder=False,
        cpu_offload=False,
        lora_names=lora_name,
        Double_FBCache=Double_FBCache,
        residual_diff_threshold_single=residual_diff_threshold_single,
        residual_diff_threshold_multi=residual_diff_threshold_multi,
        expected_lpips=expected_lpips,
    )
    directory = "test_results"

    json_files = glob.glob(os.path.join(directory, "summary_*.json"))
    data_list = []

    for file in json_files:
        with open(file, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            json_data["filename"] = os.path.basename(file)
            data_list.append(json_data)

    df = pd.DataFrame(data_list)
    excel_filename = os.path.join(directory, "summary_data.xlsx")
    df.to_excel(excel_filename, index=False)
