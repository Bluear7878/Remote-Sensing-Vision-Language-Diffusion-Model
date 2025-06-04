import argparse
import os

import torch
from safetensors.torch import save_file
from tqdm import tqdm


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("input_path", type=str, help="Path to checkpoint file")
    parser.add_argument(
        "-o", "--output-path", type=str, help="Path to save the output checkpoint file", default="output.safetensors"
    )
    args = parser.parse_args()
    return args


def swap_scale_shift(weight: torch.Tensor) -> torch.Tensor:
    shift, scale = weight.chunk(2, dim=0)
    new_weight = torch.cat([scale, shift], dim=0)
    return new_weight


def main():
    args = get_args()

    original_state_dict = torch.load(args.input_path, map_location="cpu")

    new_state_dict = {
        "transformer.x_embedder.weight": original_state_dict["img_in"]["weight"],
        "transformer.x_embedder.bias": original_state_dict["img_in"]["bias"],
        "transformer.norm_out.linear.weight": swap_scale_shift(
            original_state_dict["final_layer"]["adaLN_modulation.1.weight"]
        ),
        "transformer.norm_out.linear.bias": swap_scale_shift(
            original_state_dict["final_layer"]["adaLN_modulation.1.bias"]
        ),
        "transformer.proj_out.weight": original_state_dict["final_layer"]["linear.weight"],
        "transformer.proj_out.bias": original_state_dict["final_layer"]["linear.bias"],
    }
    original_state_dict.pop("img_in")
    original_state_dict.pop("final_layer")

    original_lora_state_dict = original_state_dict["lora"]
    for k, v in tqdm(original_lora_state_dict.items()):
        if "double_blocks" in k:
            new_k = k.replace("double_blocks", "transformer.transformer_blocks").replace(".default", "")
            if "qkv" in new_k:
                for i, p in enumerate(["q", "k", "v"]):
                    if "lora_A" in new_k:
                        # Copy the tensor
                        new_k2 = new_k.replace("img_attn.qkv", f"attn.to_{p}")
                        new_k2 = new_k2.replace("txt_attn.qkv", f"attn.add_{p}_proj")
                        new_state_dict[new_k2] = v.clone()
                    else:
                        assert "lora_B" in new_k
                        assert v.shape[0] % 3 == 0
                        chunk_size = v.shape[0] // 3
                        new_k2 = new_k.replace("img_attn.qkv", f"attn.to_{p}")
                        new_k2 = new_k2.replace("txt_attn.qkv", f"attn.add_{p}_proj")
                        new_state_dict[new_k2] = v[i * chunk_size : (i + 1) * chunk_size]
            else:
                new_k = new_k.replace("img_mod.lin", "norm1.linear")
                new_k = new_k.replace("txt_mod.lin", "norm1_context.linear")
                new_k = new_k.replace("img_mlp.0", "ff.net.0.proj")
                new_k = new_k.replace("img_mlp.2", "ff.net.2")
                new_state_dict[new_k] = v
        else:
            assert "single_blocks" in k
            new_k = k.replace("single_blocks", "transformer.single_transformer_blocks").replace(".default", "")
            if "linear1" in k:
                start = 0
                for i, p in enumerate(["q", "k", "v", "i"]):
                    if "lora_A" in new_k:
                        if p == "i":
                            new_k2 = new_k.replace("linear1", "proj_mlp")
                        else:
                            new_k2 = new_k.replace("linear1", f"attn.to_{p}")
                        new_state_dict[new_k2] = v.clone()
                    else:
                        if p == "i":
                            new_k2 = new_k.replace("linear1", "proj_mlp")
                        else:
                            new_k2 = new_k.replace("linear1", f"attn.to_{p}")
                        chunk_size = 12288 if p == "i" else 3072
                        new_state_dict[new_k2] = v[start : start + chunk_size]
                        start += chunk_size
            elif "linear2" in k:
                new_k = new_k.replace("linear2", "proj_out")
                new_k = new_k.replace("modulation_lin", ".norm.linear")
                new_state_dict[new_k] = v
            else:
                assert "modulation.lin" in k
                new_k = new_k.replace("modulation.lin", "norm.linear")
                new_state_dict[new_k] = v

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_file(new_state_dict, args.output_path)


if __name__ == "__main__":
    main()
