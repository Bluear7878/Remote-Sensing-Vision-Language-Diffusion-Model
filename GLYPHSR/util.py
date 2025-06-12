import copy
import json
import math
import os
import re
import string

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import OmegaConf
from PIL import Image
from torch.nn.functional import interpolate

from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.mm_utils import tokenizer_image_token
from sgm.util import instantiate_from_config


def get_img_describe(
    image_tensor,
    image,
    model,
    tokenizer,
    prompt,
    conv_templates,
    image_token_index,
    conv_template="llava_llama_3",
    num_beams=1,
    temperature = 0.2,
    do_sample=True,
    max_new_tokens=512,
    device="cuda"
):
    question = prompt

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)

    prompt_question = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_question,
        tokenizer,
        image_token_index,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_sizes = [image.size]

    with torch.inference_mode():
        outputs = model.generate(
        input_ids,
        images=image_tensor,
        image_sizes=image_sizes,
        do_sample=do_sample,
        temperature=temperature,
        num_beams = num_beams,
        max_new_tokens=max_new_tokens,
        return_dict_in_generate=True,
        output_scores=True
    )

    generated_output = outputs[0][0].cpu().tolist()
    image_caption = tokenizer.decode(generated_output, skip_special_tokens=True)
    image_caption = [image_caption.lstrip()]

    return image_caption

def get_state_dict(d):
    return d.get('state_dict', d)


def load_state_dict(ckpt_path, location='cpu'):
    _, extension = os.path.splitext(ckpt_path)
    if extension.lower() == ".safetensors":
        import safetensors.torch
        state_dict = safetensors.torch.load_file(ckpt_path, device=location)
    else:
        state_dict = get_state_dict(torch.load(ckpt_path, map_location=torch.device(location)))
    state_dict = get_state_dict(state_dict)
    print(f'Loaded state_dict from [{ckpt_path}]')
    return state_dict


def create_model(config_path):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    return model


def create_SR_model(config_path, SUPIR_sign=None, load_default_setting=False):
    config = OmegaConf.load(config_path)
    model = instantiate_from_config(config.model).cpu()
    print(f'Loaded model config from [{config_path}]')
    if config.SDXL_CKPT is not None:
        model.load_state_dict(load_state_dict(config.SDXL_CKPT), strict=False)
    if config.SUPIR_CKPT is not None:
        model.load_state_dict(load_state_dict(config.SUPIR_CKPT), strict=False)
    if SUPIR_sign is not None:
        assert SUPIR_sign in ['F', 'Q']
        if SUPIR_sign == 'F':
            model.load_state_dict(load_state_dict(config.SUPIR_CKPT_F), strict=False)
        elif SUPIR_sign == 'Q':
            model.load_state_dict(load_state_dict(config.SUPIR_CKPT_Q), strict=False)
    if load_default_setting:
        default_setting = config.default_setting
        return model, default_setting
    return model

def load_QF_ckpt(config_path):
    config = OmegaConf.load(config_path)
    ckpt_F = torch.load(config.SUPIR_CKPT_F, map_location='cpu')
    ckpt_Q = torch.load(config.SUPIR_CKPT_Q, map_location='cpu')
    return ckpt_Q, ckpt_F


#####################CUSTOMRIZED#####################
def degrade_image(img_pil: Image.Image, down_factor: float = 1.5) -> Image.Image:
    """
    Degrade an image by downsampling and then upsampling to introduce aliasing.
    """
    w, h = img_pil.size
    small = img_pil.resize((int(w / down_factor), int(h / down_factor)), Image.BICUBIC)
    degraded = small.resize((w, h), Image.BICUBIC)
    # Convert to CV2 array and back to ensure consistent format
    img_cv = cv2.cvtColor(np.array(degraded), cv2.COLOR_RGB2BGR)
    return Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))


def load_metadata(meta_path: str):
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def remove_special_chars_and_merge(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    merged_text = " ".join(cleaned_text.split())
    return merged_text


def beam_search_topk_per_step(step_topk, beam_size):

    beam = [("", 0.0)]

    for step_candidates in step_topk:
        new_beam = []
        for (prev_seq, prev_logprob) in beam:
            for c in step_candidates:
                if c['prob'] <= 0.0:
                    continue
                token_str = c['token_str']
                prob = c['prob']
                new_seq = remove_special_chars_and_merge(prev_seq + token_str)
                new_logprob = prev_logprob + math.log(prob)
                new_beam.append((new_seq, new_logprob))

        new_beam.sort(key=lambda x: x[1], reverse=True)
        beam = new_beam[:beam_size]

    return beam

def get_beam_search_results(outputs, tokenizer, top_k, beams_num):

    token_probs = []
    for step_logits in outputs.scores:
        step_logits = step_logits[0]
        probs = F.softmax(step_logits, dim=-1)
        topk_probs, topk_indices = probs.topk(top_k)
        topk_data = []
        for i in range(top_k):
            token_id = topk_indices[i].item()
            token_str = tokenizer.decode([token_id], skip_special_tokens=True)
            prob = topk_probs[i].item()
            topk_data.append({
                "token_id": token_id,
                "token_str": token_str,
                "prob": round(prob, 4)
            })
        token_probs.append(topk_data)


    beam_results = beam_search_topk_per_step(token_probs, beam_size=beams_num)
    beam_results_list = [repr(seq).lstrip("\\n") for seq, _ in beam_results]
    sorted_candidates = sorted(beam_results, key=lambda x: x[1], reverse=True)

    # Build a prompt string that lists the candidates in ranked order.
    prompt_lines = ["Ranked Candidates:"]
    for rank, (name, score) in enumerate(sorted_candidates, start=1):
        # Optionally, include the score with two decimals.
        prompt_lines.append(f"{rank}. {name}")

    # Combine all lines into a single prompt string.
    prompt = " ".join(prompt_lines)

    return token_probs, beam_results_list,prompt


def PIL2Tensor(img, upscale=1, min_size=1024, fix_resize=None):
    '''
    PIL.Image -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    w, h = img.size
    w *= upscale
    h *= upscale
    w0, h0 = round(w), round(h)
    if min(w, h) < min_size:
        _upsacle = min_size / min(w, h)
        w *= _upsacle
        h *= _upsacle
    if fix_resize is not None:
        _upsacle = fix_resize / min(w, h)
        w *= _upsacle
        h *= _upsacle
        w0, h0 = round(w), round(h)
    w = int(np.round(w / 64.0)) * 64
    h = int(np.round(h / 64.0)) * 64
    x = img.resize((w, h), Image.BICUBIC)
    x = np.array(x).round().clip(0, 255).astype(np.uint8)
    x = x / 255 * 2 - 1
    x = torch.tensor(x, dtype=torch.float32).permute(2, 0, 1)
    return x, h0, w0

def Tensor2PIL(x, h0, w0):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    x = x.unsqueeze(0)
    x = interpolate(x, size=(h0, w0), mode='bicubic')
    x = (x.squeeze(0).permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return Image.fromarray(x)


def HWC3(x):
    assert x.dtype == np.uint8
    if x.ndim == 2:
        x = x[:, :, None]
    assert x.ndim == 3
    H, W, C = x.shape
    assert C == 1 or C == 3 or C == 4
    if C == 3:
        return x
    if C == 1:
        return np.concatenate([x, x, x], axis=2)
    if C == 4:
        color = x[:, :, 0:3].astype(np.float32)
        alpha = x[:, :, 3:4].astype(np.float32) / 255.0
        y = color * alpha + 255.0 * (1.0 - alpha)
        y = y.clip(0, 255).astype(np.uint8)
        return y


def upscale_image(input_image, upscale, min_size=None, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    H *= upscale
    W *= upscale
    if min_size is not None:
        if min(H, W) < min_size:
            _upsacle = min_size / min(W, H)
            W *= _upsacle
            H *= _upsacle
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img


def fix_resize(input_image, size=512, unit_resolution=64):
    H, W, C = input_image.shape
    H = float(H)
    W = float(W)
    upscale = size / min(H, W)
    H *= upscale
    W *= upscale
    H = int(np.round(H / unit_resolution)) * unit_resolution
    W = int(np.round(W / unit_resolution)) * unit_resolution
    img = cv2.resize(input_image, (W, H), interpolation=cv2.INTER_LANCZOS4 if upscale > 1 else cv2.INTER_AREA)
    img = img.round().clip(0, 255).astype(np.uint8)
    return img



def Numpy2Tensor(img):
    '''
    np.array[H, w, C] [0, 255] -> Tensor[C, H, W], RGB, [-1, 1]
    '''
    # size
    img = np.array(img) / 255 * 2 - 1
    img = torch.tensor(img, dtype=torch.float32).permute(2, 0, 1)
    return img


def Tensor2Numpy(x, h0=None, w0=None):
    '''
    Tensor[C, H, W], RGB, [-1, 1] -> PIL.Image
    '''
    if h0 is not None and w0 is not None:
        x = x.unsqueeze(0)
        x = interpolate(x, size=(h0, w0), mode='bicubic')
        x = x.squeeze(0)
    x = (x.permute(1, 2, 0) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
    return x


def convert_dtype(dtype_str):
    if dtype_str == 'fp32':
        return torch.float32
    elif dtype_str == 'fp16':
        return torch.float16
    elif dtype_str == 'bf16':
        return torch.bfloat16
    else:
        raise NotImplementedError
