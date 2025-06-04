import os
from glob import glob
import torch
import argparse

import numpy as np
import PIL
from PIL import Image
from concurrent.futures import ThreadPoolExecutor

import torch, torchvision.transforms.functional as TF
import torchmetrics.functional as TMF
import lpips

# type detection ####################################################################################################
# ( list|path ) ----> list(PIL.Image.image)\

def type_navi(hq, lq):
    """
    input type : 
    1. file path : list
    2. PIL.Image.Jpeg : list
    3. PIL.Image.Png : list
    4. directory path : str
    """
    if type(hq)==list and type(lq)==list:
        # list
        if len(hq) != len(lq):
            raise ValueError(f"The number of incoming images is not equal. [(image1, image2) == ({len(hq)}, {len(lq)})]")
        
        hq_path = []
        lq_path = []
        for h, l in zip(hq, lq):
            if type(h) == str:
                if os.path.exists(h):
                    hq_path.append(Image.open(os.path.join(h)))
            if type(h) == PIL.PngImagePlugin.PngImageFile:
                hq_path.append(h.convert("RGB"))
            if type(h) == PIL.JpegImagePlugin.JpegImageFile:
                hq_path.append(h.convert("RGB"))
            
            if type(l) == str:
                if os.path.exists(l):
                    lq_path.append(Image.open(os.path.join(l)))
            if type(l) == PIL.PngImagePlugin.PngImageFile:
                lq_path.append(l.convert("RGB"))
            if type(l) == PIL.JpegImagePlugin.JpegImageFile:
                lq_path.append(l.convert("RGB"))
        return hq_path, lq_path
        
    elif type(hq)==str and type(lq)==str:
        # directory  
        if os.path.exists(hq) and os.path.exists(lq):
            if os.path.isdir(hq) and os.path.isdir(lq):
                hq_path = glob(os.path.join(hq, "*.jpg"))
                if not hq_path:
                    hq_path = glob(os.path.join(hq, "*.png"))

                lq_path = glob(os.path.join(lq, "*.jpg"))
                if not lq_path:
                    lq_path = glob(os.path.join(lq, "*.png"))

                hq = [Image.open(i).convert("RGB") for i in sorted(hq_path)]
                lq = [Image.open(i).convert("RGB") for i in sorted(lq_path)]

                if len(hq_path) != len(lq_path):
                    raise ValueError(f"The number of incoming images is not equal. [(image1, image2) == ({len(hq_path)}, {len(lq_path)})]")
                    
            return hq, lq
    else :
        raise TypeError("input is list or str(path)")


# image2numpy ####################################################################################################
# PIL.Image ----> numpy.array
from math import ceil

TARGET_MIN = 224
def image_preprocessing(hq_imgs, lq_imgs,
                        resize_filter=Image.LANCZOS,
                        workers: int = 8):
    """
    return : (HQ_np_list, LQ_np_list)
        - shape = (H, W, 3), dtype = uint8, channel : RGB
    """
    # hq_imgs, lq_imgs = type_navi(HQ_dir_or_list, LQ_dir_or_list)

    if len(hq_imgs) != len(lq_imgs):
        raise ValueError(f"HQ/LQ dismatch count. ({len(hq_imgs)} vs {len(lq_imgs)})")

    # 2) Resize image → NumPy
    def process_pair(pair):
        hi, lo = pair

        w, h = hi.size
        short = min(w, h)
        if short <= TARGET_MIN:
            scale   = 225 / short
            new_sz  = (ceil(w * scale), ceil(h * scale))
            hi = hi.resize(new_sz, resize_filter)
        lo = lo.resize(hi.size, resize_filter)

        return np.array(hi), np.array(lo)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(process_pair, zip(hq_imgs, lq_imgs)))

    HQ_np, LQ_np = map(list, zip(*results))   # unzip
    return HQ_np, LQ_np


# list(PIL.Image.image) ----> list(tensor)
def pil_to_01_tensor(pil):
    t = TF.to_tensor(pil)          # [0,1] float32
    return t

def match_size(src, ref):
    """src 를 ref 크기로 bicubic 리사이즈 (필요할 때만)"""
    if src.shape[-2:] != ref.shape[-2:]:
        src = TF.resize(src, ref.shape[-2:], antialias=True)
    return src

@torch.no_grad()
def pil2tensor(hr_pil, sr_pil):
    hr_list, sr_list = [], []
    for _sr, _hr in zip(sr_pil, hr_pil):
        # 1. PIL → tensor
        sr = pil_to_01_tensor(_sr)
        hr = pil_to_01_tensor(_hr)

        # 2. match size
        sr = match_size(sr, hr)

        # 3. CPU ---> GPU
        sr, hr = sr.unsqueeze(0).cuda(), hr.unsqueeze(0).cuda()   # (1,3,H,W)
        hr_list.append(hr); sr_list.append(sr)

    return hr_list, sr_list


# PSNR SCORE ####################################################################################################
def psnr(img1, img2):
    li = [TMF.peak_signal_noise_ratio(im1, im2, data_range=1.0) for im1, im2 in zip(img1, img2)]
    return float(sum(li)/len(li))

# SSIM SCORE ####################################################################################################
def ssim(img1:list, img2:list) -> float:
    li = [TMF.structural_similarity_index_measure(im1, im2, data_range=1.0).item() for im1, im2 in zip(img1, img2)]
    return float(sum(li)/len(li))

# LPIPS SCORE ####################################################################################################
def mlpips(img1, img2):
    _lpips = lpips.LPIPS(net='vgg').eval().cuda()   # perceptual distance
    li = [_lpips(sr*2-1, hr*2-1).mean().item() for sr, hr  in zip(img2, img1)]
    return float(sum(li)/len(li))


# MUSIQ SCORE ####################################################################################################
import tensorflow as tf
import tensorflow_hub as hub
from io import BytesIO
from tqdm import tqdm

NAME_TO_HANDLE = {
    # Model trained on SPAQ dataset: https://github.com/h4nwei/SPAQ
    'spaq': 'https://tfhub.dev/google/musiq/spaq/1',

    # Model trained on KonIQ-10K dataset: http://database.mmsp-kn.de/koniq-10k-database.html
    'koniq': 'https://tfhub.dev/google/musiq/koniq-10k/1',

    # Model trained on PaQ2PiQ dataset: https://github.com/baidut/PaQ-2-PiQ
    'paq2piq': 'https://tfhub.dev/google/musiq/paq2piq/1',

    # Model trained on AVA dataset: https://ieeexplore.ieee.org/document/6247954
    'ava': 'https://tfhub.dev/google/musiq/ava/1',
}

def MUSIQ_SCORE(resolution:list):
    byte_io = BytesIO()
    model = hub.load(NAME_TO_HANDLE['koniq'])
    predict_fn = model.signatures['serving_default']

    scores = []
    for image in tqdm(resolution):
        image.save(byte_io, format='JPEG')  # save type to JPEG
        byte_io.seek(0)  # BytesIO start position

        image_bytes = byte_io.getvalue()
        prediction = float(predict_fn(tf.constant(image_bytes))["output_0"])

        scores.append(prediction)

    return sum(scores)/len(scores)

# OCR #####################################################################################################
import re
from difflib import SequenceMatcher
from nltk.metrics import edit_distance
import sys
from collections import Counter
import os
import json
import argparse


def preprocess(text):
    """Remove punctuation, lowercase, and strip whitespace."""
    text = re.sub(r'[^\w\s]', '', text)
    return text.lower().strip()


def calculate_pos_neg(ground_truth_labels, predicted_labels):
    """Compute TP, FP, FN counts at the word level."""
    total_tp, total_fp, total_fn = 0, 0, 0
    total_gt_words = 0

    for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
        # Tokenize and normalize
        gt_words = preprocess(gt_label).split()
        pred_words = preprocess(pred_label).split()

        total_gt_words += len(gt_words)

        used_pred = set()
        used_gt = set()

        # Exact match (TP)
        for i, gt_word in enumerate(gt_words):
            for j, pred_word in enumerate(pred_words):
                if i not in used_gt and j not in used_pred and gt_word == pred_word:
                    total_tp += 1
                    used_pred.add(j)
                    used_gt.add(i)
                    break

        # Partial match (FP) and FN
        for i, gt_word in enumerate(gt_words):
            if i in used_gt:
                continue
            best_sim = 0.0
            best_idx = -1
            for j, pred_word in enumerate(pred_words):
                if j in used_pred:
                    continue
                sim = SequenceMatcher(None, gt_word, pred_word).ratio()
                if sim > best_sim:
                    best_sim = sim
                    best_idx = j
            if best_idx != -1:
                if best_sim >= 0.5:
                    total_fp += 1
                    used_pred.add(best_idx)
                    used_gt.add(i)
                else:
                    total_fn += 1  # GT word has no sufficiently similar prediction
            else:
                total_fn += 1      # No matching prediction at all

    assert (total_tp + total_fp + total_fn) == total_gt_words, (
        "Sum of TP, FP, FN does not equal the number of GT words")
    return total_tp, total_fp, total_fn, total_gt_words


def calculate_ned(ground_truth_labels, predicted_labels):
    """Compute Normalized Edit Distance (NED)."""
    neds = []
    for gt_label, pred_label in zip(ground_truth_labels, predicted_labels):
        ed = edit_distance(pred_label.strip(), gt_label.strip())
        max_len = max(len(pred_label), len(gt_label))
        ned = ed / max_len if max_len > 0 else 0
        neds.append(ned)
    return sum(neds) / len(neds) * 100 if neds else 0


def calculate_image_metrics(ground_truth_labels, predicted_labels,
                            total_tp, total_fp, total_fn, total_gt_words):
    """Return precision, recall, F1, accuracy, and NED."""
    precision = (total_tp / (total_tp + total_fp)) * 100 if total_tp + total_fp else 0
    recall = (total_tp / (total_tp + total_fn)) * 100 if total_tp + total_fn else 0
    f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) else 0
    accuracy = (total_tp / total_gt_words) * 100 if total_gt_words else 0
    ned = calculate_ned(ground_truth_labels, predicted_labels)
    return {
        "precision": precision,
        "recall": recall,
        "f1_score": f1_score,
        "accuracy": accuracy,
        "true_positive": total_tp,
        "false_positives": total_fp,
        "false_negatives": total_fn,
        "ned": ned,
    }


def parse_input(file_path):
    """Parse a result text file containing image, prediction, and ground truth."""
    images, gts, preds = [], [], []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    i = 0
    while i < len(lines):
        if lines[i].startswith("Image:"):
            img = lines[i].split(": ", 1)[1].strip() if ": " in lines[i] else "unknown"
            pred_label = ""
            if i + 1 < len(lines) and lines[i + 1].startswith("Predicted label:"):
                pred_label = lines[i + 1].split(": ", 1)[1].strip()
            gt_label = ""
            if i + 2 < len(lines) and lines[i + 2].startswith("Ground truth label:"):
                gt_label = lines[i + 2].split(": ", 1)[1].strip()
            images.append(img)
            preds.append(pred_label)
            gts.append(gt_label)
            i += 3
        else:
            i += 1
    return gts, preds


def perform_OCR(input_folder):
    """Compute OCR metrics for every .txt file in a folder."""
    all_results = []
    for fname in os.listdir(input_folder):
        file_path = os.path.join(input_folder, fname)
        if not os.path.isfile(file_path) or not fname.endswith('.txt'):
            continue
        gts, preds = parse_input(file_path)
        all_results.append({'file': fname, 'ground_truth': gts, 'predicted': preds})

    metrics_results = []
    for result in all_results:
        total_tp, total_fp, total_fn, total_gt_words = calculate_pos_neg(
            result['ground_truth'], result['predicted'])
        metrics = calculate_image_metrics(
            ground_truth_labels=result['ground_truth'],
            predicted_labels=result['predicted'],
            total_tp=total_tp,
            total_fp=total_fp,
            total_fn=total_fn,
            total_gt_words=total_gt_words)
        metrics['file'] = result['file']
        metrics_results.append(metrics)
    return metrics_results


# Quality metrics -------------------------------------------------------------
from MANIQA.predict_many_image import MANIQA_SCORE
from CLIP_IQA.demo.clipiqa_custom import CLIP_IQA_SCORE


def six_guys(original, resolution, save_path=None):
    """Compute SR metrics and attach OCR scores for each result file."""
    ori, res = type_navi(original, resolution)
    hq, lq = pil2tensor(ori, res)
    _, np_lq = image_preprocessing(ori, res)

    p = psnr(hq, lq)
    s = ssim(hq, lq)
    l = mlpips(hq, lq)

    maniqa = MANIQA_SCORE(np_lq)
    musiqa = MUSIQ_SCORE(resolution=res)
    clipiqa = CLIP_IQA_SCORE(np_lq)[0]

    metrics = {
        "PSNR": float(p),
        "SSIM": float(s),
        "LPIPS": float(l),
        "MANIQA": float(maniqa),
        "MUSIQA": float(musiqa),
        "CLIP_IQA": float(clipiqa),
    }

    # Append OCR metrics for each text file
    for entry in perform_OCR(resolution):
        metrics[entry['file']] = {
            "precision": entry['precision'],
            "recall": entry['recall'],
            "f1_score": entry['f1_score'],
            "accuracy": entry['accuracy'],
            "ned": entry['ned'],
        }

    # Optionally save results
    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(f"{save_path}.json", "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=4)
        print("Saving to:", save_path)

    # Pretty‑print
    for key, value in metrics.items():
        print(f"{key} : {value}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ori_dir', required=True,
                        help='Directory with original images')
    parser.add_argument('--sr_dir', required=True,
                        help='Directory with super‑resolved images')
    parser.add_argument('--res_path', default=None,
                        help='Save path (without extension) for JSON')
    args = parser.parse_args()
    six_guys(args.ori_dir, args.sr_dir, args.res_path)


if __name__ == "__main__":
    main()
