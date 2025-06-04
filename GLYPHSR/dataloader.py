import json
from pathlib import Path
from PIL import Image, ImageFilter
import os
import re
from datasets import Dataset, DatasetDict
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

# ─── Define Prompt ─────────────────────────────────────────
HQ_PROMPT = (
    "Cinematic, High Contrast, highly detailed, taken using a Canon EOS R camera, "
    "hyper detailed photo-realistic maximum detail, 32k, Color Grading, ultra HD, "
    "extreme meticulous detailing, skin pore detailing, hyper sharpness, perfect without deformations"
)
LQ_PROMPT = (
    "painting, oil painting, illustration, drawing, art, sketch, cartoon, CG Style, "
    "3D render, unreal engine, blurring, dirty, messy, worst quality, low quality, frames, "
    "watermark, signature, jpeg artifacts, deformed, lowres, over-smooth"
)
POSITIVE_PROMPT = (
    "The scene text in the image is rendered in crystal-clear clarity with razor-sharp edges, "
    "every letter faithfully preserved in high-resolution detail, seamless spacing, "
    "and no distortion or noise"
)

NEGATIVE_PROMPT = (
    "The scene text appears blurred, smeared or partially broken;"
    "misaligned text, shifted text, incorrect position, distorted glyphs, random letters."
)

class HQLQDataset(Dataset):
    def __init__(
        self,
        hq_meta_path: str,
        neg_meta_path: str,
        img_size: int = 512,
        include_types=("P_HQ","N_HQ","P_LQ","N_LQ")
    ):
        super().__init__()
        self.img_size = img_size
        self.include_types = set(include_types)

        self.to_tensor = transforms.Compose([
            transforms.Resize(img_size, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.ToTensor(),
             transforms.Normalize(
                mean=[0.5, 0.5, 0.5],
                std =[0.5, 0.5, 0.5]
            ), 
        ])

        self.degrade = transforms.Compose([
            transforms.Resize(img_size // 4, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.Resize(img_size,   interpolation=transforms.InterpolationMode.BICUBIC),
        ])
        self.blur = lambda img: img.filter(ImageFilter.GaussianBlur(radius=2))
        
        #self.degrade_factors   = [10,11,12,13,14,15]
        self.degrade_factors   = [4,5,6,7,8,9]
        self.blur_radius_range = (0.5, 2.5)

        self.hq_base = Path(hq_meta_path).parent
        self.rows = []
        
        unwanted = [
            "realistic photograph",
            "35 mm film style",
            "soft natural lighting",
            "overcast lighting"
        ]
        def clean_meta(prompt: str) -> str:
            parts = [p.strip() for p in prompt.rstrip('.').split(',')]
            parts = [p for p in parts if not any(u in p for u in unwanted)]
            return ", ".join(parts)

        # 1) HQ meta → P_HQ, N_HQ
        with open(hq_meta_path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                meta = clean_meta(rec["prompt"])
                img_path = self.hq_base / rec["HQ_path"]

                if "P_HQ" in self.include_types:
                    self.rows.append({
                        "type": "P_HQ",
                        "path": str(img_path),
                        "txt": f"{meta}.{POSITIVE_PROMPT}. {HQ_PROMPT}.",
                        "SR_OCR" : "",
                        "OCR" : ""
                    })
                if "P_LQ" in self.include_types:
                    self.rows.append({
                        "type": "P_LQ",
                        "path": str(img_path),
                        "txt": f"{meta}. {POSITIVE_PROMPT}. {LQ_PROMPT}.",
                        "SR_OCR" : "",
                        "OCR" : ""
                    })

        with open(neg_meta_path, encoding='utf-8') as f:
            for line in f:
                rec = json.loads(line)
                meta = clean_meta(rec["prompt"])

                # 1) if OCR is “[TEXT]” ---> skip
                ocr_text = rec.get("OCR", "").strip()
                if ocr_text == "[TEXT]":
                    continue

                # 2) Skip if there is even one OCR word that matches exactly in SR_OCR
                sr_ocr_text = rec.get("SR_OCR", "")
                ocr_words = re.findall(r"\w+", ocr_text)
                sr_words  = re.findall(r"\w+", sr_ocr_text)
                if any(word in sr_words for word in ocr_words):
                    continue

                img_path = self.hq_base / rec["neg_path"]

                if "N_HQ" in self.include_types:
                    self.rows.append({
                        "type": "N_HQ",
                        "path": str(img_path),
                        "txt": f"{meta}.{NEGATIVE_PROMPT}. {HQ_PROMPT}.",
                        "SR_OCR" : rec["SR_OCR"],
                        "OCR" : rec["OCR"]
                    })
                if "N_LQ" in self.include_types:
                    self.rows.append({
                        "type": "N_LQ",
                        "path": str(img_path),
                        "txt": f"{meta}. {NEGATIVE_PROMPT}. {LQ_PROMPT}.",
                        "SR_OCR" : rec["SR_OCR"],
                        "OCR" : rec["OCR"]
                    })

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        rec = self.rows[idx]
        img = Image.open(rec["path"]).convert("RGB")

        if rec["type"].endswith("_LQ"):
 
            factor = random.choice(self.degrade_factors)
            small  = img.resize((self.img_size//factor, self.img_size//factor), Image.BICUBIC)
            img    = small.resize((self.img_size, self.img_size), Image.BICUBIC)

            radius = random.uniform(*self.blur_radius_range)
            img    = img.filter(ImageFilter.GaussianBlur(radius=radius))
            
        tensor = self.to_tensor(img)
        return {
            "input_tensor": tensor,
            "txt": rec["txt"],
            "type": rec["type"],
            "SR_OCR": rec["SR_OCR"],
            "OCR": rec["OCR"],
        }

def make_hqlq_dataloader(
    hq_meta_path: str,
    neg_meta_path: str,
    batch_size: int = 8,
    img_size: int = 512,
    include_types=("P_HQ","P_LQ","N_HQ","N_LQ"),
    shuffle: bool = True,
    num_workers: int = 4
):
    ds = HQLQDataset(
        hq_meta_path=hq_meta_path,
        neg_meta_path=neg_meta_path,
        img_size=img_size,
        include_types=include_types
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        pin_memory=True,
        num_workers=num_workers,
        collate_fn=lambda batch: {
            "input_tensor": torch.stack([b["input_tensor"] for b in batch]),
            "txt": [b["txt"] for b in batch],
            "type": [b["type"] for b in batch],
            "OCR":         [b["OCR"] for b in batch],
            "SR_OCR":      [b["SR_OCR"] for b in batch],
        }
    )
    
####VLM FT####
def parse_icdar2017(gt_dir: str, img_dir: str):
    """
    ICDAR2017 데이터셋의 이미지와 gt 파일을 파싱하는 함수

    Args:
      gt_dir (str): ground truth txt 파일들이 위치한 디렉토리 (예: "/NasData/datasets/ICDAR2017/train/gt")
      img_dir (str): 이미지 파일들이 위치한 디렉토리 (예: "/NasData/datasets/ICDAR2017/train/img")
    
    Returns:
      List[dict]: 각 샘플은 다음과 같은 형태의 딕셔너리
         {
             "image_path": <전체 이미지 경로>,
             "ocr_text": <인식된 텍스트(개행으로 연결)>,
             ...
         }
         (필요하다면 bbox, language도 추가 가능)
    """
    samples = []
    
    for file in os.listdir(img_dir):
        if file.lower().endswith(".jpg"):
            img_path = os.path.join(img_dir, file)
            base_name = os.path.splitext(file)[0]
            gt_file = "gt_" + base_name + ".txt"
            gt_path = os.path.join(gt_dir, gt_file)

            recognized_lines = []  # A list to store the text extracted from each line

            if os.path.exists(gt_path):
                with open(gt_path, 'r', encoding='utf-8') as f:
                    # Read and process line by line
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue  # Ignore blank lines
                        
                        tokens = [token.strip() for token in line.split(',')]
                        if len(tokens) < 10:
                            # need at least 10 items, including 8 bounding boxes, language, and text.
                            print(f"Warning: gt file {gt_path} line doesn't have enough tokens: {line}")
                            continue
                        
                        # tokens[:8] => bbox coordinate, tokens[8] => language, tokens[9] => recognized_text
                        recognized_text_line = tokens[9]
                        
                        # Treat placeholders such as `###` as empty strings because they contain no actual text.
                        if recognized_text_line == '###':
                            recognized_text_line = ''
                        
                        recognized_lines.append(recognized_text_line)

            # Concatenate text read from multiple lines with newline characters
            full_text = "\n".join(recognized_lines)
            # Replace two or more consecutive line breaks (\n) with a single line break
            full_text = re.sub(r'\n+', '\n', full_text)
            # Remove unnecessary \n from front and back
            full_text = full_text.strip()
            
            sample = {
                "image_path": img_path,
                "ocr_text": full_text
            }
            samples.append(sample)
            #print(samples)
            #input()
    return samples

def make_hf_dataset(train_data, val_data=None):
    """
    Creates a Hugging Face `Dataset` object.

    Args:
        train_data (List[dict]): List of training samples (each item is the output of `parse_icdar2017`)
        val_data (List[dict], optional): List of validation samples

    Returns:
        DatasetDict: {"train": train_dataset, "validation": val_dataset} (included only if `val_data` is provided)
    """
    train_dataset = Dataset.from_list(train_data)
    if val_data is not None:
        val_dataset = Dataset.from_list(val_data)
        dset = DatasetDict({"train": train_dataset, "validation": val_dataset})
    else:
        dset = DatasetDict({"train": train_dataset})
    return dset
