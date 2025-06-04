import json
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, IGNORE_INDEX
from llava.mm_utils import process_images, get_model_name_from_path, tokenizer_image_token
import copy
import re
import random
import string
import torch
import os, json, hashlib, gc
from tqdm import tqdm
from GLYPHSR.OCR import *
from utils import *
from GLYPHSR.util import PIL2Tensor,Tensor2PIL
from sgm.util import instantiate_from_config
from omegaconf import OmegaConf
import yaml

class ImageFolderDataset(Dataset):
    def __init__(self, image_dir, exts=("jpg","jpeg","png","bmp")):
        self.image_dir = Path(image_dir)
        self.paths = [p for ext in exts
                         for p in self.image_dir.rglob(f"*.{ext}")]
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, idx):
        path = self.paths[idx]
        img  = Image.open(path).convert("RGB")
        return {
            "image": img,
            "path":  str(path)
        }

def make_image_loader(image_dir, batch_size=1, shuffle=False, num_workers=4):
    ds = ImageFolderDataset(image_dir)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=lambda batch: batch[0],
    )
    

def sanitize_prompt(raw: str, extra_style: str | None = None) -> str:
    m = re.search(r'[\"“”‘’`](.+?)[\"“”‘’`]', raw)
    if m:
        quoted = m.group(0)
        inner  = m.group(1).strip()
        raw    = raw.replace(quoted, f"{inner}", 1)

    raw = raw.rstrip().rstrip(".")

    if extra_style:
        raw = f"{raw}, {extra_style}"

    return raw

def get_synthesis_prompt(
    image_tensor,
    image,
    model, tokenizer,
    ranked_candidates: str,
    conv_templates,
    image_token_index: int,
    conv_template: str = "llava_llama_3",
    num_beams: int = 1,
    temperature: float = 0.7,
    do_sample: bool = True,
    max_new_tokens: int = 64,
    device: str = "cuda"
):
    if "[TEXT]" in ranked_candidates and len(ranked_candidates.strip()) <= len("[TEXT]")+1:
        return None

    ranked_candidates = ranked_candidates.splitlines()[0].strip()

    question = f"""{DEFAULT_IMAGE_TOKEN}
    You will write exactly one fluent English sentence to serve as a text-to-image prompt.
    The image contains the text {ranked_candidates}. Describe naturally how and where that text appears—
    specify the object or surface (e.g. plaque, sign, poster), the font style and color,
    the lighting and perspective, and the surrounding environment—while preserving the text
    exactly as shown within the asterisks."""
        

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_qa = conv.get_prompt()


    input_ids = tokenizer_image_token(
        prompt_qa,
        tokenizer,
        image_token_index,
        return_tensors="pt"
    ).unsqueeze(0).to(device)

    image_sizes = [image.size]  # [(W, H)]
    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=do_sample,
            temperature=temperature,
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False
        )

    gen_ids    = outputs.sequences[0].cpu().tolist()
    raw_prompt = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    prompt_out = "".join(ch for ch in raw_prompt if 32 <= ord(ch) <= 126).strip()

    if not prompt_out:
        return None

    return prompt_out


def generate_prompt_meta_jsonl(
    yaml_file,
    loader,
    image_processor,
    base_model,
    tokenizer,
    conv_templates,
    conv_template,
    IMAGE_TOKEN_INDEX,
    prompt_bank: str = "./prompt_bank",
    guide_prompt: str = ("realistic photograph, 35 mm film style, soft natural lighting."),
    max_samples: int = 10000,
    device: str = "cuda",
):

    prompt_config  = yaml_file
    if not os.path.isfile(prompt_config):
        raise FileNotFoundError(f"YAML file not found: {prompt_config}")
    with open(prompt_config, "r", encoding="utf-8") as f:
        prompts = yaml.safe_load(f)
    ocr_prompt = prompts["ocr_prompt"].format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN)
    os.makedirs(prompt_bank, exist_ok=True)
    meta_path = os.path.join(prompt_bank, "prompts_meta.jsonl")
    saved_cnt = 0

    with open(meta_path, "w") as meta_f:
        for batch in tqdm(loader, total=min(len(loader), max_samples)):
            img_pil = batch["image"]
            # 1) preprocess image
            image_tensor = process_images([img_pil], image_processor, base_model.config)
            image_tensor = [img.to(dtype=torch.float16, device=device) for img in image_tensor]
            image_size = [img_pil.size]

            # 2) OCR
            pred_text, _ = generate_ocr_text(
                ocr_prompt,
                image_tensor,
                image_size,
                base_model,
                tokenizer,
                conv_templates,
                conv_template,
                IMAGE_TOKEN_INDEX,
                device,
                max_new_tokens=77,
            )
            if not pred_text:
                continue

            # 3) generate raw prompt
            raw_prompt = get_synthesis_prompt(
                image_tensor,
                img_pil,
                base_model,
                tokenizer,
                ranked_candidates=pred_text,
                conv_templates=conv_templates,
                image_token_index=IMAGE_TOKEN_INDEX,
            )
            if raw_prompt is None:
                continue

            # 4) sanitize and append guide
            clean = sanitize_prompt(raw_prompt)
            clean = f"{clean}, {guide_prompt}"

            # 5) ensure OCR text is present
            if pred_text not in clean:
                continue

            # 6) wrap OCR text
            wrapped = clean.replace(pred_text, f"**{pred_text}**")

            # 7) write metadata
            stem = hashlib.md5(wrapped.encode("utf-8")).hexdigest()[:16]
            meta_obj = {"id": stem, "OCR": pred_text, "prompt": wrapped}
            meta_f.write(json.dumps(meta_obj, ensure_ascii=False) + "\n")

            saved_cnt += 1
            if saved_cnt >= max_samples:
                break

            # 8) cleanup
            del image_tensor
            gc.collect()
            torch.cuda.empty_cache()

    return meta_path, saved_cnt

##########################################
##########################################
###Make neg_lq datasets
def degrade_image(img_pil, down_factor=1.5):

    w, h = img_pil.size
    small = img_pil.resize(
        (int(w/down_factor), int(h/down_factor)),
        Image.BICUBIC
    )

    img_degraded = small.resize((w, h), Image.BICUBIC)

    return img_degraded

def random_masking_prompt(caption: str, mask_prob: float = 0.5) -> str:

    pool = string.ascii_letters + string.digits

    pattern = r'(["“”‘’`])([^"“”‘’`]+?)\1'
    def _mask_replace(m):
        inner = m.group(2)
        masked = ''.join(
            ch if (ch.isspace() or random.random() > mask_prob) 
               else random.choice(pool)
            for ch in inner
        )
        return masked

    caption = re.sub(pattern, _mask_replace, caption, count=1)

    caption = re.sub(r'[\"“”‘’`]', '', caption)

    return caption.strip().rstrip('.')

def process_record(image_processor,
                   ocr_model,
                   sr_model,
                   ocr_prompt,
                   img_prompt,
                   tokenizer,
                   conv_templates,
                   conv_template,
                   mask_prob,
                    rec: dict,
                   root_dir: str,
                   neg_dir: str,
                   down_factors: list[int],
                   args) -> tuple[bool, dict]:
    img_id = rec["id"]
    hq_rel = rec.get("hq")
    hq_path = os.path.join(root_dir, hq_rel)
    if not os.path.exists(hq_path):
        print(f"⚠️  Skipping missing file: {hq_path}")
        return False, rec

    img_pil = Image.open(hq_path)

    image_tensor = process_images([img_pil], image_processor, ocr_model.config)
    image_tensor = image_tensor.to(ocr_model.device, dtype=torch.float16)
    original_text, _ = generate_ocr_text(
        ocr_prompt,
        image_tensor,
        [img_pil.size],
        ocr_model,
        tokenizer,
        conv_templates,
        conv_template,
        IMAGE_TOKEN_INDEX,
        ocr_model.device,
        max_new_tokens=50
    )

    raw_caption = get_img_describe(
        image_tensor=image_tensor,
        image=img_pil,
        model=ocr_model,
        tokenizer=tokenizer,
        prompt=img_prompt,
        conv_templates=conv_templates,
        image_token_index=IMAGE_TOKEN_INDEX,
        device=ocr_model.device
    )
    raw_caption[0] = random_masking_prompt(raw_caption[0], mask_prob)

    for df in down_factors:
        lq = degrade_image(img_pil, down_factor=df)
        LQ_img, h0, w0 = PIL2Tensor(lq,
                                    upscale=args.upscale,
                                    min_size=args.min_size)
        LQ_img = LQ_img.unsqueeze(0).to(sr_model.device)[:, :3, :, :]

        sr_tensor = getattr(sr_model, "just_sampling")(
            LQ_img,
            raw_caption,
            num_steps=args.edm_steps,
            restoration_scale=args.s_stage1,
            s_churn=args.s_churn,
            s_noise=args.s_noise,
            cfg_scale=args.s_cfg,
            control_scale=args.s_stage2,
            seed=args.seed,
            num_samples=1,
            p_p=args.a_prompt,
            n_p=args.n_prompt,
            color_fix_type=args.color_fix_type,
            use_linear_CFG=args.linear_CFG,
            use_linear_control_scale=args.linear_s_stage2
        )
        sr_pil = Tensor2PIL(sr_tensor[0], h0, w0)
        if len(down_factors)==1:
            neg_path = os.path.join(neg_dir, f"{img_id}_neg.jpg")
            sr_pil.save(neg_path)
            rec.update({
                "neg": neg_path,
                "neg_prompt": raw_caption,
                "OCR": original_text
            })
            return True, rec

        image_tensor = process_images([sr_pil], image_processor, ocr_model.config)
        image_tensor = image_tensor.to(ocr_model.device, dtype=torch.float16)
        sr_text, _ = generate_ocr_text(
            ocr_prompt,
            image_tensor,
            [img_pil.size],
            ocr_model,
            tokenizer,
            conv_templates,
            conv_template,
            IMAGE_TOKEN_INDEX,
            ocr_model.device,
            max_new_tokens=50
        )
    
        if original_text != sr_text:
            neg_path = os.path.join(neg_dir, f"{img_id}_neg.jpg")
            sr_pil.save(neg_path)
            rec.update({
                "neg": neg_path,
                "neg_prompt": raw_caption,
                "OCR": original_text,
                "SR_OCR" : sr_text,
            })
            return True, rec
        else:
            print(f"PASS: {img_id} @ down_factor={df}")

    return False, rec

def load_metadata(meta_path: str):
    with open(meta_path, 'r', encoding='utf-8') as f:
        for line in f:
            yield json.loads(line)

def remove_special_chars_and_merge(text):
    cleaned_text = re.sub(r'[^\w\s]', '', text)
    merged_text = " ".join(cleaned_text.split())
    return merged_text


def build_neg_lq(
    root_dir: str,
    image_processor,
    ocr_model,
    sr_model,
    ocr_prompt: str,
    img_prompt: str,
    tokenizer,
    conv_templates,
    conv_template,
    down_factors: list[int],
    args,
    hq_meta_filename: str = "metadata_HQ.jsonl",
    neg_meta_filename: str = "metadata_with_neg.jsonl",
    mask_prob : float = 0.3,
    world_size: int = 1,
    local_rank: int = 0
):
    # Paths for input and output metadata
    meta_in_path = os.path.join(root_dir, "metadata.jsonl")
    hq_meta_path = os.path.join(root_dir, hq_meta_filename)
    neg_meta_path = os.path.join(root_dir, neg_meta_filename)
    neg_dir = os.path.join(root_dir, "neg_lq")
    os.makedirs(neg_dir, exist_ok=True)

    total, neg_count = 0, 0
    # Open both HQ and neg metadata files for writing
    #print(f"[DBG] cwd: {os.getcwd()}")
    
    
    with open(meta_in_path, "r", encoding="utf-8") as fin, \
         open(hq_meta_path, "w", encoding="utf-8") as hq_f, \
         open(neg_meta_path, "w", encoding="utf-8") as neg_f:
             
        #print(f"[DBG] opened HQ file → {hq_meta_path}")
        #print(f"[DBG] absolute HQ path → {os.path.abspath(hq_meta_path)}")
        #print(f"[DBG] initial exists? → {os.path.exists(hq_meta_path)}")
        #print(f"[DBG] HQ file mode → {hq_f.mode}, closed? {hq_f.closed}")

        for idx, rec in enumerate(load_metadata(meta_in_path)):
            #print(f"[DBG] total records loaded: {len(meta_in_path)}")
            if idx % world_size != local_rank:
                continue
            total += 1

            # Write HQ metadata entry with only required fields
            hq_rel = rec.get("hq", "")
            hq_entry = {
                "id": rec["id"],
                "prompt": rec.get("prompt", ""),
                "HQ_path": os.path.relpath(os.path.join(root_dir, hq_rel), root_dir),
                "OCR": rec.get("OCR", "")
            }
            #print(f"[DEBUG][HQ ENTRY] {json.dumps(hq_entry, ensure_ascii=False)}")
            hq_f.write(json.dumps(hq_entry, ensure_ascii=False) + "\n")
            #hq_f.flush()
            #os.fsync(hq_f.fileno())
            #print(f"[DBG] wrote to {hq_f.name}, tell={hq_f.tell()}, exists now? {os.path.exists(hq_f.name)}")

            # Generate negative sample and write neg metadata if successful
            success, updated = process_record(
                image_processor, ocr_model, sr_model,
                ocr_prompt, img_prompt, tokenizer,
                conv_templates, conv_template, mask_prob,
                rec, root_dir, neg_dir,
                down_factors, args
            )
            #print(f"[DEBUG] success={success}, updated keys={list(updated.keys())}")
            if success:
                neg_entry = {
                    "id": updated["id"],
                    "prompt": updated.get("prompt", ""),
                    "neg_path": os.path.relpath(updated["neg"], root_dir),
                    "OCR": updated.get("OCR", ""),
                    "SR_OCR": updated.get("SR_OCR", "")
                }
                #print(f"[DBG] about to write NEG_entry → {json.dumps(neg_entry, ensure_ascii=False)}")
                neg_f.write(json.dumps(neg_entry, ensure_ascii=False) + "\n")
                #neg_f.flush()
                #os.fsync(neg_f.fileno())
                #print(f"[DBG] wrote to {neg_f.name}, tell={neg_f.tell()}, exists now? {os.path.exists(neg_f.name)}")
                neg_count += 1

            # Cleanup between iterations
            gc.collect()
            torch.cuda.empty_cache()
            

    print(f"[Rank {local_rank}] Processed {total} records, {neg_count} neg samples.")

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


def create_SUPIR_model(config_path, SUPIR_sign=None, load_default_setting=False):
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