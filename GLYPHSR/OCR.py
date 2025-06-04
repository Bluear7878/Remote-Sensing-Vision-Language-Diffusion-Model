import copy
import re
import string
import torch
from llava.mm_utils import tokenizer_image_token
from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN

def generate_ocr_text(ocr_prompt, image_tensor, image_sizes, model, tokenizer, conv_templates, conv_template, IMAGE_TOKEN_INDEX, device, max_new_tokens=256):

    image_tensor = [_image.to(dtype=torch.float16, device=device) for _image in image_tensor]

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], ocr_prompt)
    conv.append_message(conv.roles[1], None)

    prompt_question = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt_question, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).to(device)

    with torch.inference_mode():
        outputs = model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,
            do_sample=False,
            temperature=0,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=True
        )
    pred_text = tokenizer.decode(outputs[0][0].cpu().tolist(), skip_special_tokens=True)
    pred_text = pred_text.lstrip("\n")
    return pred_text,outputs

def get_text_position(
    image_tensor,
    pred_text,            # OCRed text
    image,                # original image (used only for size information)
    model,                # LLaVA model
    tokenizer,            # tokenizer
    location_prompt,
    conv_templates,       # conversation template dictionary
    default_image_token,  # e.g., DEFAULT_IMAGE_TOKEN
    image_token_index,    # e.g., IMAGE_TOKEN_INDEX
    conv_template="llava_llama_3",
    max_new_tokens=256,
    device="cuda"
):
    if not pred_text or not pred_text.strip():
        num_texts = 0
        unique_lines = []
    else:
        raw_lines = [line.strip() for line in pred_text.split('\n') if line.strip()]
        
        seen_lines = set()
        unique_lines = []
        for line in raw_lines:
            seen_words = set()
            deduped_words = []
            for w in line.split():
                w_clean = w.strip(string.punctuation)
                if not w_clean:
                    continue
                key = w_clean.lower()
                if key not in seen_words:
                    seen_words.add(key)
                    deduped_words.append(w_clean)

            cleaned_line = " ".join(deduped_words)
            key_line = cleaned_line.lower()
            if key_line and key_line not in seen_lines:
                seen_lines.add(key_line)
                unique_lines.append(cleaned_line)

        num_texts = len(unique_lines)
        if num_texts == 0:
            return pred_text

    if num_texts == 0:
        return pred_text

    text_format_parts = []
    Text_len = "There are a total of {} words in the image.".format(num_texts)
    text_format_parts.append(Text_len)

    
    for line in unique_lines:
        
        conv = copy.deepcopy(conv_templates[conv_template])
        conv.tokenizer = tokenizer
        
        question = location_prompt.format(DEFAULT_IMAGE_TOKEN=DEFAULT_IMAGE_TOKEN, pred_text=line)
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()
        
        input_ids = tokenizer_image_token(
            prompt_question,
            tokenizer,
            IMAGE_TOKEN_INDEX,
            return_tensors="pt"
        ).unsqueeze(0).to(device)
        
        image_sizes = [image.size]


        with torch.inference_mode():
            outputs = model.generate(
                input_ids,
                images=image_tensor,
                modalities=["image"] * len(image_tensor),
                image_sizes=image_sizes,
                do_sample=False,
                temperature=0,
                max_new_tokens=max_new_tokens,
                return_dict_in_generate=True,
                output_scores=True
            )
        generated_output = outputs[0][0].cpu().tolist()
        raw_text = tokenizer.decode(generated_output, skip_special_tokens=True).strip()
        text_format_parts.append(raw_text)
   
    text_format_joined = " ".join(text_format_parts)

    return text_format_joined

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


def get_train_prompt(
    image_tensor,
    image,
    model, 
    tokenizer,
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
    pred_texts = ranked_candidates.splitlines()

    dedup = list(dict.fromkeys(pred_texts))
    ranked_candidates = ", ".join(f"[{word}]" for word in dedup)
    
    
    if "[TEXT]" in ranked_candidates:
        return None

    

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


def get_train_prompt(
    image_tensor,
    image,
    model, 
    tokenizer,
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
    pred_texts = ranked_candidates.splitlines()

    dedup = list(dict.fromkeys(pred_texts))
    ranked_candidates = ", ".join(f"[{word}]" for word in dedup)
    
    
    if "[TEXT]" in ranked_candidates:
        return None

    

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



def combine_prompts(
    image_tensor,
    image,
    model,
    tokenizer,
    text_prompt: str,     # prompt_out
    img_caption: str,     # image_caption
    conv_templates,
    image_token_index: int,
    conv_template: str = "llava_llama_3",
    num_beams: int = 1,
    temperature: float = 0.3,
    do_sample: bool = True,
    max_new_tokens: int = 128,
    device: str = "cuda"
):
    question = (
        f"{DEFAULT_IMAGE_TOKEN}\n"
        "Below are two candidate sentences:\n"
        f"(A) {text_prompt}\n"
        f"(B) {img_caption}\n\n"
        "Merge them into **fluent English sentences** that:\n"
        "• Keeps every piece of scene text exactly as is (inside quotation marks);\n"
        "• Describes precisely where that text appears and its surrounding context;\n"
        "• Removes any redundant or conflicting phrases.\n"
        "Output nothing except the merged sentence."
    )

    conv = copy.deepcopy(conv_templates[conv_template])
    conv.tokenizer = tokenizer
    conv.append_message(conv.roles[0], question)
    conv.append_message(conv.roles[1], None)
    prompt_merge = conv.get_prompt()

    input_ids = tokenizer_image_token(
        prompt_merge,
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
            num_beams=num_beams,
            max_new_tokens=max_new_tokens,
            return_dict_in_generate=True,
            output_scores=False,
        )

    merged_ids = outputs.sequences[0].cpu().tolist()
    merged = tokenizer.decode(merged_ids, skip_special_tokens=True).strip()
    merged = " ".join(merged.split())

    return merged