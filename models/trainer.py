import dataclasses
import inspect
import io
import math
import random
import warnings
from functools import wraps
from typing import Callable, Dict, List, Optional, Tuple, Union

import PIL
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate.state import PartialState
from datasets import Dataset
from datasets.arrow_writer import SchemaInferenceError
from datasets.builder import DatasetGenerationError
from torchvision import transforms
from transformers import (AutoModelForCausalLM, AutoTokenizer, DataCollator,
                          DataCollatorForLanguageModeling, PreTrainedModel,
                          PreTrainedTokenizerBase, Trainer, TrainingArguments)
from transformers.modeling_utils import unwrap_model
from transformers.trainer_callback import TrainerCallback
from transformers.trainer_utils import EvalPrediction

from llava.constants import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                             DEFAULT_IMAGE_TOKEN, IGNORE_INDEX,
                             IMAGE_TOKEN_INDEX)
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (IMAGE_TOKEN_INDEX, get_model_name_from_path,
                            process_images, tokenizer_image_token)
from llava.model.builder import load_pretrained_model

# Optional import for PEFT
try:
    from peft import (PeftConfig, PeftModel, get_peft_model,
                      prepare_model_for_kbit_training)
    _peft_available = True
except ImportError:
    PeftConfig = None
    PeftModel = None
    get_peft_model = None
    prepare_model_for_kbit_training = None
    _peft_available = False
from GLYPHSR.extras.dataset_formatting import get_formatting_func_from_dataset
from legacy.utils import (ConstantLengthDataset,
                          DataCollatorForCompletionOnlyLM,
                          neftune_post_forward_hook,
                          peft_module_casting_to_bf16,
                          trl_sanitze_kwargs_for_tagging)
#from extras.dataset_formatting import get_formatting_func_from_dataset
from SUPER_OCR.import_utils import is_peft_available


def degrade_jpeg(image: PIL.Image.Image, min_quality=30, max_quality=70) -> PIL.Image.Image:
    buffer = io.BytesIO()

    # random quality setting
    q = random.randint(min_quality, max_quality)

    # JPEG encoding
    image.save(buffer, format="JPEG", quality=q)
    buffer.seek(0)

    # Reload PIL image
    degraded = PIL.Image.open(buffer).convert("RGB")
    return degraded



def build_instruction_data(samples):
    """
    samples: [{'image_path': ..., 'ocr_text': ...}, ...]

    Returns list of dict, each containing:
      {
        "image_path": "....",
        "user_prompt": "<Image>\nPlease perform OCR for this image.",
        "assistant_answer": "ORIGINAL PANCAKE HOUSE ...",
      }
    """
    data = []
    for s in samples:
        user_prompt = DEFAULT_IMAGE_TOKEN + "\nPlease perform OCR on this image and output only the recognized text without any additional commentary."
        assistant_answer = s['ocr_text'].strip()
        data.append({
            "image_path": s["image_path"],
            "user_prompt": user_prompt,
            "assistant_answer": assistant_answer
        })
    return data

def build_instruction_data_v2(samples):
    data = []
    for s in samples:
        user_prompt = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            "Answer what text is in the image while satisfying the following conditions.\n"
            "1. Transcribe as much text as you can read.\n"
            "2. If any part is partially unreadable, enclose the partial guess or uncertain text in square brackets. Example: 'hel[lo]' for a partially blurred 'hello'.\n"
            "3. If the entire text is unreadable, provide your best guess in square brackets, such as '[unreadable text]'.\n"
            "4. Do NOT provide any additional commentary or explanation. Return only the transcribed or guessed text.\n"
        )


        assistant_answer = s['ocr_text'].strip()

        data.append({
            "image_path": s["image_path"],
            "user_prompt": user_prompt,
            "assistant_answer": assistant_answer
        })
    return data


def build_instruction_data_v3(samples):
    data = []
    for s in samples:
        user_prompt = (
            f"{DEFAULT_IMAGE_TOKEN}\n"
            "Answer what text is in the image while satisfying the following conditions.\n"
            "1. Transcribe as much text as you can read.\n"
            "2. If any part is partially unreadable, enclose the partial guess or uncertain text in square brackets. Example: 'hel[lo]' for a partially blurred 'hello'.\n"
            "3. If the entire text is unreadable, provide your best guess in square brackets, such as '[unreadable text]'.\n"
            "4. Do NOT provide any additional commentary or explanation. Return only the transcribed or guessed text.\n"
        )

        assistant_answer = s['ocr_text'].strip()

        data.append({
            "image_path": s["image_path"],
            "user_prompt": user_prompt,
            "assistant_answer": assistant_answer
        })
    return data



def preprocess_function(example, tokenizer, model, image_processor, apply_augmentation=False,min_quality=1,max_quality=100):
    train_augmentation = transforms.Compose([
    transforms.Resize(224),
    transforms.RandomApply([
        transforms.Lambda(lambda img: degrade_jpeg(img, min_quality=min_quality, max_quality=max_quality))
    ], p=0.99),
    ])

    try:
        image_path = example["image_path"]
        image = PIL.Image.open(image_path).convert("RGB")

        if apply_augmentation:
            image = train_augmentation(image)

        w, h = image.size
    except Exception as e:
        print("Error processing:", image_path, e)
        return {}


    image_tensor_list = process_images([image], image_processor, model.config)
    pixel_values = image_tensor_list[0]
    pixel_values = pixel_values.to(torch.float16)
    #pixel_values_np = pixel_values.cpu().numpy()

    prompt_text = example["user_prompt"]

    prompt_ids = tokenizer_image_token(
        prompt_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    )

    full_text = prompt_text + "\n" + example["assistant_answer"]
    full_ids = tokenizer_image_token(
        full_text,
        tokenizer,
        IMAGE_TOKEN_INDEX,
        return_tensors="pt"
    )
    labels = full_ids.clone()
    if prompt_ids.dim() == 2:
        prompt_len = prompt_ids.shape[1]
    else:
        prompt_len = prompt_ids.shape[0]

    labels[:prompt_len] = -100

    return {
        "pixel_values": pixel_values,
        "image_size": (h, w),
        "input_ids": full_ids,
        "labels": labels,
        "answers":example["assistant_answer"]
    }



def multimodal_collator(features, tokenizer):
    pixel_values_list = []
    image_sizes = []
    for f in features:
        pixel_values_list.append(torch.tensor(f["pixel_values"], dtype=torch.float16))
        image_sizes.append(f["image_size"])

    batch_pixel_values = torch.stack(pixel_values_list, dim=0)


    input_ids_list = [f["input_ids"] for f in features]
    labels_list = [f["labels"] for f in features]

    max_length = max(len(ids) for ids in input_ids_list)

    padded_input_ids = []
    padded_labels = []
    attention_masks = []

    pad_token_id = getattr(tokenizer, "pad_token_id", 0)
    if pad_token_id is None:
        pad_token_id = 0

    for i in range(len(features)):
        cur_input_ids = input_ids_list[i]
        cur_labels = labels_list[i]
        cur_len = len(cur_input_ids)
        pad_len = max_length - cur_len

        padded_input_ids.append(cur_input_ids + [pad_token_id] * pad_len)

        padded_labels.append(cur_labels + [-100] * pad_len)

        attention_masks.append([1] * cur_len + [0] * pad_len)

    input_ids_tensor = torch.tensor(padded_input_ids, dtype=torch.long)
    labels_tensor = torch.tensor(padded_labels, dtype=torch.long)
    attention_mask_tensor = torch.tensor(attention_masks, dtype=torch.long)


    batch = {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_mask_tensor,
        "labels": labels_tensor,
        "images": batch_pixel_values,
        "image_sizes": image_sizes,
    }
    return batch


class CustomSFTTrainer(Trainer):

    _tag_names = ["trl", "sft"]

    def __init__(
        self,
        model: Union[PreTrainedModel, nn.Module, str] = None,
        args: TrainingArguments = None,
        data_collator: Optional[DataCollator] = None,
        train_dataset: Optional[Dataset] = None,
        eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
        model_init: Optional[Callable[[], PreTrainedModel]] = None,
        compute_metrics: Optional[Callable[[EvalPrediction], Dict]] = None,
        callbacks: Optional[List[TrainerCallback]] = None,
        optimizers: Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR] = (None, None),
        preprocess_logits_for_metrics: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
        peft_config: Optional["PeftConfig"] = None,
        dataset_text_field: Optional[str] = None,
        packing: Optional[bool] = False,
        formatting_func: Optional[Callable] = None,
        max_seq_length: Optional[int] = None,
        infinite: Optional[bool] = None,
        num_of_sequences: Optional[int] = 1024,
        chars_per_token: Optional[float] = 3.6,
        dataset_num_proc: Optional[int] = None,
        dataset_batch_size: int = 1000,
        neftune_noise_alpha: Optional[float] = None,
        model_init_kwargs: Optional[Dict] = None,
        dataset_kwargs: Optional[Dict] = None,
    ):


        self.dataset_num_proc = dataset_num_proc
        self.dataset_batch_size = dataset_batch_size

        # NEFTune readiness check
        self._trainer_supports_neftune = hasattr(args, "neftune_noise_alpha")

        # If user passes neftune_noise_alpha explicitly
        if neftune_noise_alpha is not None and self._trainer_supports_neftune:
            args.neftune_noise_alpha = neftune_noise_alpha
            warnings.warn(
                "Passed neftune_noise_alpha to CustomSFTTrainer; overriding TrainingArguments.neftune_noise_alpha."
            )
        elif neftune_noise_alpha is not None and not self._trainer_supports_neftune:
            # We'll store it locally and handle activation ourselves
            self.neftune_noise_alpha = neftune_noise_alpha

        # Attempt to detect formatting automatically if not provided
        if formatting_func is None and dataset_text_field is None:
            formatting_func = get_formatting_func_from_dataset(train_dataset, tokenizer)



        if tokenizer.padding_side != "right":
            warnings.warn(
                "Tokenizer padding_side is not 'right'; could cause half-precision issues."
            )

        # Init base Trainer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        # Adjust infinite data usage if packing is True
        if self.args.max_steps > 0 and packing:
            warnings.warn(
                "packing=True and max_steps>0 => infinite iteration of dataset until max_steps."
            )
            self.train_dataset.infinite = True
        elif self.args.max_steps == -1 and packing:
            self.train_dataset.infinite = False

    @wraps(Trainer.train)
    def train(self, *args, **kwargs):

        if getattr(self, "neftune_noise_alpha", None) is not None and not self._trainer_supports_neftune:
            self.model = self._trl_activate_neftune(self.model)

        output = super().train(*args, **kwargs)

        if getattr(self, "neftune_noise_alpha", None) is not None and not self._trainer_supports_neftune:
            unwrapped_model = unwrap_model(self.model)
            if is_peft_available() and isinstance(unwrapped_model, PeftModel):
                embeddings = unwrapped_model.base_model.model.get_input_embeddings()
            else:
                embeddings = unwrapped_model.get_input_embeddings()

            self.neftune_hook_handle.remove()
            del embeddings.neftune_noise_alpha

        return output

    '''
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(**inputs)

        # Instead of "labels" being a single ID,
        # assume we created 'soft_labels' that has shape [batch_size, seq_len, vocab_size]
        soft_labels = inputs.get("labels", None)  # each row is a distribution over the vocab

        if soft_labels is None:
            # fallback to the standard cross-entropy if no soft labels
            return (outputs.loss, outputs) if return_outputs else outputs.loss

        # model outputs logits shape: [batch_size, seq_len, vocab_size]
        logits = outputs.logits
        # We apply a soft cross-entropy or KL
        log_probs = torch.log_softmax(logits, dim=-1)  # shape: [batch_size, seq_len, vocab_size]
        loss_per_token = -torch.sum(soft_labels * log_probs, dim=-1)  # shape [batch_size, seq_len]

        # We can ignore padding positions or positions that are -100
        mask = (soft_labels.sum(dim=-1) > 0).float()  # if distribution is all zero => ignore
        final_loss = (loss_per_token * mask).sum() / (mask.sum() + 1e-8)

        return (final_loss, outputs) if return_outputs else final_loss
    '''


    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.get("labels", None)
        outputs = model(**inputs)
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss



    def _trl_activate_neftune(self, model):
        """
        Activates NEFTune by registering a forward hook on the model embeddings.
        """
        unwrapped_model = unwrap_model(model)
        if is_peft_available() and isinstance(unwrapped_model, PeftModel):
            embeddings = unwrapped_model.base_model.model.get_input_embeddings()
        else:
            embeddings = unwrapped_model.get_input_embeddings()

        embeddings.neftune_noise_alpha = self.neftune_noise_alpha
        hook_handle = embeddings.register_forward_hook(neftune_post_forward_hook)
        self.neftune_hook_handle = hook_handle
        return model
