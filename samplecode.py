import math

import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as trans_fn
from torchvision.utils import make_grid

import data as Data
import data.single_img as Single_Img
import SR3.config.sr3 as SR3
import SR3.model as sr3_model
import SR3.utill.logger as Logger
import SR3.utill.tensor2img as T2I

if __name__ == "__main__":
    sr3_args = SR3.SR3_Config()
    sr3_opt = Logger.parse(sr3_args)
    diffusion = sr3_model.create_model(sr3_opt)
    diffusion.set_new_noise_schedule(
        sr3_opt['model']['beta_schedule']['val'], schedule_phase='val')

    loader = Single_Img.single_image_dataloader(
        '/home/delta1/GMK/raw_data/WHU-RS19_28_224/lr_28/port_53.png', 8)

    for val_data in loader:
        diffusion.feed_data(val_data)

    diffusion.test(continous=True)
    sr_tensor = diffusion.SR
    if sr_tensor.dim() == 4:
        sr_tensor = sr_tensor[-1]

    sr_img_np = T2I.tensor2img(sr_tensor, min_max=(-1, 1))
    sr_pil = Image.fromarray(sr_img_np)
    sr_pil.save("results/samplecode/single_sr_result_tensor2img_6.png")

    width, height = sr_pil.size
    image_sizes = [sr_pil.size]

    print(width, height,image_sizes)
