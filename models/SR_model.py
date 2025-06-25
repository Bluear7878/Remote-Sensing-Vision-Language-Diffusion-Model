import copy
import os
import random
from datetime import datetime

import lpips
import torch
import torchmetrics.functional as TMF
import torchvision.transforms.functional as TF
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate

from models.modules.DFBCache import *
from models.util import PIL2Tensor, Tensor2PIL, convert_dtype, degrade_image
from sgm.models.diffusion import DiffusionEngine
from sgm.modules.distributions.distributions import \
    DiagonalGaussianDistribution
from sgm.util import instantiate_from_config
from utils.colorfix import (adaptive_instance_normalization,
                            wavelet_reconstruction)
from utils.tilevae import VAEHook


class SR_backbone(DiffusionEngine):
    def __init__(self, control_stage_config, ae_dtype='fp32', diffusion_dtype='fp32', p_p='', n_p='', *args, **kwargs):
        super().__init__(*args, **kwargs)
        control_model = instantiate_from_config(control_stage_config)
        self.model.load_control_model(control_model)
        self.first_stage_model.denoise_encoder = copy.deepcopy(self.first_stage_model.encoder)
        self.sampler_config = kwargs['sampler_config']

        assert (ae_dtype in ['fp32', 'fp16', 'bf16']) and (diffusion_dtype in ['fp32', 'fp16', 'bf16'])
        if ae_dtype == 'fp32':
            ae_dtype = torch.float32
        elif ae_dtype == 'fp16':
            raise RuntimeError('fp16 cause NaN in AE')
        elif ae_dtype == 'bf16':
            ae_dtype = torch.bfloat16

        if diffusion_dtype == 'fp32':
            diffusion_dtype = torch.float32
        elif diffusion_dtype == 'fp16':
            diffusion_dtype = torch.float16
        elif diffusion_dtype == 'bf16':
            diffusion_dtype = torch.bfloat16

        self.ae_dtype = ae_dtype
        self.model.dtype = diffusion_dtype

        self.p_p = p_p
        self.n_p = n_p

        self._lpips = None
        self.set_lpips()
        self.val_sample_kwargs  = None

        self.upscale = 1
        self.min_size = 256

    def set_lpips(self):
        if self._lpips is None:
            self._lpips = lpips.LPIPS(net="vgg").eval().to(self.device)

    @torch.no_grad()
    def encode_first_stage(self, x):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            z = self.first_stage_model.encode(x)
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def encode_first_stage_with_denoise(self, x, use_sample=True, is_stage1=False):
        with torch.autocast("cuda", dtype=self.ae_dtype):
            if is_stage1:
                h = self.first_stage_model.denoise_encoder_s1(x)
            else:
                h = self.first_stage_model.denoise_encoder(x)
            moments = self.first_stage_model.quant_conv(h)
            posterior = DiagonalGaussianDistribution(moments)
            if use_sample:
                z = posterior.sample()
            else:
                z = posterior.mode()
        z = self.scale_factor * z
        return z

    @torch.no_grad()
    def decode_first_stage(self, z):
        z = 1.0 / self.scale_factor * z
        with torch.autocast("cuda", dtype=self.ae_dtype):
            out = self.first_stage_model.decode(z)
        return out.float()

    @torch.no_grad()
    def batchify_denoise(self, x, is_stage1=False):
        '''
        [N, C, H, W], [-1, 1], RGB
        '''
        x = self.encode_first_stage_with_denoise(x, use_sample=False, is_stage1=is_stage1)
        return self.decode_first_stage(x)


    def init_tile_vae(self, encoder_tile_size=512, decoder_tile_size=64):
        self.first_stage_model.denoise_encoder.original_forward = self.first_stage_model.denoise_encoder.forward
        self.first_stage_model.encoder.original_forward = self.first_stage_model.encoder.forward
        self.first_stage_model.decoder.original_forward = self.first_stage_model.decoder.forward
        self.first_stage_model.denoise_encoder.forward = VAEHook(
            self.first_stage_model.denoise_encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.encoder.forward = VAEHook(
            self.first_stage_model.encoder, encoder_tile_size, is_decoder=False, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)
        self.first_stage_model.decoder.forward = VAEHook(
            self.first_stage_model.decoder, decoder_tile_size, is_decoder=True, fast_decoder=False,
            fast_encoder=False, color_fix=False, to_gpu=True)

    def prepare_condition(self, _z, p, p_p, n_p, N):
        batch = {}
        batch['original_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['crop_coords_top_left'] = torch.tensor([0, 0]).repeat(N, 1).to(_z.device)
        batch['target_size_as_tuple'] = torch.tensor([1024, 1024]).repeat(N, 1).to(_z.device)
        batch['aesthetic_score'] = torch.tensor([9.0]).repeat(N, 1).to(_z.device)
        batch['control'] = _z

        batch_uc = copy.deepcopy(batch)
        batch_uc['txt'] = [n_p for _ in p]

        if not isinstance(p[0], list):
            batch['txt'] = [' '.join([_p,p_p]) for _p in p]

            with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
        else:
            assert len(p) == 1, 'Support bs=1 only for local prompt conditioning.'
            p_tiles = p[0]
            c = []
            for i, p_tile in enumerate(p_tiles):
                batch['txt'] = [' '.join([p_tile, p_p])]

                with torch.cuda.amp.autocast(dtype=self.ae_dtype):
                    if i == 0:
                        _c, uc = self.conditioner.get_unconditional_conditioning(batch, batch_uc)
                    else:
                        _c, _ = self.conditioner.get_unconditional_conditioning(batch, None)
                c.append(_c)
        return c, uc


    @torch.no_grad()
    def calc_metrics(self, sr, hr):
        if sr.shape[-2:] != hr.shape[-2:]:
            sr = TF.resize(sr, hr.shape[-2:], antialias=True)

        sr01 = (sr + 1) / 2
        hr01 = (hr + 1) / 2

        psnr  = TMF.peak_signal_noise_ratio(sr01, hr01, data_range=1.0)
        ssim  = TMF.structural_similarity_index_measure(sr01, hr01, data_range=1.0)
        lpips = self._lpips(sr, hr).mean()
        return {"PSNR": psnr, "SSIM": ssim, "LPIPS": lpips}

    @torch.no_grad()
    def validation_step(self, batch, batch_idx):

        hr_tensor   = batch["input_tensor"]
        txt = batch["txt"]

        pil_hr = Tensor2PIL(hr_tensor.squeeze(0), hr_tensor.shape[-2], hr_tensor.shape[-1])
        pil_lq = degrade_image(pil_hr, down_factor=10)
        lq_tensor, h0, w0 = PIL2Tensor(pil_lq, upscale=self.upscale, min_size=self.min_size)
        lq_tensor = lq_tensor.unsqueeze(0)[:, :3, :, :].to(self.device)  # [1,3,H,W]

        sr_tensors = self.just_sampling(
            x=lq_tensor,
            p=txt,
            **self.val_sample_kwargs
        )

        metrics = self.calc_metrics(sr_tensors[0].unsqueeze(0), hr_tensor.to(self.device))
        self.log_dict(
            {
                "val/psnr":  metrics["PSNR"],
                "val/ssim":  metrics["SSIM"],
                "val/lpips": metrics["LPIPS"],
            },
            prog_bar=True, logger=True,
            on_step=False, on_epoch=True, sync_dist=True
        )

        return metrics["PSNR"]

    @torch.no_grad()
    def just_sampling(self, x, p, p_p='default', n_p='default',img_threshold=0.1, dec_img=1.0, num_steps=100, restoration_scale=4.0, s_churn=0, s_noise=1.003, cfg_scale=4.0, seed=-1,
                        num_samples=1, control_scale=1, color_fix_type='None', use_linear_CFG=False, use_linear_control_scale=False,
                        cfg_scale_start=1.0, control_scale_start=0.0, **kwargs):
        '''
        [N, C], [-1, 1], RGB
        '''
        assert len(x) == len(p)
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        N = len(x)
        if num_samples > 1:
            assert N == 1
            N = num_samples
            x = x.repeat(N, 1, 1, 1)
            p = p * N

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p

        self.sampler_config.params.num_steps = num_steps
        if use_linear_CFG:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale_start
        else:
            self.sampler_config.params.guider_config.params.scale_min = cfg_scale
            self.sampler_config.params.guider_config.params.scale = cfg_scale
        self.sampler_config.params.restore_cfg = restoration_scale
        self.sampler_config.params.s_churn = s_churn
        self.sampler_config.params.s_noise = s_noise
        self.sampler = instantiate_from_config(self.sampler_config)

        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)

        c_img, uc_img = self.prepare_condition(_z, p, p_p, n_p, N)

        image_cache = MyCacheContext()

        denoiser = lambda inp, sigma, c, *args, **kwargs: self.denoiser(
        self.model, inp, sigma, c, *args, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)

        z, s_in, sigmas, num_sigmas, c_img, uc_img = self.sampler.init_loop(
            noised_z, c_img, uc=uc_img, num_steps=num_steps
        )

        x_center_cur = z_stage1
        step_count = num_sigmas - 1

        with cache_context(image_cache):
            for i in range(step_count):
                z,img_threshold = self.sampler.step(
                    z, i, s_in, sigmas, denoiser, c_img, uc_img,
                    x_center=x_center_cur,
                    control_scale=control_scale,
                    use_linear_control_scale=use_linear_control_scale,
                    control_scale_start=control_scale_start,
                    threshold = img_threshold
                )
                x_center_cur = z
                img_threshold = img_threshold*dec_img

        samples = self.decode_first_stage(z)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)
        return samples
