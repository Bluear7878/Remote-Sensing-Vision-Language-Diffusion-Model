import torch
from sgm.models.diffusion import DiffusionEngine
from sgm.util import instantiate_from_config
import copy
from sgm.modules.distributions.distributions import DiagonalGaussianDistribution
import random
from pytorch_lightning import seed_everything
from torch.nn.functional import interpolate
from utils.colorfix import wavelet_reconstruction, adaptive_instance_normalization
from utils.tilevae import VAEHook
from GLYPHSR.util import PIL2Tensor, Tensor2PIL, convert_dtype, degrade_image
import os
import random
from datetime import datetime
import lpips

import torchvision.transforms.functional as TF
import torchmetrics.functional as TMF

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
        
        self.upscale = 2
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
    def ping_pong_sampling(
        self,
        x,
        p,          # prompt for image restoration
        p_text,     # prompt for text restoration
        p_p='default',
        n_p='default',
        a_text_prompt="default",
        n_text_prompt="default",
        image_steps=50,     # Stage-1 image steps
        text_steps=20,      # Stage-2 text steps
        restoration_scale=4.0,
        s_churn=0,
        s_noise=1.003,
        cfg_scale=4.0,
        seed=-1,
        num_samples=1,
        control_scale=1,
        color_fix_type='None',
        use_linear_CFG=False,
        use_linear_control_scale=False,
        cfg_scale_start=1.0,
        control_scale_start=0.0,
        **kwargs
    ):
        """
        v2 version:
        - During the early phase (before the switch point), alternately apply an
        image guide on even steps and a text guide on odd steps.
        - During the later phase (after the switch point), apply only the text guide.

        Intermediate and final result images are saved in a timestamp-based directory.
        """
        import os
        import torch
        import random
        from datetime import datetime

        # ----------------------------------------------------------------
        # 2) Preferences and Sampler Configuration
        # ----------------------------------------------------------------
        assert len(x) == len(p), "The number of input images does not match the number of prompts."
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        num_steps = image_steps + text_steps
        N = len(x)

        if num_samples > 1:
            assert N == 1, "num_samples >1 : only support single input"
            N = num_samples
            x = x.repeat(N, 1, 1, 1)
            p = p * N
            p_text = p_text * N

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p

        # Sampler parameter settings
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

        # set seed
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # ----------------------------------------------------------------
        # 3) Initial latent space encoding and conditioning
        # ----------------------------------------------------------------
        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)

        # Condition preparation: Image and text respectively
        c_img, uc_img = self.prepare_condition(_z, p, p_p, n_p, N)
        c_text, uc_text = self.prepare_condition(_z, p_text, a_text_prompt, n_text_prompt, N)

        # denoiser setting
        denoiser_img = lambda inp, sigma, c, *args, **kwargs: self.denoiser(
            self.model, inp, sigma, c, *args, **kwargs
        )
        denoiser_txt = lambda inp, sigma, c, *args, **kwargs: self.denoiser(
            self.model, inp, sigma, c, *args, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)
        z, s_in, sigmas, num_sigmas, c_img, uc_img = self.sampler.init_loop(
            noised_z, c_img, uc=uc_img, num_steps=num_steps
        )

        step_count = num_sigmas - 1

        for i in range(step_count):
            if i == 0:
                x_center_cur = z_stage1
            else:
                x_center_cur = z
            if i % 2 == 0:
                guide_fn, cond, ucond = denoiser_img, c_img, uc_img
            else:
                guide_fn, cond, ucond = denoiser_txt, c_text, uc_text

            z = self.sampler.step(
                z, i, s_in, sigmas, guide_fn, cond, ucond,
                x_center=x_center_cur,
                control_scale=control_scale,
                use_linear_control_scale=use_linear_control_scale,
                control_scale_start=control_scale_start,
            )
        samples = self.decode_first_stage(z)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)

        return samples
    

    
    #################################################################################
    ############################## 강우가 바꿈 #######################################
    #################################################################################
    @torch.no_grad()
    def knagwoo_sampling(
        self,
        x,
        p,          # prompt for image restoration
        p_text,     # prompt for text restoration
        p_p='default',
        n_p='default',
        a_text_prompt="default",
        n_text_prompt="default",
        image_steps=50,     # Stage-1 image steps
        text_steps=20,      # Stage-2 text steps
        restoration_scale=4.0,
        s_churn=0,
        s_noise=1.003,
        cfg_scale=4.0,
        seed=-1,
        num_samples=1,
        control_scale=1,
        color_fix_type='None',
        use_linear_CFG=False,
        use_linear_control_scale=False,
        cfg_scale_start=1.0,
        control_scale_start=0.0,
        image_focus_ratio = 0.5, 
        **kwargs
    ):
        """
        v2 version:
        - During the early phase (before the switch point), alternately apply an
        image guide on even steps and a text guide on odd steps.
        - During the later phase (after the switch point), apply only the text guide.

        Intermediate and final result images are saved in a timestamp-based directory.
        """
        import os
        import torch
        import random
        from datetime import datetime

        # ----------------------------------------------------------------
        # 2) Preferences and Sampler Configuration
        # ----------------------------------------------------------------
        assert len(x) == len(p), "The number of input images does not match the number of prompts."
        assert color_fix_type in ['Wavelet', 'AdaIn', 'None']

        num_steps = image_steps + text_steps
        N = len(x)

        if num_samples > 1:
            assert N == 1, "num_samples >1 : only support single input"
            N = num_samples
            x = x.repeat(N, 1, 1, 1)
            p = p * N
            p_text = p_text * N

        if p_p == 'default':
            p_p = self.p_p
        if n_p == 'default':
            n_p = self.n_p

        # Sampler parameter settings
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

        # set seed
        if seed == -1:
            seed = random.randint(0, 65535)
        seed_everything(seed)

        # ----------------------------------------------------------------
        # 3) Initial latent space encoding and conditioning
        # ----------------------------------------------------------------
        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)

        # Condition preparation: Image and text respectively
        c_img, uc_img = self.prepare_condition(_z, p, p_p, n_p, N)
        c_text, uc_text = self.prepare_condition(_z, p_text, a_text_prompt, n_text_prompt, N)

        # denoiser setting
        denoiser_img = lambda inp, sigma, c, *args, **kwargs: self.denoiser(
            self.model, inp, sigma, c, *args, **kwargs
        )
        denoiser_txt = lambda inp, sigma, c, *args, **kwargs: self.denoiser(
            self.model, inp, sigma, c, *args, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)
        z, s_in, sigmas, num_sigmas, c_img, uc_img = self.sampler.init_loop(
            noised_z, c_img, uc=uc_img, num_steps=num_steps
        )

        step_count = num_sigmas - 1
        phase_switch_step = int(image_focus_ratio * step_count)

        for i in range(step_count):
            if i == 0:
                x_center_cur = z_stage1
            else:
                x_center_cur = z

            # 변경 부분
            if i <= phase_switch_step:
                if i % 2 == 0:
                    guide_fn, cond, ucond = denoiser_img, c_img, uc_img
                else:
                    guide_fn, cond, ucond = denoiser_txt, c_text, uc_text
            else:
                guide_fn, cond, ucond = denoiser_txt, c_text, uc_text

            z = self.sampler.step(
                z, i, s_in, sigmas, guide_fn, cond, ucond,
                x_center=x_center_cur,
                control_scale=control_scale,
                use_linear_control_scale=use_linear_control_scale,
                control_scale_start=control_scale_start,
            )
        samples = self.decode_first_stage(z)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)

        return samples


    @torch.no_grad()
    def mixing_sampling(
        self,
        x,
        p,
        p_text,
        p_p='default',
        n_p='default',
        a_text_prompt="default",
        n_text_prompt="default",
        image_steps=50,
        text_steps=20,
        restoration_scale=4.0,
        s_churn=0,
        s_noise=1.003,
        cfg_scale=4.0,
        seed=-1,
        num_samples=1,
        control_scale=1,
        color_fix_type='None',
        use_linear_CFG=False,
        use_linear_control_scale=False,
        cfg_scale_start=1.0,
        control_scale_start=0.0,
        lambda_t=0.5,
        **kwargs
    ):

        import torch, random, os
        from datetime import datetime

        num_steps = image_steps + text_steps
        assert len(x) == len(p)
        N = len(x)
        if num_samples > 1:
            assert N == 1, "num_samples >1 : only support single input"
            x = x.repeat(num_samples,1,1,1)
            p = p * num_samples; p_text = p_text * num_samples; N = num_samples
        if p_p=='default': p_p=self.p_p
        if n_p=='default': n_p=self.n_p

        self.sampler_config.params.num_steps = num_steps
        scales = (cfg_scale_start if use_linear_CFG else cfg_scale)
        self.sampler_config.params.guider_config.params.scale = scales
        self.sampler_config.params.guider_config.params.scale_min = cfg_scale
        self.sampler_config.params.restore_cfg = restoration_scale
        self.sampler_config.params.s_churn = s_churn
        self.sampler_config.params.s_noise = s_noise
        self.sampler = instantiate_from_config(self.sampler_config)
        if seed<0: seed=random.randint(0,65535)
        seed_everything(seed)

        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z0 = self.encode_first_stage(x_stage1)
        c_img, uc_img = self.prepare_condition(_z,p,p_p,n_p,N)
        c_txt, uc_txt = self.prepare_condition(_z,p_text,a_text_prompt,n_text_prompt,N)

        if isinstance(c_img, dict) and isinstance(c_txt, dict):
            c_mix = {k: lambda_t*c_img[k] + (1-lambda_t)*c_txt[k] for k in c_img}
        else:
            c_mix = lambda_t*c_img + (1-lambda_t)*c_txt
            
        if isinstance(uc_img, dict) and isinstance(uc_txt, dict):
            uc_mix = {k: lambda_t*uc_img[k] + (1-lambda_t)*uc_txt[k] for k in uc_img}
        else:
            uc_mix = lambda_t*uc_img + (1-lambda_t)*uc_txt
            
        denoise = lambda inp, s, c, *a, **k: self.denoiser(self.model, inp, s, c, *a, **k)

        z, s_in, sigmas, num_sigmas, *_ = self.sampler.init_loop(
            torch.randn_like(_z).to(_z.device), c_mix, uc=uc_mix, num_steps=num_steps)
        steps = num_sigmas - 1
        guide_c, guide_uc = c_mix, uc_mix

        for i in range(steps):
            center = z0 if i==0 else z

            z = self.sampler.step(z,i,s_in,sigmas,denoise,guide_c,guide_uc, x_center=center,
                                  control_scale=control_scale,
                                  use_linear_control_scale=use_linear_control_scale,
                                  control_scale_start=control_scale_start)
        samples = self.decode_first_stage(z)
        if color_fix_type=='Wavelet': samples = wavelet_reconstruction(samples,x_stage1)
        elif color_fix_type=='AdaIn': samples = adaptive_instance_normalization(samples,x_stage1)
        return samples
    
    @torch.no_grad()
    def just_sampling(self, x, p, p_p='default', n_p='default', num_steps=100, restoration_scale=4.0, s_churn=0, s_noise=1.003, cfg_scale=4.0, seed=-1,
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

     
        #if seed == -1:
        #    seed = random.randint(0, 65535)
        #seed_everything(seed)


        _z = self.encode_first_stage_with_denoise(x, use_sample=False)
        x_stage1 = self.decode_first_stage(_z)
        z_stage1 = self.encode_first_stage(x_stage1)

        c, uc = self.prepare_condition(_z, p, p_p, n_p, N)

        denoiser = lambda input, sigma, c, control_scale: self.denoiser(
            self.model, input, sigma, c, control_scale, **kwargs
        )

        noised_z = torch.randn_like(_z).to(_z.device)

        _samples = self.sampler(denoiser, noised_z, cond=c, uc=uc, x_center=z_stage1, control_scale=control_scale,
                                use_linear_control_scale=use_linear_control_scale, control_scale_start=control_scale_start)
        samples = self.decode_first_stage(_samples)
        if color_fix_type == 'Wavelet':
            samples = wavelet_reconstruction(samples, x_stage1)
        elif color_fix_type == 'AdaIn':
            samples = adaptive_instance_normalization(samples, x_stage1)
        return samples

    