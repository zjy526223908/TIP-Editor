import sys

from transformers import  logging
from diffusers import  DDIMScheduler, DiffusionPipeline
from diffusers.utils.import_utils import is_xformers_available
from diffusers.models.embeddings import TimestepEmbedding
from torch.cuda.amp import custom_bwd, custom_fwd
logging.set_verbosity_error()

import os
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.utils import logging
logging.set_verbosity(40)

from models.sd_pipeline import ClassStableDiffusionImg2ImgPipeline

class SpecifyGradient(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, input_tensor, gt_grad):
        ctx.save_for_backward(gt_grad)
        # we return a dummy value 1, which will be scaled by amp's scaler so we get the scale in backward.
        return torch.ones([1], device=input_tensor.device, dtype=input_tensor.dtype)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_scale):
        gt_grad, = ctx.saved_tensors
        gt_grad = gt_grad * grad_scale
        return gt_grad, None

class ToWeightsDType(nn.Module):
    def __init__(self, module: nn.Module, dtype: torch.dtype):
        super().__init__()
        self.module = module
        self.dtype = dtype

    def forward(self, x):
        return self.module(x).to(self.dtype)

class StableDiffusion(nn.Module):
    def __init__(self, opt, device, sd_path, sd_version='2.0'):
        super().__init__()

        try:
            with open('./TOKEN', 'r') as f:
                self.token = f.read().replace('\n', '') # remove the last \n!
                print(f'[INFO] loaded hugging face access token from ./TOKEN!')
        except FileNotFoundError as e:
            self.token = True
            print(f'[INFO] try to load hugging face access token from the default place, make sure you have run `huggingface-cli login`.')

        self.opt = opt
        self.device = device
        self.sd_version = sd_version

        self.num_train_timesteps = 1000
        self.min_step = int(self.num_train_timesteps * 0.02)
        self.max_step = int(self.num_train_timesteps * self.opt.sd_max_step_start)

        print(f'[INFO] loading stable diffusion...')


        pipe = DiffusionPipeline.from_pretrained(sd_path)

        if os.path.exists(os.path.join(sd_path, 'class_embedding.pth')):
            camera_embedding = ToWeightsDType(
                TimestepEmbedding(16, 1280), dtype=torch.float32
            )
            pipe.unet.class_embedding = camera_embedding
            pipe.unet.class_embedding.load_state_dict(
                torch.load(os.path.join(sd_path, 'class_embedding.pth')))
            self.class_embedding = True
        else:
            self.class_embedding = False

        if os.path.exists(os.path.join(sd_path, 'pytorch_lora_weights.safetensors')):
            pipe.load_lora_weights(sd_path, weight_name="pytorch_lora_weights.safetensors")


        self.vae = pipe.vae.to(self.device)
        self.tokenizer = pipe.tokenizer
        self.text_encoder = pipe.text_encoder.to(self.device)
        self.unet = pipe.unet.to(self.device)


        self.pipeline = ClassStableDiffusionImg2ImgPipeline.from_pretrained(
            sd_path,
            tokenizer=self.tokenizer,
            text_encoder=self.text_encoder,
            unet=self.unet
        )
        self.pipeline.scheduler = DDIMScheduler.from_config(self.pipeline.scheduler.config,
                                                       subfolder="scheduler")

        self.pipeline = self.pipeline.to(self.device)
        self.pipeline.set_progress_bar_config(disable=True)

        if is_xformers_available():
            print('*'*100)
            print('enable xformers')
            try:
                self.unet.enable_xformers_memory_efficient_attention()
            except Exception as e:
                print(
                    "Could not enable memory efficient attention. Make sure xformers is installed"
                    f" correctly and a GPU is available: {e}"
                )

        self.scheduler = DDIMScheduler.from_config(sd_path, subfolder="scheduler")

        self.alphas = self.scheduler.alphas_cumprod.to(self.device)  # for conve

        print(f'[INFO] loaded stable diffusion!')

    def get_text_embeds(self, prompt, negative_prompt):
        # prompt, negative_prompt: [str]

        # Tokenize text and get embeddings
        text_input = self.tokenizer(prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                    truncation=True, return_tensors='pt')

        with torch.no_grad():
            text_embeddings = self.text_encoder(text_input.input_ids.to(self.device))[0]

        # Do the same for unconditional embeddings
        uncond_input = self.tokenizer(negative_prompt, padding='max_length', max_length=self.tokenizer.model_max_length,
                                      return_tensors='pt')

        with torch.no_grad():
            uncond_embeddings = self.text_encoder(uncond_input.input_ids.to(self.device))[0]

        # Cat for final embeddings
        text_embeddings = torch.cat([text_embeddings, uncond_embeddings])
        return text_embeddings

    def train_step(self, text_embeddings, pred_rgb, pose, ratio=0, guidance_scale=100, lamb = 1.0):
        
        # interp to 512x512 to be fed into vae.
        batch_size = pred_rgb.shape[0]
        # _t = time.time()
        if self.opt.sd_img_size == 512:
            pred_rgb = F.interpolate(pred_rgb, (512, 512), mode='bilinear', align_corners=False)
        else:
            pred_rgb = F.interpolate(pred_rgb, (768, 768), mode='bilinear', align_corners=False)

        if ratio > 0:

            max_s = int((self.opt.sd_max_step_end * ratio + self.opt.sd_max_step_start *(1-ratio))  * self.num_train_timesteps)
            t = torch.randint(self.min_step, max_s + 1, [batch_size], dtype=torch.long, device=self.device)
        else:
            t = torch.randint(self.min_step, self.max_step + 1, [batch_size], dtype=torch.long, device=self.device)

        # t[0] = 750
        # encode image into latents with vae, requires grad!
        # _t = time.time()

        latents = self.encode_imgs(pred_rgb)
        # torch.cuda.synchronize(); print(f'[TIME] guiding: vae enc {time.time() - _t:.4f}s')

        # predict the noise residual with unet, NO grad!
        # _t = time.time()

        class_labels = torch.cat(
            [
                pose.view(batch_size, -1),
                torch.zeros_like(pose.view(batch_size, -1)),
            ],
            dim=0,
        )


        with torch.no_grad():
            # add noise

            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            # pred noise
            latent_model_input = torch.cat([latents_noisy] * 2)
            # class_labels = torch.cat([pose.view(batch_size, -1)] * 2)
            if self.class_embedding:
                noise_pred = self.unet(latent_model_input, torch.cat([t] * 2), class_labels=class_labels,
                                       encoder_hidden_states=text_embeddings).sample
            else:
                noise_pred = self.unet(latent_model_input,  torch.cat([t] * 2), encoder_hidden_states=text_embeddings).sample

        # torch.cuda.synchronize(); print(f'[TIME] guiding: unet {time.time() - _t:.4f}s')

        # perform guidance (high scale from paper!)
        noise_pred_text, noise_pred_uncond  = noise_pred.chunk(2)

        noise_pred = noise_pred_text + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # w(t), sigma_t^2
        w = (1 - self.alphas[t]).view(-1, 1, 1, 1)

        # w = self.alphas[t] ** 0.5 * (1 - self.alphas[t])

        grad = w * (noise_pred - noise)
        grad = lamb * grad.clamp(-1, 1)

        grad = torch.nan_to_num(grad)


        target = (latents - grad).detach()
        # d(loss)/d(latents) = latents - target = latents - (latents - grad) = grad
        loss_sds = 0.5 * F.mse_loss(latents.float(), target, reduction="sum") / batch_size

        return loss_sds


    def encode_imgs(self, imgs):
        # imgs: [B, 3, H, W]

        imgs = 2 * imgs - 1

        posterior = self.vae.encode(imgs).latent_dist
        latents = posterior.sample() * 0.18215

        return latents




