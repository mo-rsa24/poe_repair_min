import os
import torch
import torch.nn as nn
import torch.functional as F
import torchvision.transforms as T
import argparse
import importlib
from PIL import Image
from tqdm import tqdm
from transformers import logging
from diffusers import DDIMScheduler, StableDiffusionXLPipeline, UNet2DConditionModel,AutoencoderKL
from diffusers.image_processor import VaeImageProcessor
import gc
from .utils_custom import *
from sentence_transformers.util import (semantic_search, 
                                        dot_score, 
                                        normalize_embeddings)
import inspect
from .utils import ptp_utils
from .utils.ptp_utils import (tokenize_prompt, encode_prompt, get_concept_indices)
import cv2
import numpy as np
from .utils.attn_utils import  Co3Corrector
from .utils.gaussian_smoothing import GaussianSmoothing
from collections import defaultdict
import pickle 

def rescale_noise_cfg(noise_cfg, noise_pred_text, guidance_rescale=0.0):
    std_text = noise_pred_text.std(dim=list(range(1, noise_pred_text.ndim)), keepdim=True)
    std_cfg = noise_cfg.std(dim=list(range(1, noise_cfg.ndim)), keepdim=True)
    # rescale the results from guidance (fixes overexposure)
    noise_pred_rescaled = noise_cfg * (std_text / std_cfg)
    # mix with the original results from guidance by factor guidance_rescale to avoid "plain looking" images
    noise_cfg = guidance_rescale * noise_pred_rescaled + (1 - guidance_rescale) * noise_cfg
    return noise_cfg

logging.set_verbosity_error()





class Co3(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        sd_version = self.config.sd_version

        if sd_version == '2.1':
            model_key = "stabilityai/stable-diffusion-2-1-base"
        elif sd_version == '2.0':
            model_key = "stabilityai/stable-diffusion-2-base"
        elif sd_version == '1.5':
            model_key = "runwayml/stable-diffusion-v1-5"
        elif sd_version =='1.4':
            model_key = "CompVis/stable-diffusion-v1-4"
        elif sd_version =='xl':
            model_key= "stabilityai/stable-diffusion-xl-base-1.0"
        else:
            raise ValueError(f'Stable-diffusion version {sd_version} not supported.')

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Create SD models
        print('Loading SD model')
        pipe = StableDiffusionXLPipeline.from_pretrained(model_key, torch_dtype=torch.float16, variant="fp16",use_safetensors=True).to(device)

        pipe.enable_xformers_memory_efficient_attention()
        pipe.enable_vae_slicing()
        self.vae = AutoencoderKL.from_pretrained("madebyollin/sdxl-vae-fp16-fix", torch_dtype=torch.float16).to(device)

        self.vae_scale_factor = 2 ** (len(self.vae.config.block_out_channels) - 1)
        self.image_processor = VaeImageProcessor(vae_scale_factor=self.vae_scale_factor)
        
        self.tokenizer = pipe.tokenizer
        self.tokenizer_2 = pipe.tokenizer_2
        self.text_encoder = pipe.text_encoder
        self.text_encoder_2 = pipe.text_encoder_2
        self.unet = pipe.unet
        self.unet.enable_xformers_memory_efficient_attention() 
        self.device = self.unet.device
        self.prepare_prompts(self.config)
        self.prepare_embeds()
        
    
        self.scheduler = DDIMScheduler.from_pretrained(model_key, subfolder="scheduler")
        N_ts = len(self.scheduler.timesteps)
        self.scheduler.set_timesteps(self.config.n_timesteps, device=self.unet.device)
        
        self.skip = N_ts // self.config.n_timesteps
        self.final_alpha_cumprod = self.scheduler.final_alpha_cumprod.to(self.unet.device)
        self.scheduler.alphas_cumprod = torch.cat([torch.tensor([1.0]), self.scheduler.alphas_cumprod])

        print('custom checkpoint loaded')

        self.add_time_ids = self.compute_time_ids()
        self.add_time_ids = self.add_time_ids.to(self.unet.device)


        self.config.latent_corrector_ts = []
        if self.config.perform_latent_correction:
            if config.latent_corrector_type == 'Co3Corrector':
                t_cond = self.config.num_ts_to_correct
                self.config.latent_corrector_ts = self.scheduler.timesteps[:t_cond] if t_cond >= 0 else []  # self.scheduler.timesteps[:3]

                self.latentCorrector = Co3Corrector(
                                            # num_corrector_steps=self.config.num_latent_corrector_steps,
                                            lmda=self.config.lmda, # It should be less than 1. Should be >= self.config.guidance_scale when using cfg++
                                            modulate_comp_weights=self.config.modulate_comp_weights,
                                            step_size=1.0,
                                            use_cfg_corrector=True,
                                            verbose=True, #False if self.config.eval_on_full_promptset else True
                                            )
            else:
                raise ValueError(f'Invalid latent_corrector_type: {config.latent_corrector_type}')
        
        del pipe.tokenizer, pipe.tokenizer_2, pipe.text_encoder, pipe.text_encoder_2, pipe.unet,pipe.vae
        gc.collect()
        torch.cuda.empty_cache()


    def upcast_vae(self):
        dtype = self.vae.dtype
        self.vae.to(dtype=torch.float32)
        use_torch_2_0_or_xformers = isinstance(
            self.vae.decoder.mid_block.attentions[0].processor,
                AttnProcessor2_0,
            (
                XFormersAttnProcessor,
                FusedAttnProcessor2_0,
            ),
        )
        # if xformers or torch_2_0 is used attention block does not need
        # to be in float32 which can save lots of memory
        if use_torch_2_0_or_xformers:
            self.vae.post_quant_conv.to(dtype)
            self.vae.decoder.conv_in.to(dtype)
            self.vae.decoder.mid_block.to(dtype)
    
    def find_disc(self,embed,embed2):
        with torch.no_grad():
            token_embedding = self.text_encoder.get_input_embeddings()
            token_embedding2 = self.text_encoder_2.get_input_embeddings()

            embedding_matrix = token_embedding.weight
            embedding_matrix2 = token_embedding2.weight

            embed = embed.unsqueeze(0)
            embed2 = embed2.unsqueeze(0)
            hits = semantic_search(embed, embedding_matrix.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)
            hits2 = semantic_search(embed2, embedding_matrix2.float(), 
                                query_chunk_size=1, 
                                top_k=1,
                                score_function=dot_score)

            nn_indices = torch.tensor([hit[0]["corpus_id"] for hit in hits], device=embed.device)
            nn_indices2 = torch.tensor([hit[0]["corpus_id"] for hit in hits2], device=embed.device)
            
    @torch.no_grad()
    def get_text_embeds(self, prompt, negative_prompt, device="cuda"):        
        prompt_embeds, pooled_prompt_embeds = encode_prompt(
                    text_encoders=[self.text_encoder, self.text_encoder_2],
                    tokenizers=[self.tokenizer,self.tokenizer_2],
                    prompt=prompt,
                    text_input_ids_list=None
                )
        uncond_embeds, pooled_uncond_embeds = encode_prompt(
                    text_encoders=[self.text_encoder, self.text_encoder_2],
                    tokenizers=[self.tokenizer,self.tokenizer_2],
                    prompt=negative_prompt,
                    text_input_ids_list=None
                )
        text_embeddings = torch.cat([uncond_embeds,prompt_embeds])
        pooled_text_embeddings = torch.cat([pooled_uncond_embeds,pooled_prompt_embeds])
        return text_embeddings,pooled_text_embeddings
    
    def prepare_extra_step_kwargs(self, generator, eta):

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs
    
    @torch.no_grad()
    def decode_latent(self, latent):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
        return img
    
    def alpha(self, t):
        at = self.scheduler.alphas_cumprod[t] if t >= 0 else self.final_alpha_cumprod
        return at
    

    def compute_time_ids(self):
        # Adapted from pipeline.StableDiffusionXLPipeline._get_add_time_ids
        original_size = (self.config.resolution_h, self.config.resolution_w)
        target_size = (self.config.resolution_h, self.config.resolution_w)
        crops_coords_top_left = (self.config.crops_coords_top_left_h, self.config.crops_coords_top_left_w)
        add_time_ids = list(original_size + crops_coords_top_left + target_size)
        add_time_ids = torch.tensor([add_time_ids])
        # add_time_ids = add_time_ids.to(accelerator.device, dtype=weight_dtype)
        return add_time_ids
    
    def get_effective_scales(self, concept_stats, guidance_scale):
        """
        Calculate effective scales for each concept based on configuration.
        
        Args:
            concept_stats (list): Statistics for each concept (e.g., max attention values)
            guidance_scale (float): Base guidance scale
            
        Returns:
            list: Effective scales for each concept
        """
        if not concept_stats:
            return [guidance_scale] * (self.concept_num-1) + [0.01] # Last concept gets a very small scale
        
        total_weight = sum(concept_stats)
        target_weight = total_weight / self.concept_num
        
        # Calculate balance factors
        balance_factors = []
        for cc in range(self.concept_num - 1):
            balance_factors.append(target_weight / concept_stats[cc] if concept_stats[cc] > 0 else 1.0)
        balance_factors.append(-10.0)  # Last concept gets negative factor for near-zero scale
        
        effective_scales = []
        
        if self.config.use_cfgpp:
            # CFG++ scaling logic
            for cc in range(self.concept_num):
                if cc == self.concept_num - 1:
                    # Last effective scale close to zero
                    effective_scales.append(0.01 * guidance_scale)
                else:
                    effective_scales.append(guidance_scale * balance_factors[cc]) # Todo: torch.clip to [0,1]?
        else:
            # Exponential logic
            for cc in range(self.concept_num):
                if cc == self.concept_num - 1:
                    # Last effective scale close to zero
                    effective_scales.append(0.01)
                else:
                    effective_scales.append((guidance_scale) ** (balance_factors[cc]))
        
        return effective_scales
    

    @torch.no_grad()
    def denoise_step(self, x, t, step):

        text_embed_cond,text_embed_cond_pool=self.text_embeds

        next_t = t - self.skip
        at = self.alpha(t)
        at_next = self.alpha(next_t)

        sizes = x.shape
        log_step  = int(1000 - t.item())
           

        text_embed_uncond = text_embed_cond[0].unsqueeze(0)
        text_embed_multi = text_embed_cond[1].unsqueeze(0)

        text_embed_uncond_pool = text_embed_cond_pool[0].unsqueeze(0)
        text_embed_multi_pool = text_embed_cond_pool[1].unsqueeze(0)

        if t in self.config.latent_corrector_ts:
            text_embed_corrector = text_embed_cond[:-1]  # Exclude the last concept for correction
            text_embed_pool_corrector = text_embed_cond_pool[:-1]  # Exclude the last concept for correction
            corrector_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed_corrector.shape[0], 1)}
            corrector_added_conditions.update({"text_embeds": text_embed_pool_corrector})
            
            if self.config.corrector_algo == 'co3-resampling':
                x, _ = self.latentCorrector.co3_resampling(num_corrector_steps=self.config.num_latent_corrector_steps, text_embeddings=text_embed_corrector, 
                                                    unet=self.unet, x=x, unet_added_conditions=corrector_added_conditions, 
                                                    t=t,
                                                    at=at, add_time_ids=self.add_time_ids,
                                                    step_size=1.0,
                                                    beta=0.9
                                                    )

            elif self.config.corrector_algo == 'co3-corrector':
                x, _ = self.latentCorrector.co3_corrector(num_corrector_steps=self.config.num_latent_corrector_steps, text_embeddings=text_embed_corrector, 
                                                unet=self.unet, x=x, unet_added_conditions=corrector_added_conditions, 
                                                t=t,
                                                at=at, add_time_ids=self.add_time_ids,
                                                step_size=1.0,
                                                beta=1.0
                                                )   
            elif self.config.corrector_algo == 'co3-hybrid':
                # for first 3 steps perform sum_zero resampling but only T will have 10 steps. 
                # then sum_1 correction.
                assert self.config.num_resampling_steps > 0, "num_resampling_steps should be > 0 for co3-hybrid"
                beta = 1.0 if t == self.start_t else self.config.beta
                num_corrector_steps = 10 if t >= self.scheduler.timesteps[1] else self.config.num_latent_corrector_steps

                if t >= self.scheduler.timesteps[self.config.num_resampling_steps-1]:  
                    print(f"resampling step at t={t.item()}")
                    # if t == self.start_t or t==self.scheduler.timesteps[1]: # 10 steps at T and T-1
                    x, _ = self.latentCorrector.co3_resampling(num_corrector_steps=num_corrector_steps, 
                            text_embeddings=text_embed_corrector, 
                            unet=self.unet, x=x, 
                            unet_added_conditions=corrector_added_conditions, 
                            t=t,
                            at=at, add_time_ids=self.add_time_ids,
                            step_size=1.0,
                            beta=beta
                            )

                else:
                    print(f"correction step at t={t.item()}")
                    x, _ = self.latentCorrector.co3_corrector(num_corrector_steps=num_corrector_steps, 
                            text_embeddings=text_embed_corrector, 
                            unet=self.unet, x=x, 
                            unet_added_conditions=corrector_added_conditions, 
                            t=t,
                            at=at, add_time_ids=self.add_time_ids,
                            step_size=1.0,
                            beta=beta
                            )  
            
            
            else:
                raise ValueError(f"Corrector algo {self.config.corrector_algo} not recognized.")

        latent_model_input = torch.cat([x]+[x])
        text_embed = torch.cat([ text_embed_uncond
                            ,text_embed_multi], dim=0)
        text_embed_pool = torch.cat([text_embed_uncond_pool
                            ,text_embed_multi_pool], dim=0)

        unet_added_conditions = {"time_ids": self.add_time_ids.repeat(text_embed.shape[0], 1)}
        unet_added_conditions.update({"text_embeds": text_embed_pool})

        # with torch.no_grad():
        noise_pred = self.unet(latent_model_input, t, encoder_hidden_states=text_embed,added_cond_kwargs=unet_added_conditions)['sample']
        
        noise_pred_uncond = noise_pred[:1]
        # ------------------------------
        
        noise_pred_cond = noise_pred[1:2]
        noise_pred = noise_pred_uncond + self.config.guidance_scale * (noise_pred_cond - noise_pred_uncond)
        denoised_tweedie = (x - (1-at).sqrt() * noise_pred) / at.sqrt()
        if self.config.use_cfgpp:
            denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred_uncond
        else:
            denoised_latent = at_next.sqrt() * denoised_tweedie + (1-at_next).sqrt() * noise_pred
    
        if t == 1:
            denoised_latent = denoised_tweedie
        
        return denoised_latent

    def init_sampling(self):
        self.start_t = self.scheduler.timesteps[0]
        self.last_t_corr = self.scheduler.timesteps[self.config.num_ts_to_correct-1] if self.config.perform_latent_correction else self.scheduler.timesteps[1]
        os.makedirs(self.config.output_path_all, exist_ok=True)
        
    def run_sampling(self):
        self.init_sampling()
        normal = torch.randn(1,4,self.config.resolution_h//8,self.config.resolution_w//8).to(self.unet.device)  * self.scheduler.init_noise_sigma
        image = self.sample_loop(normal)
        return image
    
    @torch.no_grad()
    def sample_loop(self, x):
        with torch.autocast(device_type='cuda', dtype=torch.float16):
            for i, t in enumerate(tqdm(self.scheduler.timesteps, desc="Sampling")):
                x = self.denoise_step(x, t, i)
            
            with torch.no_grad():
                needs_upcasting = self.vae.dtype == torch.float16 and self.vae.config.force_upcast

                if needs_upcasting:
                    self.upcast_vae()
                    x = x.to(next(iter(self.vae.post_quant_conv.parameters())).dtype)
                elif x.dtype != self.vae.dtype:
                    if torch.backends.mps.is_available():
                        self.vae = self.vae.to(x.dtype)

                # unscale/denormalize the latents
                # denormalize with the mean and std if available and not None
                has_latents_mean = hasattr(self.vae.config, "latents_mean") and self.vae.config.latents_mean is not None
                has_latents_std = hasattr(self.vae.config, "latents_std") and self.vae.config.latents_std is not None
                if has_latents_mean and has_latents_std:
                    latents_mean = (
                        torch.tensor(self.vae.config.latents_mean).view(1, 4, 1, 1).to(x.device, x.dtype)
                    )
                    latents_std = (
                        torch.tensor(self.vae.config.latents_std).view(1, 4, 1, 1).to(x.device, x.dtype)
                    )
                    x = x * latents_std / self.vae.config.scaling_factor + latents_mean
                else:
                    x = x / self.vae.config.scaling_factor

                decoded_latent = self.vae.decode(x, return_dict=False)[0]

                # cast back to fp16 if needed
                if needs_upcasting:
                    self.vae.to(dtype=torch.float16)
            
            os.makedirs(self.config.output_path_all, exist_ok=True)
            image = self.image_processor.postprocess(decoded_latent, output_type='pil')
            

        return image
    
    @torch.no_grad()
    def decode_latent_for_viz(self, latent, path):
        with torch.autocast(device_type='cuda', dtype=torch.float32):
            latent = 1 / 0.18215 * latent
            img = self.vae.decode(latent).sample
            img = (img / 2 + 0.5).clamp(0, 1)
            
        T.ToPILImage()(img[0]).save(path)
    
    def prepare_prompts(self, config):
       
        self.prompt_orig = config.prompt_orig.split('+')[0]
        self.prompt_sep = config.prompt.split('+')
        prompts = []
        prompts.append(self.prompt_orig)
        concept_num = len(self.prompt_sep)
        self.concept_num = concept_num
        self.prompts_single = self.prompt_sep

        self.prompts = prompts + self.prompt_sep
        self.null_prompt = [config.negative_prompt]
        print(f"Processed prompts: prompts {self.prompts}, prompts_single {self.prompts_single}, null_prompt {self.null_prompt}")

    def prepare_embeds(self):
        self.text_embeds = self.get_text_embeds(self.prompts, self.null_prompt, device=self.unet.device) 
        self.text_embeds_single = self.get_text_embeds(self.prompts_single, self.null_prompt,device=self.unet.device)
        