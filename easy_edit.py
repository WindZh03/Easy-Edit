import math
import torch

from tqdm.notebook import tqdm
from PIL import Image, ImageOps

from diffusers import DDIMScheduler, DDIMInverseScheduler
from pipeline_stable_diffusion_grounded_instruct_pix2pix import StableDiffusionInstructPix2PixPipeline

from transformers import logging
logging.set_verbosity_error()

from external_mask_extractor import ExternalMaskExtractor  

import os
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"



def load_pil_image(image_path, resolution=512):
    image = Image.open(image_path).convert("RGB")
    width, height = image.size
    factor = resolution / max(width, height)
    factor = math.ceil(min(width, height) * factor / 64) * 64 / min(width, height)
    width = int((width * factor) // 64) * 64
    height = int((height * factor) // 64) * 64
    image = ImageOps.fit(image, (width, height), method=Image.Resampling.LANCZOS)
    return image


def inference(pipeline, image_pil, instruction, 
              image_guidance_scale=1.5, text_guidance_scale=7.5, seed=42, blending_range=[100,1],use_pix=False):
    verbose = True
    num_timesteps = 100
    device = 'cuda:0'
    external_mask_pil, chosen_noun_phrase = mask_extractor.get_external_mask(image_pil, instruction, verbose=verbose,use_pix=use_pix)

    inv_results = pipeline.invert(instruction, image_pil, num_inference_steps=num_timesteps, inv_range=blending_range)

    generator = torch.Generator(device).manual_seed(seed) if seed is not None else torch.Generator(device)
    edited_image = pipeline(instruction, src_mask=external_mask_pil, image=image_pil,
                            guidance_scale=text_guidance_scale, image_guidance_scale=image_guidance_scale,
                            num_inference_steps=num_timesteps, generator=generator).images[0]
    print('generate edited_image')
    torch.cuda.empty_cache()  # 推理完成后释放缓存
    return edited_image



def initialize_model():
    global pipeline
    if pipeline is None:
        # 仅在第一次调用时加载模型
        pipeline = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            'timbrooks/instruct-pix2pix',
            torch_dtype=torch.float16,
            safety_checker=None,
            local_files_only=True
        ).to(device)
        pipeline.inverse_scheduler = DDIMInverseScheduler.from_config(pipeline.scheduler.config, set_alpha_to_zero=False)
        pipeline.scheduler.set_timesteps(num_timesteps)
        pipeline.inverse_scheduler.set_timesteps(num_timesteps)
    return pipeline


def process_image(image, edit_prompt, image_guidance_scale=1.5, text_guidance_scale=7.5, seed=42, blending_range=[100, 1],use_pix=False):
    # 懒加载模型
    global pipeline
    if pipeline is None:
        pipeline = initialize_model()
        print('initialize_model')
    # 调用推理函数
    print('start inference')
    edited_image = inference(pipeline, image, edit_prompt, image_guidance_scale, 
                           text_guidance_scale, seed, blending_range,use_pix)
    return edited_image


pipeline = None
device = "cuda" if torch.cuda.is_available() else "cpu"
num_timesteps = 100
mask_extractor = ExternalMaskExtractor(device=device)