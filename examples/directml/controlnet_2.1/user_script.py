# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
#https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16/blob/a01a901735bb6fefc6a72b9ebc1d1beac3a387b8/conv_sd_to_onnx.py
#https://github.com/huggingface/diffusers/blob/1d686bac8146037e97f3fd8c56e4063230f71751/examples/community/stable_diffusion_controlnet_img2img.py
# --------------------------------------------------------------------------
import config
import torch
from diffusers import AutoencoderKL, UNet2DConditionModel, ControlNetModel
from diffusers.models.unet_2d_condition import UNet2DConditionOutput
from diffusers.pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
from huggingface_hub import model_info
from transformers.models.clip.modeling_clip import CLIPTextModel
from typing import Union, Optional, Tuple


class UNet2DConditionModel_Cnet(UNet2DConditionModel):
    def forward(
        self,
        sample: torch.FloatTensor,
        timestep: Union[torch.Tensor, float, int],
        encoder_hidden_states: torch.Tensor,
        down_block_add_res00: Optional[torch.Tensor] = None,
        down_block_add_res01: Optional[torch.Tensor] = None,
        down_block_add_res02: Optional[torch.Tensor] = None,
        down_block_add_res03: Optional[torch.Tensor] = None,
        down_block_add_res04: Optional[torch.Tensor] = None,
        down_block_add_res05: Optional[torch.Tensor] = None,
        down_block_add_res06: Optional[torch.Tensor] = None,
        down_block_add_res07: Optional[torch.Tensor] = None,
        down_block_add_res08: Optional[torch.Tensor] = None,
        down_block_add_res09: Optional[torch.Tensor] = None,
        down_block_add_res10: Optional[torch.Tensor] = None,
        down_block_add_res11: Optional[torch.Tensor] = None,
        mid_block_additional_residual: Optional[torch.Tensor] = None,
        return_dict: bool = False,
    ) -> Union[UNet2DConditionOutput, Tuple]:
        down_block_add_res = (
            down_block_add_res00, down_block_add_res01, down_block_add_res02,
            down_block_add_res03, down_block_add_res04, down_block_add_res05,
            down_block_add_res06, down_block_add_res07, down_block_add_res08,
            down_block_add_res09, down_block_add_res10, down_block_add_res11)
        return super().forward(
            sample = sample,
            timestep = timestep,
            encoder_hidden_states = encoder_hidden_states,
            down_block_additional_residuals = down_block_add_res,
            mid_block_additional_residual = mid_block_additional_residual,
            return_dict = return_dict
        )
# Helper latency-only dataloader that creates random tensors with no label
class RandomDataLoader:
    def __init__(self, create_inputs_func, batchsize, torch_dtype):
        self.create_input_func = create_inputs_func
        self.batchsize = batchsize
        self.torch_dtype = torch_dtype

    def __getitem__(self, idx):
        label = None
        return self.create_input_func(self.batchsize, self.torch_dtype), label

def get_base_model_name(model_name):
    return model_info(model_name).cardData.get("base_model", model_name)

def is_lora_model(model_name):
    return model_name != model_name #get_base_model_name(model_name)

# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------
def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, config.num_tokens), dtype=torch_dtype)

def text_encoder_load(model_name):
    #base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(model_name, subfolder="text_encoder")
    return model

def text_encoder_conversion_inputs(model):
    return text_encoder_inputs(1, torch.int32)

def text_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(text_encoder_inputs, batchsize, torch.int32)

# -----------------------------------------------------------------------------
# UNET
# -----------------------------------------------------------------------------
def unet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, config.unet_in_channels, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, config.num_tokens, config.cross_attention_dim), dtype=torch_dtype),
        "down_block_0": torch.randn((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_1": torch.randn((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_2": torch.randn((batchsize, 320, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "down_block_3": torch.randn((batchsize, 320, config.unet_sample_size//2, config.unet_sample_size//2), dtype=torch_dtype),
        "down_block_4": torch.randn((batchsize, 640, config.unet_sample_size//2, config.unet_sample_size//2), dtype=torch_dtype),
        "down_block_5": torch.randn((batchsize, 640, config.unet_sample_size//2, config.unet_sample_size//2), dtype=torch_dtype),
        "down_block_6": torch.randn((batchsize, 640, config.unet_sample_size//4, config.unet_sample_size//4), dtype=torch_dtype),
        "down_block_7": torch.randn((batchsize, 1280, config.unet_sample_size//4, config.unet_sample_size//4), dtype=torch_dtype),
        "down_block_8": torch.randn((batchsize, 1280, config.unet_sample_size//4, config.unet_sample_size//4), dtype=torch_dtype),
        "down_block_9": torch.randn((batchsize, 1280, config.unet_sample_size//8, config.unet_sample_size//8), dtype=torch_dtype),
        "down_block_10": torch.randn((batchsize, 1280, config.unet_sample_size//8, config.unet_sample_size//8), dtype=torch_dtype),
        "down_block_11": torch.randn((batchsize, 1280, config.unet_sample_size//8, config.unet_sample_size//8), dtype=torch_dtype),
        "mid_block_additional_residual": torch.randn((batchsize, 1280, config.unet_sample_size//8, config.unet_sample_size//8), dtype=torch_dtype),
   
    }

    # use as kwargs since they won't be in the correct position if passed along with the tuple of inputs
    kwargs = {
        "return_dict": False,
    }
   
    return inputs

def unet_load(model_name):
    #base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel_Cnet.from_pretrained(model_name, subfolder="unet")
    return model

def unet_conversion_inputs(model):
    return tuple(unet_inputs(1, torch.float32, True).values())


def unet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(unet_inputs, batchsize, torch.float16)

# -----------------------------------------------------------------------------
# CONTROLNET
# -----------------------------------------------------------------------------
def controlnet_inputs(batchsize, torch_dtype, is_conversion_inputs=False):
    inputs = {
        "sample": torch.rand((batchsize, config.unet_in_channels, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "timestep": torch.rand((batchsize,), dtype=torch_dtype),
        "encoder_hidden_states": torch.rand((batchsize, config.num_tokens, config.cross_attention_dim), dtype=torch_dtype),
        "controlnet_cond": torch.rand((batchsize, config.unet_in_channels, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype),
        "conditioning_scale": torch.rand((1,), dtype=torch_dtype),
    }
    return inputs

def controlnet_load(model_name):
    base_model_id = model_name
    model = ControlNetModel.from_pretrained(base_model_id)
    return model

def controlnet_conversion_inputs(model):
    return tuple(controlnet_inputs(1, torch.float32, True).values())

def controlnet_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(controlnet_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE ENCODER
# -----------------------------------------------------------------------------
def vae_encoder_inputs(batchsize, torch_dtype):
    return {
        "sample": torch.rand((batchsize, 3, config.vae_sample_size, config.vae_sample_size), dtype=torch_dtype),
        "return_dict": False,
    }

def vae_encoder_load(model_name):
    base_model_id = model_name#get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = lambda sample, return_dict: model.encode(sample, return_dict)[0].sample()
    return model

def vae_encoder_conversion_inputs(model):
    return tuple(vae_encoder_inputs(1, torch.float32).values())

def vae_encoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_encoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# VAE DECODER
# -----------------------------------------------------------------------------
def vae_decoder_inputs(batchsize, torch_dtype):
    return {
        "latent_sample": torch.rand(
            (batchsize, 4, config.unet_sample_size, config.unet_sample_size), dtype=torch_dtype
        ),
        "return_dict": False,
    }

def vae_decoder_load(model_name):
    base_model_id = model_name#get_base_model_name(model_name)
    model = AutoencoderKL.from_pretrained(base_model_id, subfolder="vae")
    model.forward = model.decode
    return model

def vae_decoder_conversion_inputs(model):
    return tuple(vae_decoder_inputs(1, torch.float32).values())

def vae_decoder_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(vae_decoder_inputs, batchsize, torch.float16)


# -----------------------------------------------------------------------------
# SAFETY CHECKER
# -----------------------------------------------------------------------------
def safety_checker_inputs(batchsize, torch_dtype):
    return {
        "clip_input": torch.rand((batchsize, 3, 224, 224), dtype=torch_dtype),
        "images": torch.rand((batchsize, config.vae_sample_size, config.vae_sample_size, 3), dtype=torch_dtype),
    }

def safety_checker_load(model_name):
    base_model_id = model_name #get_base_model_name(model_name)
    model = StableDiffusionSafetyChecker.from_pretrained(base_model_id, subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model

def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())

def safety_checker_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)
