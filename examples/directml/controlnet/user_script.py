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
    return model_name != get_base_model_name(model_name)


# Merges LoRA weights into the layers of a base model
def merge_lora_weights(base_model, lora_model_id, submodel_name="unet", scale=1.0):
    from collections import defaultdict
    from functools import reduce

    from diffusers.loaders import LORA_WEIGHT_NAME
    from diffusers.models.attention_processor import LoRAAttnProcessor
    from diffusers.utils import DIFFUSERS_CACHE
    from diffusers.utils.hub_utils import _get_model_file

    # Load LoRA weights
    model_file = _get_model_file(
        lora_model_id,
        weights_name=LORA_WEIGHT_NAME,
        cache_dir=DIFFUSERS_CACHE,
        force_download=False,
        resume_download=False,
        proxies=None,
        local_files_only=False,
        use_auth_token=None,
        revision=None,
        subfolder=None,
        user_agent={
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        },
    )
    lora_state_dict = torch.load(model_file, map_location="cpu")

    # All keys in the LoRA state dictionary should have 'lora' somewhere in the string.
    keys = list(lora_state_dict.keys())
    assert all("lora" in k for k in keys)

    if all(key.startswith(submodel_name) for key in keys):
        # New format (https://github.com/huggingface/diffusers/pull/2918) supports LoRA weights in both the
        # unet and text encoder where keys are prefixed with 'unet' or 'text_encoder', respectively.
        submodel_state_dict = {k: v for k, v in lora_state_dict.items() if k.startswith(submodel_name)}
    else:
        # Old format. Keys will not have any prefix. This only applies to unet, so exit early if this is
        # optimizing the text encoder.
        if submodel_name != "unet":
            return
        submodel_state_dict = lora_state_dict

    # Group LoRA weights into attention processors
    attn_processors = {}
    lora_grouped_dict = defaultdict(dict)
    for key, value in submodel_state_dict.items():
        attn_processor_key, sub_key = ".".join(key.split(".")[:-3]), ".".join(key.split(".")[-3:])
        lora_grouped_dict[attn_processor_key][sub_key] = value

    for key, value_dict in lora_grouped_dict.items():
        rank = value_dict["to_k_lora.down.weight"].shape[0]
        cross_attention_dim = value_dict["to_k_lora.down.weight"].shape[1]
        hidden_size = value_dict["to_k_lora.up.weight"].shape[0]

        attn_processors[key] = LoRAAttnProcessor(
            hidden_size=hidden_size, cross_attention_dim=cross_attention_dim, rank=rank
        )
        attn_processors[key].load_state_dict(value_dict)

    # Merge LoRA attention processor weights into existing Q/K/V/Out weights
    for name, proc in attn_processors.items():
        attention_name = name[: -len(".processor")]
        attention = reduce(getattr, attention_name.split(sep="."), base_model)
        attention.to_q.weight.data += scale * torch.mm(proc.to_q_lora.up.weight, proc.to_q_lora.down.weight)
        attention.to_k.weight.data += scale * torch.mm(proc.to_k_lora.up.weight, proc.to_k_lora.down.weight)
        attention.to_v.weight.data += scale * torch.mm(proc.to_v_lora.up.weight, proc.to_v_lora.down.weight)
        attention.to_out[0].weight.data += scale * torch.mm(proc.to_out_lora.up.weight, proc.to_out_lora.down.weight)


# -----------------------------------------------------------------------------
# TEXT ENCODER
# -----------------------------------------------------------------------------
def text_encoder_inputs(batchsize, torch_dtype):
    return torch.zeros((batchsize, 77), dtype=torch_dtype)

def text_encoder_load(model_name):
    base_model_id = get_base_model_name(model_name)
    model = CLIPTextModel.from_pretrained(base_model_id, subfolder="text_encoder")
    if is_lora_model(model_name):
        merge_lora_weights(model, model_name, "text_encoder")
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
    base_model_id = get_base_model_name(model_name)
    model = UNet2DConditionModel_Cnet.from_pretrained(base_model_id, subfolder="unet")
    if is_lora_model(model_name):
        merge_lora_weights(model, model_name, "unet")
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
        "controlnet_cond": torch.rand((batchsize, 3,config.unet_sample_size*8, config.unet_sample_size*8), dtype=torch_dtype),
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
    base_model_id = get_base_model_name(model_name)
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
    base_model_id = get_base_model_name(model_name)
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
    base_model_id = get_base_model_name(model_name)
    model = StableDiffusionSafetyChecker.from_pretrained(base_model_id, subfolder="safety_checker")
    model.forward = model.forward_onnx
    return model

def safety_checker_conversion_inputs(model):
    return tuple(safety_checker_inputs(1, torch.float32).values())

def safety_checker_data_loader(data_dir, batchsize, *args, **kwargs):
    return RandomDataLoader(safety_checker_inputs, batchsize, torch.float16)
