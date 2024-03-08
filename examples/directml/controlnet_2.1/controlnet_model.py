# Copyright 2024 The HuggingFace Team.
# based on https://github.com/huggingface/diffusers/blob/main/examples/community/stable_diffusion_controlnet_img2img.py 
# and https://github.com/Amblyopius/Stable-Diffusion-ONNX-FP16
#                   GNU GENERAL PUBLIC LICENSE
#                      Version 3, 29 June 2007

# Copyright (C) 2007 Free Software Foundation, Inc. <https://fsf.org/>
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.


import inspect
from typing import Callable, List, Optional, Union

import numpy as np
import torch
import PIL
from transformers import CLIPFeatureExtractor, CLIPTokenizer

from diffusers.configuration_utils import FrozenDict
from diffusers.schedulers import DDIMScheduler, LMSDiscreteScheduler, PNDMScheduler
from diffusers.utils import deprecate, logging, PIL_INTERPOLATION
from diffusers.pipelines.onnx_utils import ORT_TO_NP_TYPE, OnnxRuntimeModel
from diffusers  import DiffusionPipeline
from diffusers.pipelines.stable_diffusion import StableDiffusionPipelineOutput
#from optimum.onnxruntime import _prepare_io_binding,get_ordered_input_names
import onnxruntime as ort
import time


logger = logging.get_logger(__name__)

class OnnxStableDiffusionControlNetPipeline(DiffusionPipeline):
    vae_encoder: OnnxRuntimeModel
    vae_decoder: OnnxRuntimeModel
    text_encoder: OnnxRuntimeModel
    tokenizer: CLIPTokenizer
    unet: OnnxRuntimeModel
    controlnet: OnnxRuntimeModel
    scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler]
    safety_checker: OnnxRuntimeModel
    feature_extractor: CLIPFeatureExtractor

    _optional_components = ["safety_checker", "feature_extractor"]

    def __init__(
        self,
        vae_encoder: OnnxRuntimeModel,
        vae_decoder: OnnxRuntimeModel,
        text_encoder: OnnxRuntimeModel,
        tokenizer: CLIPTokenizer,
        unet: OnnxRuntimeModel,
        controlnet: OnnxRuntimeModel,
        scheduler: Union[DDIMScheduler, PNDMScheduler, LMSDiscreteScheduler],
        safety_checker: OnnxRuntimeModel,
        feature_extractor: CLIPFeatureExtractor,
        requires_safety_checker: bool = True,
    ):
        super().__init__()

        if hasattr(scheduler.config, "steps_offset") and scheduler.config.steps_offset != 1:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} is outdated. `steps_offset`"
                f" should be set to 1 instead of {scheduler.config.steps_offset}. Please make sure "
                "to update the config accordingly as leaving `steps_offset` might led to incorrect results"
                " in future versions. If you have downloaded this checkpoint from the Hugging Face Hub,"
                " it would be very nice if you could open a Pull request for the `scheduler/scheduler_config.json`"
                " file"
            )
            deprecate("steps_offset!=1", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["steps_offset"] = 1
            scheduler._internal_dict = FrozenDict(new_config)

        if hasattr(scheduler.config, "clip_sample") and scheduler.config.clip_sample is True:
            deprecation_message = (
                f"The configuration file of this scheduler: {scheduler} has not set the configuration `clip_sample`."
                " `clip_sample` should be set to False in the configuration file. Please make sure to update the"
                " config accordingly as not setting `clip_sample` in the config might lead to incorrect results in"
                " future versions. If you have downloaded this checkpoint from the Hugging Face Hub, it would be very"
                " nice if you could open a Pull request for the `scheduler/scheduler_config.json` file"
            )
            deprecate("clip_sample not set", "1.0.0", deprecation_message, standard_warn=False)
            new_config = dict(scheduler.config)
            new_config["clip_sample"] = False
            scheduler._internal_dict = FrozenDict(new_config)

        if safety_checker is None and requires_safety_checker:
            logger.warning(
                f"You have disabled the safety checker for {self.__class__} by passing `safety_checker=None`. Ensure"
                " that you abide to the conditions of the Stable Diffusion license and do not expose unfiltered"
                " results in services or applications open to the public. Both the diffusers team and Hugging Face"
                " strongly recommend to keep the safety filter enabled in all public facing circumstances, disabling"
                " it only for use-cases that involve analyzing network behavior or auditing its results. For more"
                " information, please have a look at https://github.com/huggingface/diffusers/pull/254 ."
            )

        if safety_checker is not None and feature_extractor is None:
            raise ValueError(
                "Make sure to define a feature extractor when loading {self.__class__} if you want to use the safety"
                " checker. If you do not want to use the safety checker, you can pass `'safety_checker=None'` instead."
            )

        self.register_modules(
            vae_encoder = vae_encoder,
            vae_decoder = vae_decoder,
            text_encoder = text_encoder,
            tokenizer = tokenizer,
            unet = unet,
            controlnet = controlnet,
            scheduler = scheduler,
            safety_checker = safety_checker,
            feature_extractor = feature_extractor,
        )
        self.register_to_config(requires_safety_checker = requires_safety_checker)
         
    def get_timesteps(self,num_inference_steps,strength):
        # get the original timestep using init_timestep
        init_timestep = min(int(num_inference_steps * strength), num_inference_steps)

        t_start = max(num_inference_steps - init_timestep, 0)
        timesteps = self.scheduler.timesteps[t_start:]

        return timesteps, num_inference_steps - t_start    
    
    def _default_height_width(self, height, width, image):
        if isinstance(image, list):
            image = image[0]

        if height is None:
            if isinstance(image, PIL.Image.Image):
                height = image.height
            elif isinstance(image, np.ndarray):
                height = image.shape[3]

            height = (height // 8) * 8  # round down to nearest multiple of 8

        if width is None:
            if isinstance(image, PIL.Image.Image):
                width = image.width
            elif isinstance(image, np.ndarray):
                width = image.shape[2]

            width = (width // 8) * 8  # round down to nearest multiple of 8

        return height, width
        
    def prepare_image(self, image, width, height, batch_size, num_images_per_prompt,dtype, use_vae_encoder):
        if not isinstance(image, np.ndarray):
            if isinstance(image, PIL.Image.Image):
                image = [image]

            if isinstance(image[0], PIL.Image.Image):
                image = [
                    np.array(i.resize((width, height), resample=PIL_INTERPOLATION["lanczos"]))[None, :] for i in image
                ]
                image = np.concatenate(image, axis=0)
                image = np.array(image).astype(np.float16 if dtype=="fp16" else np.float32 ) / 255.0
                image = image.transpose(0, 3, 1, 2)
                image = (image - 0.5)/0.5
                image = torch.from_numpy(image)
            elif isinstance(image[0], np.ndarray):
                image = np.concatenate(image, axis=0)
                image = torch.from_numpy(image)
        else:
            image = np.clip(np.array(image).astype(np.float16 if dtype=="fp16" else np.float32) / 255.0,0,1)
            image = image.transpose(0, 3, 1, 2)
            image = (image - 0.5)/0.5
            image = torch.from_numpy(image)

        image_batch_size = image.shape[0]

        if image_batch_size == 1:
            repeat_by = batch_size
        else:
            # image batch size is the same as prompt batch size
            repeat_by = num_images_per_prompt

        if use_vae_encoder:
            image=self.vae_encoder(sample=image)[0]* 0.182158
        else:
            image = image.repeat_interleave(repeat_by, dim=0)
        return image
        
    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_latents
    def prepare_latents(self, batch_size, num_channels_latents, latents_shape, dtype, generator, latents=None):
        shape = latents_shape
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        if latents is None:
            latents = generator.randn(*shape).astype(dtype)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta, torch_gen):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = torch_gen
        return extra_step_kwargs

    def _encode_prompt(self, prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt):
        r"""
        Encodes the prompt into text encoder hidden states.

        Args:
            prompt (`str` or `List[str]` or 'np.ndarray'):
                prompt to be encoded
            num_images_per_prompt (`int`):
                number of images that should be generated per prompt
            do_classifier_free_guidance (`bool`):
                whether to use classifier free guidance or not
            negative_prompt (`str` or `List[str]` or 'np.ndarray'):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
        """
        # if type prompt is np.ndarray, concatenate prompt and negative prompt here
        if isinstance(prompt, np.ndarray):
            prompt_embeds = np.repeat(prompt, num_images_per_prompt, axis=0)
            
        else:
            batch_size = len(prompt) if isinstance(prompt, list) else 1

            # get prompt text embeddings
            text_inputs = self.tokenizer(
                prompt,
                padding = "max_length",
                max_length = self.tokenizer.model_max_length,
                truncation = True,
                return_tensors = "np",
            )
            text_input_ids = text_inputs.input_ids
            untruncated_ids = self.tokenizer(prompt, padding="max_length", return_tensors="np").input_ids

            if not np.array_equal(text_input_ids, untruncated_ids):
                removed_text = self.tokenizer.batch_decode(untruncated_ids[:, self.tokenizer.model_max_length - 1 : -1])
                logger.warning(
                    "The following part of your input was truncated because CLIP can only handle sequences up to"
                    f" {self.tokenizer.model_max_length} tokens: {removed_text}"
                )

            prompt_embeds = self.text_encoder(input_ids=text_input_ids.astype(np.int32))[0]
            prompt_embeds = np.repeat(prompt_embeds, num_images_per_prompt, axis=0)

        # get unconditional embeddings for classifier free guidance
        if do_classifier_free_guidance:
            if isinstance(prompt, np.ndarray):
                if negative_prompt is None:
                    negative_prompt = np.zeros_like(prompt)
                negative_prompt_embeds = np.repeat(negative_prompt, num_images_per_prompt, axis=0)

            else:
                uncond_tokens: List[str]
                if negative_prompt is None:
                    uncond_tokens = [""] * batch_size
                elif type(prompt) is not type(negative_prompt):
                    raise TypeError(
                        f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                        f" {type(prompt)}."
                    )
                elif isinstance(negative_prompt, str):
                    uncond_tokens = [negative_prompt] * batch_size
                elif batch_size != len(negative_prompt):
                    raise ValueError(
                        f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                        f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                        " the batch size of `prompt`."
                    )
                else:
                    uncond_tokens = negative_prompt

                max_length = text_input_ids.shape[-1]
                uncond_input = self.tokenizer(
                    uncond_tokens,
                    padding = "max_length",
                    max_length = max_length,
                    truncation = True,
                    return_tensors = "np",
                )
                negative_prompt_embeds = self.text_encoder(input_ids=uncond_input.input_ids.astype(np.int32))[0]
                negative_prompt_embeds = np.repeat(negative_prompt_embeds, num_images_per_prompt, axis=0)

            # For classifier free guidance, we need to do two forward passes.
            # Here we concatenate the unconditional and text embeddings into a single batch
            # to avoid doing two forward passes
            prompt_embeds = np.concatenate([negative_prompt_embeds, prompt_embeds])

        return prompt_embeds
  
    def __call__(
        self,
        prompt: Union[str, List[str], np.ndarray],
        use_vae_encoder: bool = False,
        image: np.ndarray = None,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: Optional[int] = 50,
        guidance_scale: Optional[float] = 4.0,
        negative_prompt: Optional[Union[str, List[str],np.ndarray]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: Optional[float] = 0.0,
        generator: Optional[np.random.RandomState] = None,
        latents: Optional[np.ndarray] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, np.ndarray], None]] = None,
        callback_steps: int = 1,
        controlnet_conditioning_scale: float = 1.0,
        strength: float = 1.0,
        controlnet_guidance_start: float = 0.0,
        controlnet_guidance_end: float = 1.0,
        model_dtype: str = "fp16",
        use_io_binding: bool = True
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`, `np.ndarray`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, one has to pass `prompt_embeds`.
                instead.
            use_vae_encoder (`bool`, *optional*, defaults to `False`):
                Whether to use the VAE encoder to encode the image. If `True`, the VAE encoder will be used to encode the
            image (`torch.Tensor` or `PIL.Image.Image`):
                `Image`, or tensor representing an image batch which will be inpainted, *i.e.* parts of the image will
                be masked out with `mask_image` and repainted according to `prompt`.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator` or `List[torch.Generator]`, *optional*):
                One or a list of [torch generator(s)](https://pytorch.org/docs/stable/generated/torch.Generator.html)
                to make generation deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.
            controlnet_conditioning_scale (`float`, *optional*, defaults to 1.0):
                The outputs of the controlnet are multiplied by `controlnet_conditioning_scale` before they are added
                to the residual in the original unet.
            strength (`float`, *optional*):
                Conceptually, indicates how much to transform the reference `image`. Must be between 0 and 1. `image`
                will be used as a starting point, adding more noise to it the larger the `strength`. The number of
                denoising steps depends on the amount of noise initially added. When `strength` is 1, added noise will
                be maximum and the denoising process will run for the full number of iterations specified in
                `num_inference_steps`. A value of 1, therefore, essentially ignores `image`.
            controlnet_guidance_start ('float', *optional*, defaults to 0.0):
                The percentage of total steps the controlnet starts applying. Must be between 0 and 1.
            controlnet_guidance_end ('float', *optional*, defaults to 1.0):
                The percentage of total steps the controlnet ends applying. Must be between 0 and 1. Must be greater
                than `controlnet_guidance_start`.
            model_dtype ('str', *optional*, default to fp16):
                This is being used when doing inference for a model in float32 or in float16
            use_io_binding('bool', *optional*, default to True)
                Whether use io binding when communicating between controlnet and unet model. setting it to True will
                improve the performance
        """

        if model_dtype == "fp16":
            data_type = np.float16
        else:
            data_type = np.single
      
        # Define call parameters
        if isinstance(prompt, str):
            batch_size = 1
        elif isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
            
        if generator:
            torch_seed = generator.randint(2147483647)
            torch_gen = torch.Generator().manual_seed(torch_seed)
        else:
            generator = np.random
            torch_gen = None

        # Default height and width to unet 
        height, width = self._default_height_width(height, width, image)

        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if (callback_steps is None) or (
            callback_steps is not None and (not isinstance(callback_steps, int) or callback_steps <= 0)
        ):
            raise ValueError(
                f"`callback_steps` has to be a positive integer but is {callback_steps} of type"
                f" {type(callback_steps)}."
            )

        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0 

        print("guidance scale is: ",guidance_scale)

        # Encode input prompt
        prompt_embeds = self._encode_prompt(
            prompt, num_images_per_prompt, do_classifier_free_guidance, negative_prompt
        )
        
        # Prepare image
        image = self.prepare_image(
            image,
            width,
            height,
            batch_size * num_images_per_prompt,
            num_images_per_prompt,
            model_dtype,
            use_vae_encoder
        )
     
        if do_classifier_free_guidance:
            if use_vae_encoder:
                image = np.tile(image,(2,1,1,1))
            else:
                image = np.concatenate([image] * 2)
        # get the initial random noise unless the user supplied it
        latents_dtype = prompt_embeds.dtype
        latents_shape = (batch_size * num_images_per_prompt, 4, image.shape[2], image.shape[3])
       
        # Prepare latent variables
        num_channels_latents = 4
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            latents_shape,
            latents_dtype,
            generator,
            latents,
        )

        # set timesteps
        self.scheduler.set_timesteps(num_inference_steps)
        timesteps = self.scheduler.timesteps
        timesteps, num_inference_steps = self.get_timesteps(num_inference_steps, strength)

        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://arxiv.org/abs/2010.02502
        # and should be between [0, 1]
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta, torch_gen)

        timestep_dtype = next(
            (input.type for input in self.unet.model.get_inputs() if input.name == "timestep"), "tensor(float)"
        )
        timestep_dtype = ORT_TO_NP_TYPE[timestep_dtype]
        
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        
        controlnet_cond_latent = image.astype(data_type)
       
        controlnet_cond=controlnet_cond_latent#np.concatenate([controlnet_cond_latent]*2) if do_classifier_free_guidance else controlnet_cond_latent
      
        conditioning_scale = data_type([1.0])

        if use_io_binding:
            binding_device = "dml"
            unet_sample_size = height//8
            #negative prompt increase the batch size
            batch_negative_prompt = batch_size * 2 if do_classifier_free_guidance else batch_size

            controlnet_binding = self.controlnet.model.io_binding()
            unet_binding = self.unet.model.io_binding()

            prompt_embeds           = ort.OrtValue.ortvalue_from_numpy(prompt_embeds,binding_device)
            controlnet_cond         = ort.OrtValue.ortvalue_from_numpy(controlnet_cond,binding_device)
            conditioning_scale      = ort.OrtValue.ortvalue_from_numpy(conditioning_scale,binding_device)
            down_block_res_samples  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,320,unet_sample_size,unet_sample_size],data_type,binding_device)
            b_2775                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,320,unet_sample_size,unet_sample_size],data_type,binding_device)
            b_2776                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,320,unet_sample_size,unet_sample_size],data_type,binding_device)
            b_2777                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,320,unet_sample_size//2,unet_sample_size//2],data_type,binding_device)
            b_2778                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,640,unet_sample_size//2,unet_sample_size//2],data_type,binding_device)
            b_2779                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,640,unet_sample_size//2,unet_sample_size//2],data_type,binding_device)
            b_2780                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,640,unet_sample_size//4,unet_sample_size//4],data_type,binding_device)
            b_2781                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,1280,unet_sample_size//4,unet_sample_size//4],data_type,binding_device)
            b_2782                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,1280,unet_sample_size//4,unet_sample_size//4],data_type,binding_device)
            b_2783                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,1280,unet_sample_size//8,unet_sample_size//8],data_type,binding_device)
            b_2784                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,1280,unet_sample_size//8,unet_sample_size//8],data_type,binding_device)
            b_2785                  = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt,1280,unet_sample_size//8,unet_sample_size//8],data_type,binding_device)
            mid_block_res_sample    = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt, 1280, unet_sample_size//8,unet_sample_size//8],data_type,binding_device)
            out_sample              = ort.OrtValue.ortvalue_from_shape_and_type([batch_negative_prompt, 4, unet_sample_size,unet_sample_size],data_type,binding_device)

            controlnet_binding.bind_ortvalue_input("encoder_hidden_states",prompt_embeds)
            controlnet_binding.bind_ortvalue_input("controlnet_cond",controlnet_cond)
            controlnet_binding.bind_ortvalue_input("conditioning_scale",conditioning_scale)
            controlnet_binding.bind_ortvalue_output("down_block_res_samples",down_block_res_samples)
            controlnet_binding.bind_ortvalue_output("mid_block_res_sample",b_2775)
            controlnet_binding.bind_ortvalue_output("2775",b_2776)
            controlnet_binding.bind_ortvalue_output("2776",b_2777)
            controlnet_binding.bind_ortvalue_output("2777",b_2778)
            controlnet_binding.bind_ortvalue_output("2778",b_2779)
            controlnet_binding.bind_ortvalue_output("2779",b_2780)
            controlnet_binding.bind_ortvalue_output("2780",b_2781)
            controlnet_binding.bind_ortvalue_output("2781",b_2782)
            controlnet_binding.bind_ortvalue_output("2782",b_2783)
            controlnet_binding.bind_ortvalue_output("2783",b_2784)
            controlnet_binding.bind_ortvalue_output("2784",b_2785)
            controlnet_binding.bind_ortvalue_output("2785",mid_block_res_sample)

            unet_binding.bind_ortvalue_input("encoder_hidden_states",prompt_embeds)
            unet_binding.bind_ortvalue_input("down_block_0",down_block_res_samples)
            unet_binding.bind_ortvalue_input("down_block_1",b_2775)
            unet_binding.bind_ortvalue_input("down_block_2",b_2776)
            unet_binding.bind_ortvalue_input("down_block_3",b_2777)
            unet_binding.bind_ortvalue_input("down_block_4",b_2778)
            unet_binding.bind_ortvalue_input("down_block_5",b_2779)
            unet_binding.bind_ortvalue_input("down_block_6",b_2780)
            unet_binding.bind_ortvalue_input("down_block_7",b_2781)
            unet_binding.bind_ortvalue_input("down_block_8",b_2782)
            unet_binding.bind_ortvalue_input("down_block_9",b_2783)
            unet_binding.bind_ortvalue_input("down_block_10",b_2784)
            unet_binding.bind_ortvalue_input("down_block_11",b_2785)
            unet_binding.bind_ortvalue_input("mid_block_additional_residual",mid_block_res_sample)
            unet_binding.bind_ortvalue_output("out_sample",out_sample)
            
        avg_it_s=0
        with self.progress_bar(total = num_inference_steps) as progress_bar:
            for i, t in enumerate(timesteps):
                start_iter=time.time()
                # expand the latents if we are doing classifier free guidance
                latent_model_input = np.concatenate([latents] * 2) if do_classifier_free_guidance else latents
                latent_model_input = self.scheduler.scale_model_input(torch.from_numpy(latent_model_input), t)
                latent_model_input = latent_model_input.numpy()
 
                timestep = np.array([t], dtype=timestep_dtype)
                # compute the percentage of total steps we are at
                current_sampling_percent = i / len(timesteps)

                if (
                        current_sampling_percent < controlnet_guidance_start
                        or current_sampling_percent > controlnet_guidance_end 
                    ):  
                    raise ValueError("current sampling percent is out of controlnet_guidance_start and controlnet_guidance_end range")

                if use_io_binding:
                    timestep = ort.OrtValue.ortvalue_from_numpy(timestep,binding_device)
                    sample   = ort.OrtValue.ortvalue_from_numpy(latent_model_input,binding_device)

                    controlnet_binding.bind_ortvalue_input("timestep",timestep)
                    controlnet_binding.bind_ortvalue_input("sample",sample)
                    unet_binding.bind_ortvalue_input("timestep",timestep)
                    unet_binding.bind_ortvalue_input("sample",sample)

                    controlnet_binding.synchronize_inputs()
                    self.controlnet.model.run_with_iobinding(controlnet_binding)
                    controlnet_binding.synchronize_outputs()

                    unet_binding.synchronize_inputs()
                    self.unet.model.run_with_iobinding(unet_binding)
                    unet_binding.synchronize_outputs()
                    noise_pred = unet_binding.copy_outputs_to_cpu()[0]

                else:
                    blocksamples = self.controlnet(
                        sample=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        controlnet_cond=controlnet_cond,
                        conditioning_scale=conditioning_scale
                    )

                    mid_block_res_sample   = blocksamples[12]
                    down_block_res_samples = blocksamples[0:12]

                    down_block_res_samples = [
                        down_block_res_sample * controlnet_conditioning_scale
                        for down_block_res_sample in down_block_res_samples
                    ]
                    mid_block_res_sample *= controlnet_conditioning_scale

                    # predict the noise residual
                    out_sample = self.unet(
                        sample = latent_model_input,
                        timestep = timestep,
                        encoder_hidden_states = prompt_embeds,
                        down_block_0 = down_block_res_samples[0],
                        down_block_1 = down_block_res_samples[1],
                        down_block_2 = down_block_res_samples[2],
                        down_block_3 = down_block_res_samples[3],
                        down_block_4 = down_block_res_samples[4],
                        down_block_5 = down_block_res_samples[5],
                        down_block_6 = down_block_res_samples[6],
                        down_block_7 = down_block_res_samples[7],
                        down_block_8 = down_block_res_samples[8],
                        down_block_9 = down_block_res_samples[9],
                        down_block_10 = down_block_res_samples[10],
                        down_block_11 = down_block_res_samples[11],
                        mid_block_additional_residual = mid_block_res_sample
                    )
                    noise_pred = out_sample[0]

                # perform guidance
                if do_classifier_free_guidance:
                    noise_pred_uncond, noise_pred_text = np.split(noise_pred, 2)
                    noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

                # compute the previous noisy sample x_t -> x_t-1
                scheduler_output = self.scheduler.step(
                    torch.from_numpy(noise_pred), t, torch.from_numpy(latents), **extra_step_kwargs)
                latents = scheduler_output.prev_sample.numpy()

                # call the callback, if provided
                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()
                    if callback is not None and i % callback_steps == 0:
                        callback(i, t, latents)

                it_s=1/(time.time()-start_iter)
                avg_it_s+=it_s
        
        
        average_it_s=avg_it_s/num_inference_steps
        print(f"average iteration/sec: {average_it_s}")
        #Encode the images to the latent space with magical scaling number. more info about the magic number https://github.com/huggingface/diffusers/issues/437#issuecomment-1241827515
        latents = 1 / 0.18215 * latents

        # it seems likes there is a strange result for using half-precision vae decoder if batchsize>1
        image = np.concatenate(
            [self.vae_decoder(latent_sample=latents[i : i + 1])[0] for i in range(latents.shape[0])]
        )
       
        image = np.clip(image / 2 + 0.5, 0, 1)
        
        image = image.transpose((0,2, 3, 1))
        

        if self.safety_checker is not None:
            safety_checker_input = self.feature_extractor(
                self.numpy_to_pil(image), return_tensors="np"
            ).pixel_values.astype(image.dtype)

            images, has_nsfw_concept = [], []
            for i in range(image.shape[0]):
                image_i, has_nsfw_concept_i = self.safety_checker(
                    clip_input=safety_checker_input[i : i + 1], images=image[i : i + 1]
                )
                images.append(image_i)
                has_nsfw_concept.append(has_nsfw_concept_i[0])
            image = np.concatenate(images)
        else:
            has_nsfw_concept = None

        if output_type == "pil":
            image = self.numpy_to_pil(image)

        if not return_dict:
            return (image, has_nsfw_concept)

        return StableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
