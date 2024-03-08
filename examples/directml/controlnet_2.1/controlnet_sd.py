# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import argparse
import json
import shutil
import sys
import tempfile
import threading
import tkinter as tk
import tkinter.ttk as ttk
import warnings
from pathlib import Path
from typing import Dict

import config
import onnxruntime as ort
import torch
import diffusers
from diffusers import OnnxRuntimeModel, UniPCMultistepScheduler,UNet2DConditionModel,StableDiffusionPipeline, ControlNetModel
from diffusers.utils import load_image
from controlnet_model import OnnxStableDiffusionControlNetPipeline
from packaging import version
from PIL import Image, ImageTk
from user_script import get_base_model_name
from olive.model import ONNXModelHandler
from olive.workflows import run as olive_run
import cv2
from PIL import Image
import numpy as np
from controlnet_aux import OpenposeDetector

# pylint: disable=redefined-outer-name


diffusers.logging.disable_progress_bar()
def prepare_canny(canny_path):
    image= load_image(canny_path)
    image = np.array(image)
    low_threshold = 100
    high_threshold = 200
    image = cv2.Canny(image, low_threshold, high_threshold)
    image = image[:, :, None]
    image = np.concatenate([image, image, image], axis=2)
    canny_image = Image.fromarray(image)
    return canny_image

def prepare_openpose(openpose_path):
    openpose_image = load_image(openpose_path)
    openpose = OpenposeDetector.from_pretrained("lllyasviel/ControlNet")
    openpose_image = openpose(openpose_image)
    return openpose_image

def prepare_custom(image_path,input_size,upscale):
    image = Image.open(image_path).convert("RGB") 
    original_shape = np.array(image).shape
    if original_shape[0] != input_size or original_shape[1] != input_size: 
        raise RuntimeError("not supported input size")  
    image = image.resize(
        (original_shape[1]*upscale, original_shape[0]*upscale),
        Image.BICUBIC
    )
    image = np.array(image)
    image=np.expand_dims(image,axis=0)
    return image


def run_inference_loop(
    use_vae_encoder,
    model_dtype,
    controlnet_image,
    guidance_scale,
    negative_prompt,
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    image_callback=None,
    step_callback=None
    
):
    images_saved = 0
    
    def update_steps(step, timestep, latents):
        if step_callback:
            step_callback((images_saved // batch_size) * num_inference_steps + step)

    #while images_saved < num_images:
    print(f"\nInference Batch Start (batch size = {batch_size}).")

    kwargs = {}
    generator = np.random.RandomState()
    try:
        result = pipeline(
            model_dtype=model_dtype,
            prompt=prompt,
            image=controlnet_image,
            width=image_size,
            height=image_size,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale ,
            generator=generator,
            use_vae_encoder=use_vae_encoder,
            callback=update_steps if step_callback else None,
            **kwargs,
        )
    except:
        pipeline.set_progress_bar_config(disable=True)
        result = pipeline(
            model_dtype=model_dtype,
            prompt=prompt,
            image=controlnet_image,
            width=image_size,
            height=image_size,
            negative_prompt=negative_prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale ,
            generator=generator,
            use_vae_encoder=use_vae_encoder,
            callback=update_steps if step_callback else None,
            **kwargs,
        )

    passed_safety_checker = 0

    for image_index in range(batch_size):
        if result.nsfw_content_detected is None or not result.nsfw_content_detected[image_index]:
            passed_safety_checker += 1
            if images_saved < num_images:
                output_path = f"result_{images_saved}.png"
                result.images[image_index].save(output_path)
                if image_callback:
                    image_callback(images_saved, output_path)
                images_saved += 1
                print(f"Generated {output_path}")

    print(f"Inference Batch End ({passed_safety_checker}/{batch_size} images passed the safety checker).")


def run_inference_gui(
    use_vae_encoder,
    model_dtype,
    controlnet_image,
    guidance_scale,
    negative_prompt,
    pipeline,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps
):
    def update_progress_bar(total_steps_completed):
        progress_bar["value"] = total_steps_completed

    def image_completed(index, path):
        img = Image.open(path)
        photo = ImageTk.PhotoImage(img)
        gui_images[index].config(image=photo)
        gui_images[index].image = photo
        if index == num_images - 1:
            generate_button["state"] = "normal"

    def on_generate_click():
        generate_button["state"] = "disabled"
        progress_bar["value"] = 0
        threading.Thread(
            target=run_inference_loop,
            args=(
                use_vae_encoder,
                model_dtype,
                controlnet_image,
                guidance_scale,
                negative_prompt,
                pipeline,
                prompt_textbox.get(),
                num_images,
                batch_size,
                image_size,
                num_inference_steps,
                image_completed,
                update_progress_bar
            ),
        ).start()

    if num_images > 9:
        print("WARNING: interactive UI only supports displaying up to 9 images")
        num_images = 9

    image_rows = 1 + (num_images - 1) // 3
    image_cols = 2 if num_images == 4 else min(num_images, 3)
    min_batches_required = 1 + (num_images - 1) // batch_size

    bar_height = 10
    button_width = 80
    button_height = 30
    padding = 2
    window_width = image_cols * image_size + (image_cols + 1) * padding
    window_height = image_rows * image_size + (image_rows + 1) * padding + bar_height + button_height

    window = tk.Tk()
    window.title("Stable Diffusion")
    window.resizable(width=False, height=False)
    window.geometry(f"{window_width}x{window_height}")

    gui_images = []
    for row in range(image_rows):
        for col in range(image_cols):
            label = tk.Label(window, width=image_size, height=image_size, background="black")
            gui_images.append(label)
            label.place(x=col * image_size, y=row * image_size)

    y = image_rows * image_size + (image_rows + 1) * padding

    progress_bar = ttk.Progressbar(window, value=0, maximum=num_inference_steps * min_batches_required)
    progress_bar.place(x=0, y=y, height=bar_height, width=window_width)

    y += bar_height

    prompt_textbox = tk.Entry(window)
    prompt_textbox.insert(tk.END, prompt)
    prompt_textbox.place(x=0, y=y, width=window_width - button_width, height=button_height)

    generate_button = tk.Button(window, text="Generate", command=on_generate_click)
    generate_button.place(x=window_width - button_width, y=y, width=button_width, height=button_height)

    window.mainloop()


def run_inference(
    controlnet_image,
    guidance_scale,
    negative_prompt,
    optimized_model_dir,
    provider,
    prompt,
    num_images,
    batch_size,
    image_size,
    num_inference_steps,
    static_dims,
    interactive,
    model_dtype, 
    use_vae_encoder
):
    ort.set_default_logger_severity(3)

    print("Loading models into ORT session...")
    sess_options = ort.SessionOptions()
    sess_options.enable_mem_pattern = False
    
    if static_dims:
        disable_classifier_free_guidance= guidance_scale <= 1.0

        # batch_size is doubled for sample & hidden state because of classifier free guidance:
        # https://github.com/huggingface/diffusers/blob/46c52f9b9607e6ecb29c782c052aea313e6487b7/src/diffusers/pipelines/stable_diffusion/pipeline_stable_diffusion.py#L672
        hidden_batch_size = batch_size if  disable_classifier_free_guidance else batch_size * 2
        
        unet_sample_size=64#image_size // 8
        sess_options.add_free_dimension_override_by_name("unet_sample_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("unet_sample_channels", 4)
        sess_options.add_free_dimension_override_by_name("unet_sample_size",unet_sample_size )
        sess_options.add_free_dimension_override_by_name("unet_time_batch", 1)
        sess_options.add_free_dimension_override_by_name("unet_hidden_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("unet_hidden_sequence", 77)

        sess_options.add_free_dimension_override_by_name("cnet_db0_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db0_height", unet_sample_size)
        sess_options.add_free_dimension_override_by_name("cnet_db0_width", unet_sample_size)

        sess_options.add_free_dimension_override_by_name("cnet_db1_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db1_height", unet_sample_size)
        sess_options.add_free_dimension_override_by_name("cnet_db1_width", unet_sample_size)

        sess_options.add_free_dimension_override_by_name("cnet_db2_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db2_height", unet_sample_size)
        sess_options.add_free_dimension_override_by_name("cnet_db2_width", unet_sample_size)

        sess_options.add_free_dimension_override_by_name("cnet_db3_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db3_height2", unet_sample_size//2)
        sess_options.add_free_dimension_override_by_name("cnet_db3_width2", unet_sample_size//2)

        sess_options.add_free_dimension_override_by_name("cnet_db4_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db4_height2", unet_sample_size//2)
        sess_options.add_free_dimension_override_by_name("cnet_db4_width2", unet_sample_size//2)

        sess_options.add_free_dimension_override_by_name("cnet_db5_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db5_height2", unet_sample_size//2)
        sess_options.add_free_dimension_override_by_name("cnet_db5_width2", unet_sample_size//2)

        sess_options.add_free_dimension_override_by_name("cnet_db6_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db6_height4", unet_sample_size//4)
        sess_options.add_free_dimension_override_by_name("cnet_db6_width4", unet_sample_size//4)

        sess_options.add_free_dimension_override_by_name("cnet_db7_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db7_height4", unet_sample_size//4)
        sess_options.add_free_dimension_override_by_name("cnet_db7_width4", unet_sample_size//4)

        sess_options.add_free_dimension_override_by_name("cnet_db8_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db8_height4", unet_sample_size//4)
        sess_options.add_free_dimension_override_by_name("cnet_db8_width4", unet_sample_size//4)

        sess_options.add_free_dimension_override_by_name("cnet_db9_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db9_height8", unet_sample_size//8)
        sess_options.add_free_dimension_override_by_name("cnet_db9_width8", unet_sample_size//8)

        sess_options.add_free_dimension_override_by_name("cnet_db10_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db10_height8", unet_sample_size//8)
        sess_options.add_free_dimension_override_by_name("cnet_db10_width8", unet_sample_size//8)

        sess_options.add_free_dimension_override_by_name("cnet_db11_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_db11_height8", unet_sample_size//8)
        sess_options.add_free_dimension_override_by_name("cnet_db11_width8", unet_sample_size//8)

        sess_options.add_free_dimension_override_by_name("cnet_mbar_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("cnet_mbar_height8", unet_sample_size//8)
        sess_options.add_free_dimension_override_by_name("cnet_mbar_width8", unet_sample_size//8)

        sess_options.add_free_dimension_override_by_name("condition_batch", hidden_batch_size)
        sess_options.add_free_dimension_override_by_name("condition_height", unet_sample_size)
        sess_options.add_free_dimension_override_by_name("condition_width", unet_sample_size)


    provider_map = {
        "dml": "DmlExecutionProvider",
        "cuda": "CUDAExecutionProvider",
    }
    assert provider in provider_map, f"Unsupported provider: {provider}"

    pipeline = OnnxStableDiffusionControlNetPipeline.from_pretrained(
        optimized_model_dir, sess_options=sess_options, provider=provider_map[provider]
    )  

    if interactive:
        run_inference_gui(
            pipeline=pipeline, 
            prompt=prompt, 
            num_images=num_images, 
            batch_size=batch_size, 
            image_size=image_size, 
            num_inference_steps=num_inference_steps, 
            controlnet_image=controlnet_image,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            model_dtype=model_dtype,
            use_vae_encoder=use_vae_encoder
        )
    else:
        run_inference_loop(
            pipeline=pipeline, 
            prompt=prompt, 
            num_images=num_images, 
            batch_size=batch_size, 
            image_size=image_size, 
            num_inference_steps=num_inference_steps, 
            controlnet_image=controlnet_image,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            model_dtype=model_dtype,
            use_vae_encoder=use_vae_encoder
        )


def update_config_with_provider(config: Dict, provider: str):
    if provider == "dml":
        # DirectML EP is the default, so no need to update config.
        return config
    elif provider == "cuda":
        config["pass_flows"] = [["convert", "optimize_cuda"]]
        config["engine"]["execution_providers"] = ["CUDAExecutionProvider"]
        return config
    else:
        raise ValueError(f"Unsupported provider: {provider}")


def optimize(
    model_id: str,
    provider: str,
    unoptimized_model_dir: Path,
    optimized_model_dir: Path,
    controlnet_dir: str,
    unet_dir: str
):
    from google.protobuf import __version__ as protobuf_version

    # protobuf 4.x aborts with OOM when optimizing unet
    if version.parse(protobuf_version) > version.parse("3.20.3"):
        print("This script requires protobuf 3.20.3. Please ensure your package version matches requirements.txt.")
        sys.exit(1)

    ort.set_default_logger_severity(3)
    script_dir = Path(__file__).resolve().parent
    parent_dir=script_dir.parent

    # Clean up previously optimized models, if any.
    shutil.rmtree(script_dir / "footprints", ignore_errors=True)
    shutil.rmtree(unoptimized_model_dir, ignore_errors=True)
    shutil.rmtree(optimized_model_dir, ignore_errors=True)

    # The model_id and base_model_id are identical when optimizing a standard stable diffusion model like
    # runwayml/stable-diffusion-v1-5. These variables are only different when optimizing a LoRA variant.
    base_model_id = get_base_model_name(model_id)
    #This is for customized Unet and Controlnet models
    controlnet_dir = controlnet_dir if controlnet_dir is not None else base_model_id
    unet_dir = unet_dir if unet_dir is not None else base_model_id

    # Load the entire PyTorch pipeline to ensure all models and their configurations are downloaded and cached.
    # This avoids an issue where the non-ONNX components (tokenizer, scheduler, and feature extractor) are not
    # automatically cached correctly if individual models are fetched one at a time.
    print("Download stable diffusion and Controlnet PyTorch pipeline...")

    controlnet = ControlNetModel.from_pretrained(controlnet_dir, torch_dtype=torch.float32)
    unet = UNet2DConditionModel.from_pretrained(unet_dir, controlnet=controlnet,torch_dtype=torch.float32)
    pipeline=StableDiffusionPipeline.from_pretrained(base_model_id,torch_dtype=torch.float32)
    pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
    config.vae_sample_size = config.vae_sample_size
    config.cross_attention_dim = config.cross_attention_dim
    config.unet_sample_size = config.unet_sample_size

    model_info = {}
    submodel_names = ["controlnet","unet","vae_encoder","vae_decoder","text_encoder"] 

    has_safety_checker = getattr(pipeline, "safety_checker", None) is not None

    if has_safety_checker:
        submodel_names.append("safety_checker")

    for submodel_name in submodel_names:
        print(f"\nOptimizing {submodel_name}")

        olive_config = None

        #borrowing from stable_diffusion folder   
        if submodel_name in ["controlnet","unet"]:
            with (script_dir / f"config_{submodel_name}.json").open() as fin:
                olive_config = json.load(fin)
        else:
            with (parent_dir / 'stable_diffusion' / f"config_{submodel_name}.json").open() as fin:
                olive_config = json.load(fin)

        olive_config = update_config_with_provider(olive_config, provider)

        if submodel_name in ("unet"):
            olive_config["input_model"]["config"]["model_path"] = unet_dir
        elif submodel_name in ("controlnet"):
            olive_config["input_model"]["config"]["model_path"] = controlnet_dir
        
        else:
            # Only the unet & text encoder are affected by LoRA, so it's better to use the base model ID for
            # other models: the Olive cache is based on the JSON config, and two LoRA variants with the same
            # base model ID should be able to reuse previously optimized copies.
            olive_config["input_model"]["config"]["model_path"] = base_model_id

        olive_run(olive_config)

        footprints_file_path = (
            Path(__file__).resolve().parent / "footprints" / f"{submodel_name}_gpu-{provider}_footprints.json"
        )

        with footprints_file_path.open("r") as footprint_file:
            footprints = json.load(footprint_file)

            conversion_footprint = None
            optimizer_footprint = None
            for footprint in footprints.values():
                if footprint["from_pass"] == "OnnxConversion":
                    conversion_footprint = footprint
                elif footprint["from_pass"] == "OrtTransformersOptimization":
                    optimizer_footprint = footprint
           
            assert conversion_footprint and optimizer_footprint
                        
            optimized_olive_model = ONNXModelHandler(**optimizer_footprint["model_config"]["config"])

            unoptimized_olive_model = ONNXModelHandler(**conversion_footprint["model_config"]["config"])

            model_info[submodel_name] = {
                "unoptimized": {
                    "path": Path(unoptimized_olive_model.model_path),
                },
                "optimized": {
                    "path": Path(optimized_olive_model.model_path),
                },
            }

            print(f"Unoptimized Model : {model_info[submodel_name]['unoptimized']['path']}")
            print(f"Optimized Model   : {model_info[submodel_name]['optimized']['path']}")

    # Save the unoptimized models in a directory structure that the diffusers library can load and run.
    # This is optional, and the optimized models can be used directly in a custom pipeline if desired.
    print("\nCreating ONNX pipeline...")

    if has_safety_checker:
        safety_checker = OnnxRuntimeModel.from_pretrained(model_info["safety_checker"]["unoptimized"]["path"].parent)
    else:
        safety_checker = None

    onnx_pipeline = OnnxStableDiffusionControlNetPipeline(
        vae_encoder=OnnxRuntimeModel.from_pretrained(model_info["vae_encoder"]["unoptimized"]["path"].parent),
        vae_decoder=OnnxRuntimeModel.from_pretrained(model_info["vae_decoder"]["unoptimized"]["path"].parent),
        text_encoder=OnnxRuntimeModel.from_pretrained(model_info["text_encoder"]["unoptimized"]["path"].parent),
        tokenizer=pipeline.tokenizer,
        unet=OnnxRuntimeModel.from_pretrained(model_info["unet"]["unoptimized"]["path"].parent),
        controlnet=OnnxRuntimeModel.from_pretrained(model_info["controlnet"]["unoptimized"]["path"].parent),
        scheduler=pipeline.scheduler,
        safety_checker=safety_checker,
        feature_extractor=pipeline.feature_extractor,
        requires_safety_checker=True,
 
    )

    print("Saving unoptimized models...")
    onnx_pipeline.save_pretrained(unoptimized_model_dir)

    # Create a copy of the unoptimized model directory, then overwrite with optimized models from the olive cache.
    print("Copying optimized models...")
    shutil.copytree(unoptimized_model_dir, optimized_model_dir, ignore=shutil.ignore_patterns("weights.pb"))
    for submodel_name in submodel_names:
        src_path = model_info[submodel_name]["optimized"]["path"]
        dst_path = optimized_model_dir / submodel_name / "model.onnx"
        shutil.copyfile(src_path, dst_path)

    print(f"The optimized pipeline is located here: {optimized_model_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", default="stabilityai/stable-diffusion-2-1-base", type=str)
    parser.add_argument("--controlnet_id", default="custom", type=str)
    parser.add_argument("--control_image_path", default="./test_data/0855_128.png" , type=str)
    parser.add_argument("--guidance_scale",default=7.5, type=float, 
                        help=" Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,usually at the expense of lower image quality.between 3-4 is best")
    parser.add_argument("--custom_prompt", action="store_true", help="Use custom prompts which are npy")
    parser.add_argument("--prompt", default="./test_data/pos.npy")
    parser.add_argument("--negative_prompt",default="./test_data/neg.npy" )
    parser.add_argument("--onnx_model_dir", default="./models", type=str, help="path to the optimized onnx olive")
    parser.add_argument("--provider", default="dml", type=str, choices=["dml", "cuda"], help="Execution provider to use")
    parser.add_argument("--num_images", default=1, type=int, help="Number of images to generate")
    parser.add_argument("--batch_size", default=1, type=int, help="Number of images to generate per batch")
    parser.add_argument("--image_size", default=512, type=int, help="Width and height of the images to generate")
    parser.add_argument("--input_size", default=128, type=int, help="Width and height of the controlnet image")
    parser.add_argument("--upscale", default=4, type=int, help="upscale used for the controlnet image")
    parser.add_argument("--disable_classifier_free_guidance", action="store_true",
        help="Whether to disable classifier free guidance. Classifier free guidance should be disabled for turbo models.")
    parser.add_argument("--num_inference_steps", default=50, type=int, help="Number of steps in diffusion process")
    parser.add_argument("--static_dims",action="store_true",help="DEPRECATED (now enabled by default). Use --dynamic_dims to disable static_dims.")
    parser.add_argument("--dynamic_dims", action="store_true", help="Disable static shape optimization")
    parser.add_argument("--tempdir", default=None, type=str, help="Root directory for tempfile directories and files")
    parser.add_argument("--controlnet_dir", default=None, type=str)
    parser.add_argument("--unet_dir",default=None, type=str)
    parser.add_argument("--interactive", action="store_true", help="Run with a GUI")
    parser.add_argument("--optimize", action="store_true", help="Runs the optimization step")
    parser.add_argument("--clean_cache", action="store_true", help="Deletes the Olive cache")
    parser.add_argument("--test_unoptimized", action="store_true", help="Use unoptimized model for inference")
    args = parser.parse_args()

    if args.static_dims:
        print(
            "WARNING: the --static_dims option is deprecated, and static shape optimization is enabled by default. "
            "Use --dynamic_dims to disable static shape optimization."
        )

    if args.provider == "dml" and version.parse(ort.__version__) < version.parse("1.16.0"):
        print("This script requires onnxruntime-directml 1.16.0 or newer")
        sys.exit(1)
    elif args.provider == "cuda" and version.parse(ort.__version__) < version.parse("1.17.0"):
        print("This script requires onnxruntime-gpu 1.17.0 or newer")
        sys.exit(1)

    script_dir = Path(__file__).resolve().parent
    unoptimized_model_dir = script_dir / "models" / "unoptimized" / args.model_id
    optimized_dir_name = "optimized" if args.provider == "dml" else "optimized-cuda"
    optimized_model_dir = script_dir / "models" / optimized_dir_name / args.model_id

    if args.clean_cache:
        shutil.rmtree(script_dir / "cache", ignore_errors=True)

    disable_classifier_free_guidance = args.disable_classifier_free_guidance

    if args.model_id == "stabilityai/sd-turbo" and not disable_classifier_free_guidance:
        disable_classifier_free_guidance = True
        print(
            f"WARNING: Classifier free guidance has been forcefully disabled since {args.model_id} doesn't support it."
        )

    if args.optimize:
        if args.tempdir is not None:
            # set tempdir if specified
            tempdir = Path(args.tempdir).resolve()
            tempdir.mkdir(parents=True, exist_ok=True)
            tempfile.tempdir = str(tempdir)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            optimize(model_id=args.model_id, 
                     provider= args.provider, 
                     unoptimized_model_dir=unoptimized_model_dir, 
                     optimized_model_dir= optimized_model_dir,
                     controlnet_dir=args.controlnet_dir,
                     unet_dir=args.unet_dir)

    if not args.optimize:
        # random seed 

        model_dir =  args.onnx_model_dir
        use_static_dims = not args.dynamic_dims
        use_vae_encoder=False

        if args.controlnet_id=="lllyasviel/sd-controlnet-canny":
            controlnet_image=prepare_canny(args.control_image_path)
        elif args.controlnet_id=="lllyasviel/sd-controlnet-openpose":
            controlnet_image=prepare_openpose(args.control_image_path)
        elif args.controlnet_id=="custom":
            controlnet_image=prepare_custom(args.control_image_path,args.input_size,args.upscale)
            use_vae_encoder=True
        else:
            raise ValueError("We currently only support canny and openpose controlnet")

        print(f"Inference with controlnet - {args.controlnet_id} ")
        
        if disable_classifier_free_guidance and args.guidance_scale > 1.0:
            raise ValueError("if disable_classifier_free_guidance is on, guidance scale should be less <=1.0")
        
        
        if args.test_unoptimized:
            model_dtype="fp32"
        else:
            model_dtype="fp16"

        if args.custom_prompt:
            if not (args.prompt.endswith(".npy") and args.negative_prompt.endswith(".npy")):
                parser.error("Custom prompts require .npy format (e.g., prompt.npy)")
            prompt = np.load(args.prompt).astype(np.float16 if model_dtype == "fp16" else np.float32)
            negative_prompt = np.load(args.negative_prompt).astype(np.float16 if model_dtype == "fp16" else np.float32)
        else:
            prompt = args.prompt
            negative_prompt = args.negative_prompt

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            run_inference(
                controlnet_image=controlnet_image,
                optimized_model_dir=model_dir,
                guidance_scale=args.guidance_scale,
                provider=args.provider,
                prompt=args.prompt,
                num_images=args.num_images,
                batch_size=args.batch_size,
                image_size=args.image_size,
                num_inference_steps=args.num_inference_steps,
                static_dims=use_static_dims,
                interactive=args.interactive,
                negative_prompt=args.negative_prompt,
                model_dtype=model_dtype,
                use_vae_encoder=use_vae_encoder
            )
