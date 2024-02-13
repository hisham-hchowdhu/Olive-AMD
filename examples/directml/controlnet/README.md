# ControlNet Optimization with DirectML <!-- omit in toc -->

This sample shows how to optimize [Control Net](https://huggingface.co/lllyasviel/sd-controlnet-openpose) with [Stable Diffusion v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) to run with ONNX Runtime and DirectML.

Control Net comprises multiple PyTorch models tied together into a *pipeline*. This Olive sample will convert each PyTorch model to float16 and ONNX, and then run the converted ONNX models through the `OrtTransformersOptimization` pass. The transformer optimization pass performs several time-consuming graph transformations that make the models more efficient for inference at runtime. Output models are only guaranteed to be compatible with onnxruntime-directml 1.16.0 or newer.

**Contents**:
- [Setup](#setup)
- [Conversion to ONNX and Latency Optimization](#conversion-to-onnx-and-latency-optimization)
- [Test Inference](#test-inference)

## Setup

Olive is currently under pre-release, with constant updates and improvements to the functions and usage. This sample code will be frequently updated as Olive evolves, so it is important to install Olive from source when checking out this code from the main branch. See the [README for examples](https://github.com/microsoft/Olive/blob/main/examples/README.md#important) for detailed instructions on how to do this.

**Alternatively**, you may install a stable release that we have validated. For example:

```
# Clone Olive repo to access sample code
git clone https://github.com/microsoft/olive --branch v0.5.0
```

Once you've installed Olive, install the requirements for this sample matching the version of the library you are using:
```
cd olive/examples/directml/stable_diffusion
pip install -e .
pip install -r requirements.txt
```

## Conversion to ONNX and Latency Optimization
The easiest way to optimize the pipeline is with the `controlnet_sd.py` helper script:

```
python controlnet_sd.py --optimize --model_id "runwayml/stable-diffusion-v1-5" --controlnet_id "lllyasviel/sd-controlnet-openpose"
```

The above command will enumerate the `config_<model_name>.json` files and optimize each with Olive, then gather the optimized models into a directory structure suitable for testing inference.

The stable diffusion models are large, and the optimization process is resource intensive. It is recommended to run optimization on a system with a minimum of 16GB of memory (preferably 32GB). Expect optimization to take several minutes (especially the U-Net models).

Once the script successfully completes:
- The optimized ONNX pipeline will be stored under `models/optimized/[model_id]` (for example `models/optimized/runwayml/stable-diffusion-v1-5`).
- The unoptimized ONNX pipeline (models converted to ONNX, but not run through transformer optimization pass) will be stored under `models/unoptimized/[model_id]` (for example `models/unoptimized/runwayml/stable-diffusion-v1-5`).

Re-running the script with `--optimize` will delete the output models, but it will *not* delete the Olive cache. Subsequent runs will complete much faster since it will simply be copying previously optimized models; you may use the `--clean_cache` option to start from scratch (not typically used unless you are modifying the scripts, for example).

## Test Inference
Invoke the script with `--interactive` (and optionally `--num_images <count>`) to present a simple GUI where you may enter a prompt and generate images.

```
python controlnet_sd.py --onnx_model_dir="controlnet\models\optimized\runwayml\stable-diffusion-v1-5" --interactive
Loading models into ORT session...

Inference Batch Start (batch size = 1).
100%|███████████████████████████████████████████████| 50/50 [00:09<00:00, 5.20it/s]
Generated result_0.png
Inference Batch End (1/1 images passed the safety checker).

```

Inference will loop until the generated image passes the safety checker (otherwise you would see black images). The result will be saved as `result_<i>.png` on disk, which is then loaded and displayed in the UI.

Run `python stable_diffusion.py --help` for additional options. A few particularly relevant ones:
- `--model_id <string>` : name of a stable diffusion model ID hosted by huggingface.co. This script has been tested with `runwayml/stable-diffusion-v1-5` (default)
- `--controlnet_id <string>` : name of a controlnet model ID hosted by huggingface.co. This script has been tested with the following:
	- `lllyasviel/sd-controlnet-openpose` (default)
	- `lllyasviel/sd-controlnet-canny` 
- `--control_image_path <string>` : path of controlling image
- `--num_inference_steps <int>` : the number of sampling steps per inference. The default value is 50. A lower value (e.g. 20) will speed up inference at the expensive of quality, and a higher value (e.g. 100) may produce higher quality images.
- `--num_images <int>` : the number of images to generate per script invocation (non-interactive UI) or per click of the generate button (interactive UI). The default value is 1.
- `--batch_size <int>` : the number of images to generate per inference (default of 1). It is typically more efficient to use a larger batch size when producing multiple images than generating a single image at a time; however, larger batches also consume more video memory.

If you omit `--interactive`, the script will generate the requested number of images without displaying a UI and then terminate. Use the `--prompt` option to specify the prompt when using non-interactive mode.

The minimum number of inferences will be `ceil(num_images / batch_size)`; additional inferences may be required of some outputs are flagged by the safety checker to ensure the desired number of outputs are produced.

```
