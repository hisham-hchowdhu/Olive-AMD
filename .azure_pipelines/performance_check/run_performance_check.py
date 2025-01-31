# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

# The Optimum optimization levels are:
# O1: basic general optimizations.
# O2: basic and extended general optimizations, transformers-specific fusions.
# O3: same as O2 with GELU approximation.
# O4: same as O3 with mixed precision (fp16, GPU-only, requires --device cuda).

import argparse
import ast
import copy
import json
import subprocess
from pathlib import Path

from optimum.exporters.onnx import main_export
from optimum.onnxruntime import ORTOptimizer, ORTQuantizer
from optimum.onnxruntime.configuration import AutoOptimizationConfig, AutoQuantizationConfig
from tabulate import tabulate

from olive.data.template import huggingface_data_config_template
from olive.workflows import run as olive_run

# ruff: noqa: T201

MODEL_NAME_MAP = {
    "bert": "Intel/bert-base-uncased-mrpc",
    "deberta": "microsoft/deberta-base-mnli",
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta_large": "roberta-large-mnli",
}

MODEL_NAME_TO_CONFIG_MAP = {
    "bert": {
        "model_name": "Intel/bert-base-uncased-mrpc",
        "task": "text-classification",
        "dataset": {
            "data_name": "glue",
            "subset": "mrpc",
            "split": "validation",
            "input_cols": ["sentence1", "sentence2"],
            "label_cols": ["label"],
            "batch_size": 1,
            "max_samples": 100,
        },
    },
    "deberta": {
        "model_name": "microsoft/deberta-base-mnli",
        "task": "text-classification",
        "dataset": {
            "data_name": "glue",
            "subset": "mnli_matched",
            "split": "validation",
            "input_cols": ["premise", "hypothesis"],
            "label_cols": ["label"],
            "batch_size": 1,
            "max_samples": 100,
            "component_kwargs": {"pre_process_data": {"align_labels": True}},
        },
    },
    "distilbert": {
        "model_name": "distilbert-base-uncased-finetuned-sst-2-english",
        "task": "text-classification",
        "dataset": {
            "data_name": "glue",
            "subset": "sst2",
            "split": "validation",
            "input_cols": ["sentence"],
            "label_cols": ["label"],
            "batch_size": 1,
            "max_samples": 100,
            "component_kwargs": {"pre_process_data": {"align_labels": True}},
        },
    },
    "roberta_large": {
        "model_name": "roberta-large-mnli",
        "task": "text-classification",
        "dataset": {
            "data_name": "glue",
            "subset": "mnli_matched",
            "split": "validation",
            "input_cols": ["premise", "hypothesis"],
            "label_cols": ["label"],
            "batch_size": 1,
            "max_samples": 100,
            "component_kwargs": {"pre_process_data": {"align_labels": True}},
        },
    },
}

ACC_METRIC = {
    "name": "accuracy",
    "type": "accuracy",
    "backend": "huggingface_metrics",
    "sub_types": [{"name": "accuracy", "priority": 1}],
}

LAT_METRIC = {
    "name": "latency",
    "type": "latency",
    "sub_types": [{"name": "avg", "priority": 2}],
}


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--model_name", type=str, help="The name of the model to run the perf comparison on")
    parser.add_argument("--device", type=str, default="cpu", help="The device to run the perf comparison on")
    parser.add_argument("--test_num", type=int, default=10, help="The number of times to run the perf comparison")

    return parser.parse_args()


def export_onnx(model_name, model_root_path, device="cpu"):
    onnx_model_path = model_root_path / "onnx"
    (
        main_export(model_name, onnx_model_path)
        if device == "cpu"
        else main_export(model_name, onnx_model_path, device="cuda")
    )
    return onnx_model_path


def export_optimum_o1(optimizer, model_root_path):
    o1_model_path = model_root_path / "optimum_o1"
    optimization_config = AutoOptimizationConfig.O1()
    optimizer.optimize(save_dir=o1_model_path, optimization_config=optimization_config)


def export_optimum_o2(optimizer, model_root_path):
    o2_model_path = model_root_path / "optimum_o2"
    optimization_config = AutoOptimizationConfig.O2()
    optimizer.optimize(save_dir=o2_model_path, optimization_config=optimization_config)


def export_optimum_o3(optimizer, model_root_path):
    o3_model_path = model_root_path / "optimum_o3"
    optimization_config = AutoOptimizationConfig.O3()
    optimizer.optimize(save_dir=o3_model_path, optimization_config=optimization_config)


def export_optimum_o4(optimizer, model_root_path):
    o4_model_path = model_root_path / "optimum_o4"
    optimization_config = AutoOptimizationConfig.O4()
    optimizer.optimize(save_dir=o4_model_path, optimization_config=optimization_config)


def export_optimum_dynamic_quantization(onnx_model_path, model_root_path):
    quantizer = ORTQuantizer.from_pretrained(onnx_model_path)
    quantization_model_path = model_root_path / "optimum_dynamic_quantization"
    dqconfig = AutoQuantizationConfig.avx512_vnni(is_static=False, per_channel=False)
    quantizer.quantize(
        save_dir=quantization_model_path,
        quantization_config=dqconfig,
    )


def run_with_config(tool, olive_config, metric_res):
    print(f"Start evaluating {tool} model")
    outputs = olive_run(olive_config)
    if tool == "olive":
        metric = str(next(iter(next(iter(outputs.values())).nodes.values())).metrics.value)
    else:
        metric = str(next(iter(outputs.values())))
    metric_dict = ast.literal_eval(metric)

    for metric_name, metric_value in metric_dict.items():
        if metric_name not in metric_res[tool]:
            metric_res[tool][metric_name] = []
        metric_res[tool][metric_name].append(metric_value)


def run_perf_comparison(cur_dir, model_name, device, model_root_path, test_num):
    print(f"Start running perf comparison on {model_name} model {test_num} times...")
    model_list = ["hf_pytorch", "pytorch_compile", "onnx", "optimum_o1", "optimum_o2", "optimum_o3"]
    if device == "gpu":
        model_list.append("optimum_o4")
    if device == "cpu":
        model_list.append("optimum_dynamic_quantization")
    metric_res = {}
    config_json_path = cur_dir / "configs" / "perf.json"
    for optimized_model in model_list:
        metric_res[f"{optimized_model}"] = {}
    metric_res["olive"] = {}
    for i in range(test_num):
        print(f"Start running {i} time...")
        for optimized_model in model_list:
            accuracy_metric = copy.deepcopy(ACC_METRIC)
            latency_metric = copy.deepcopy(LAT_METRIC)
            print(f"Start evaluating {optimized_model} model")
            with config_json_path.open() as fin:
                olive_config = json.load(fin)
                user_script_path = str(cur_dir / "user_scripts" / f"{model_name}.py")
                hf_model_config = MODEL_NAME_TO_CONFIG_MAP[model_name]
                if optimized_model == "onnx":
                    olive_config["input_model"]["config"]["model_path"] = str(
                        Path(model_root_path / optimized_model / "model.onnx")
                    )
                elif optimized_model == "optimum_dynamic_quantization":
                    olive_config["input_model"]["config"]["model_path"] = str(
                        Path(model_root_path / optimized_model / "model_quantized.onnx")
                    )
                elif optimized_model in ["optimum_o1", "optimum_o2", "optimum_o3", "optimum_o4"]:
                    olive_config["input_model"]["config"]["model_path"] = str(
                        Path(model_root_path / optimized_model / "model_optimized.onnx")
                    )
                elif optimized_model == "hf_pytorch":
                    olive_config["input_model"]["type"] = "PyTorchModel"
                    hf_config = {"hf_config": hf_model_config}
                    olive_config["input_model"]["config"] = hf_config
                elif optimized_model == "pytorch_compile":
                    olive_config["input_model"]["type"] = "PyTorchModel"
                    olive_config["input_model"]["config"]["model_script"] = user_script_path
                    olive_config["input_model"]["config"]["model_loader"] = "torch_complied_model"

                olive_config["systems"]["local_system"]["config"]["accelerators"] = (
                    ["cpu"] if device == "cpu" else ["gpu"]
                )
                olive_config["engine"]["cache_dir"] = str(Path(model_root_path / optimized_model / "cache"))
                olive_config["engine"]["output_dir"] = str(Path(model_root_path / optimized_model / "output"))
                olive_config["engine"]["execution_providers"] = (
                    ["CPUExecutionProvider"] if device == "cpu" else ["CUDAExecutionProvider"]
                )
                olive_config["evaluators"]["common_evaluator"]["metrics"].append(accuracy_metric)
                olive_config["evaluators"]["common_evaluator"]["metrics"].append(latency_metric)
                olive_config["evaluators"]["common_evaluator"]["metrics"][0]["data_config"] = (
                    huggingface_data_config_template(
                        hf_model_config["model_name"], hf_model_config["task"], **hf_model_config["dataset"]
                    )
                )
                olive_config["evaluators"]["common_evaluator"]["metrics"][1]["data_config"] = (
                    huggingface_data_config_template(
                        hf_model_config["model_name"], hf_model_config["task"], **hf_model_config["dataset"]
                    )
                )

            run_with_config(optimized_model, olive_config, metric_res)

        olive_config = f"{model_name}.json" if device == "cpu" else f"{model_name}_gpu.json"
        olive_config_path = cur_dir / "configs" / olive_config
        run_with_config("olive", olive_config_path, metric_res)
    print(f"All metric results {metric_res}")
    for v in metric_res.values():
        for metric_name, metric_value_list in v.items():
            vsum = sum(float(v) for v in metric_value_list)
            v[metric_name] = round((vsum / len(metric_value_list)), 4)
    print(f"Avg metric results {metric_res}")
    return metric_res


def print_perf_table(metric_res, device):
    for key, value in metric_res.items():
        json_value = str(value).replace("'", '"')
        metric_res[key] = ast.literal_eval(json_value)

    columns = [f"tool({device})", *list(metric_res[next(iter(metric_res))].keys())]
    rows = [[key, *list(values.values())] for key, values in metric_res.items()]
    table = tabulate(rows, headers=columns, tablefmt="pipe")
    print(table)


def regression_check(model_name, metrics, device, cpu_info):
    best_metric_path = Path(__file__).absolute().parent / "best_metrics.json"
    if not best_metric_path.exists():
        print(f"Best metrics file {best_metric_path} does not exist, skip regression check")
        return
    metrics_of_interest = ["accuracy-accuracy", "latency-avg"]
    regression_res = {}
    with best_metric_path.open("r") as f:
        best_metric_json = json.load(f)
        best_metrics = best_metric_json[model_name][device]
        for metric_name in metrics_of_interest:
            # There are 4 types of cpus in our cpu pool
            # Intel(R) Xeon(R) Platinum 8272CL CPU @ 2.60GHz
            # Intel(R) Xeon(R) CPU E5-2673 v4 @ 2.30GHz
            # Intel(R) Xeon(R) Platinum 8171M CPU @ 2.60GHz
            # Intel(R) Xeon(R) Platinum 8370C CPU @ 2.80GHz
            # Need to collect the best metrics for each type of cpu
            best_metric = best_metrics[metric_name]
            if device == "cpu" and metric_name == "latency-avg":
                if "8272CL" in cpu_info:
                    best_metric = best_metric["8272CL"]
                elif "E5-2673" in cpu_info:
                    best_metric = best_metric["E5-2673"]
                elif "8171M" in cpu_info:
                    best_metric = best_metric["8171M"]
                elif "8370C" in cpu_info:
                    best_metric = best_metric["8370C"]
                else:
                    print(f"Unknown cpu type {cpu_info}, skip regression check")
                    return
            if best_metric == 0:
                print("No data found, skip regression check")
                return
            no_regression, percentage_change = regression_cal(
                best_metric, metrics[metric_name], metric_name.startswith("accuracy")
            )
            regression_res[metric_name] = {
                "best_metric": best_metric,
                "actual_metric": metrics[metric_name],
                "no_regression": no_regression,
                "percentage_change": percentage_change,
            }
        print(f"Regression check result: {regression_res}")
        for metric_name, metric_value in regression_res.items():
            assert metric_value["no_regression"], (
                f"Regression check failed for {metric_name} metric with {metric_value['percentage_change']}"
                "percentage change"
            )


def regression_cal(best_metric, real_metric, is_acc):
    percentage_change = (real_metric - best_metric) / best_metric
    tolerance = 0.01 if is_acc else 0.05
    diff = real_metric - best_metric
    if is_acc:
        diff = -diff
    no_regression = diff <= tolerance * abs(best_metric)
    return no_regression, percentage_change


def main():
    args = get_args()
    model_name = args.model_name
    model_id = MODEL_NAME_MAP[model_name]
    device = args.device
    test_num = args.test_num

    cur_dir = Path(__file__).absolute().parent
    model_root_path = cur_dir / "run_cache" / model_name
    model_root_path.mkdir(parents=True, exist_ok=True)

    # export the model to onnx
    onnx_model_path = export_onnx(model_id, model_root_path, device)

    optimizer = ORTOptimizer.from_pretrained(onnx_model_path)

    # Optimum optimization
    export_optimum_o1(optimizer, model_root_path)
    export_optimum_o2(optimizer, model_root_path)
    export_optimum_o3(optimizer, model_root_path)
    if device == "gpu":
        export_optimum_o4(optimizer, model_root_path)
    if device == "cpu":
        export_optimum_dynamic_quantization(onnx_model_path, model_root_path)

    metric_res = run_perf_comparison(cur_dir, model_name, device, model_root_path, test_num)
    lscpu = subprocess.check_output(["lscpu"]).decode("utf-8")
    print(lscpu)
    if device == "cpu":
        import psutil

        process = [(proc.name(), proc.cpu_percent()) for proc in psutil.process_iter()]
        print(process)
    elif device == "gpu":
        nvidia_smi = subprocess.check_output(["nvidia-smi"])
        print(nvidia_smi.decode("utf-8"))
    print_perf_table(metric_res, device)
    regression_check(model_name, metric_res["olive"], device, lscpu)


if __name__ == "__main__":
    main()
