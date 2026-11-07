"""Offline interface checks; no training, attack, or domain accuracy claim."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HOME"] = str(ROOT / ".cache/huggingface")
os.environ["HF_HUB_CACHE"] = str(ROOT / ".cache/huggingface/hub")
os.environ["TORCH_HOME"] = str(ROOT / ".cache/torch")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
MODEL = ROOT / "models/SmolVLM-256M-Instruct/7e3e67edbbed"


def timed(report, name, function, device):
    import torch

    if device == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    result = {}
    try:
        result = function()
        if device == "cuda":
            torch.cuda.synchronize()
    finally:
        report[name] = {"seconds": time.perf_counter() - start, **result}
        if device == "cuda":
            report[name].update(
                peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                memory_scope="PyTorch allocator including resident model; not whole-device usage",
            )
        print(json.dumps({name: report[name]}, ensure_ascii=False), flush=True)


def make_inputs(report):
    from PIL import Image
    from transformers import AutoProcessor

    inventory = [json.loads(line) for line in
                 (ROOT / "resources/manifests/resource_inventory.jsonl").read_text(
                     encoding="utf-8").splitlines() if line.strip()]
    entry = min((row for row in inventory
                 if row["resource"] == "SAMPLE_qpm_real" and row["path"].endswith(".png")),
                key=lambda row: row["path"])
    path = ROOT / entry["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != entry["sha256"]:
        raise ValueError("Probe image differs from resource manifest")
    with Image.open(path) as original:
        original.load()
        original_mode = original.mode
        image = original.convert("RGB")
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False)
    # Explicit low-memory diagnostic settings; do not modify downloaded configs.
    processor.image_processor.size = {"longest_edge": 512}
    processor.image_processor.do_image_splitting = False
    messages = [{"role": "user", "content": [
        {"type": "image"},
        {"type": "text", "text": "Describe the visible shapes briefly."},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = processor(text=prompt, images=[image], return_tensors="pt")
    pixel_shape = list(inputs["pixel_values"].shape)
    if pixel_shape != [1, 1, 3, 512, 512]:
        raise ValueError(f"Unexpected diagnostic image shape: {pixel_shape}")
    image_tokens = int((inputs["input_ids"] == 49190).sum())
    if image_tokens != 64:
        raise ValueError(f"Expected 64 image tokens, got {image_tokens}")
    report["input"] = {
        "path": entry["path"], "sha256": digest,
        "original_size": list(image.size), "original_mode": original_mode,
        "prompt": messages[0]["content"][1]["text"],
        "ground_truth_label_supplied": False,
        "formal_evaluation_sample": False,
        "processor_overrides": {"size": {"longest_edge": 512},
                                "do_image_splitting": False},
        "image_tokens": image_tokens,
        "tensor_shapes": {key: list(value.shape) for key, value in inputs.items()},
        "pixel_range": [float(inputs["pixel_values"].min()),
                        float(inputs["pixel_values"].max())],
        "gradient_domain": "processor_output_normalized_pixels_only",
    }
    return processor, inputs


def run(args, report):
    import torch

    torch.set_num_threads(4)
    torch.manual_seed(0)
    report["environment"] = {
        "python": sys.version, "executable": sys.executable,
        "platform": platform.platform(), "torch_cuda_runtime": torch.version.cuda,
        "versions": {name: importlib.metadata.version(name) for name in
                     ("torch", "transformers", "numpy", "pillow", "safetensors", "tokenizers")},
        "cuda_available": torch.cuda.is_available(), "cpu_threads": torch.get_num_threads(),
    }
    if torch.cuda.is_available():
        report["environment"].update(
            gpu_name=torch.cuda.get_device_name(),
            compute_capability=list(torch.cuda.get_device_capability()),
            compiled_architectures=torch.cuda.get_arch_list(),
            total_vram_mib=torch.cuda.get_device_properties(0).total_memory / 2**20,
            native_bf16_supported=torch.cuda.is_bf16_supported(including_emulation=False),
        )
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; no silent CPU fallback")

    if args.phase == "runtime":
        def operators():
            x = torch.randn(1, 3, 32, 32, device=args.device, requires_grad=True)
            kernel = torch.randn(4, 3, 3, 3, device=args.device)
            conv = torch.nn.functional.conv2d(x, kernel)
            matrix = torch.randn(32, 32, device=args.device, requires_grad=True)
            loss = conv.square().mean() + (matrix @ matrix.T).square().mean()
            gradients = torch.autograd.grad(loss, (x, matrix))
            valid = all(bool(torch.isfinite(gradient).all() and torch.count_nonzero(gradient))
                        for gradient in gradients)
            if not valid:
                raise RuntimeError("CUDA/CPU operator gradient is invalid")
            return {"matmul_conv2d_input_gradient_passed": True, "dtype": str(x.dtype)}
        timed(report, "operators", operators, args.device)
        return

    processor, inputs = make_inputs(report)
    if args.phase == "processor":
        return

    from transformers import AutoModelForVision2Seq

    model_holder = {}

    def load():
        model, info = AutoModelForVision2Seq.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False,
            use_safetensors=True, torch_dtype=torch.float32,
            attn_implementation="eager", output_loading_info=True)
        info = {key: sorted(value) if isinstance(value, set) else value
                for key, value in info.items()}
        for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs"):
            if info.get(key):
                raise ValueError(f"Checkpoint loading issue {key}: {info[key]}")
        model.eval().requires_grad_(False)
        model.to(args.device)
        model_holder["model"] = model
        attention = {"root": model.config._attn_implementation,
                     "text": model.config.text_config._attn_implementation,
                     "vision": model.config.vision_config._attn_implementation}
        if set(attention.values()) != {"eager"}:
            raise ValueError(f"Unexpected attention implementation: {attention}")
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "dtype": str(next(model.parameters()).dtype), "attention": attention,
                "checkpoint_loading_info": info}

    timed(report, "model_load", load, args.device)
    model = model_holder["model"]
    inputs = inputs.to(args.device)

    if args.phase == "forward":
        def forward():
            with torch.inference_mode():
                output = model(**inputs, use_cache=False)
                valid = bool(torch.isfinite(output.logits).all())
                shape = list(output.logits.shape)
                if not valid:
                    raise RuntimeError("Nonfinite model logits")
                del output
                generated = model.generate(**inputs, max_new_tokens=12,
                                           do_sample=False, use_cache=True)
            answer = processor.batch_decode(
                generated[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            return {"logits_finite": valid, "logits_shape": shape,
                    "max_new_tokens": 12, "generated_tokens":
                    int(generated.shape[1] - inputs["input_ids"].shape[1]),
                    "answer": answer, "answer_scored_for_domain_ability": False}
        timed(report, "forward", forward, args.device)
    else:
        def backward():
            # A diagnostic continuation, explicitly not a ground-truth answer.
            target = processor.tokenizer(" object.", add_special_tokens=False,
                                         return_tensors="pt")["input_ids"].to(args.device)
            token_ids = torch.cat((inputs["input_ids"], target), dim=1)
            labels = torch.full_like(token_ids, -100)
            labels[:, -target.shape[1]:] = target
            pixels = inputs["pixel_values"].detach().clone().requires_grad_(True)
            model_inputs = dict(inputs)
            model_inputs.update(input_ids=token_ids,
                                attention_mask=torch.cat((inputs["attention_mask"],
                                                          torch.ones_like(target)), dim=1),
                                pixel_values=pixels)
            output = model(**model_inputs, labels=labels, use_cache=False)
            gradient, = torch.autograd.grad(output.loss, pixels)
            finite = bool(torch.isfinite(output.loss) and torch.isfinite(gradient).all())
            nonzero = int(torch.count_nonzero(gradient))
            parameter_gradients = sum(p.grad is not None for p in model.parameters())
            if not finite or not nonzero or parameter_gradients:
                raise RuntimeError("Input gradient check failed")
            return {"loss": float(output.loss.detach()),
                    "diagnostic_continuation": " object.",
                    "continuation_is_ground_truth": False,
                    "target_tokens": target.tolist(), "gradient_finite": finite,
                    "gradient_nonzero_elements": nonzero, "gradient_elements": gradient.numel(),
                    "gradient_l2": float(gradient.norm()),
                    "gradient_abs_max": float(gradient.abs().max()),
                    "model_parameters_with_grad": parameter_gradients,
                    "weights_updated": False, "perturbation_generated": False,
                    "gradient_domain": "processor_output_normalized_pixels_only"}
        timed(report, "input_gradient", backward, args.device)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("runtime", "processor", "forward", "gradient"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"phase": args.phase, "device": args.device, "model_path": str(MODEL),
              "model_revision": "7e3e67edbbed1bf9888184d9df282b700a323964",
              "started_at": datetime.datetime.now(
                  datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
              "random_seed": 0,
              "scope": "interface_diagnostic_only", "status": "running"}
    code = 0
    started = time.perf_counter()
    try:
        run(args, report)
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        code = 1
        print(report["traceback"], file=sys.stderr)
    finally:
        report["total_seconds"] = time.perf_counter() - started
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{report['status']}: {args.output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
