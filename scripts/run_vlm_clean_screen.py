"""Offline VLM compatibility phases and the fixed clean development screen."""
import argparse
import datetime
import hashlib
import importlib.metadata
import io
import json
import os
from pathlib import Path
import platform
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "models/llava-interleave-qwen-0.5b-hf/1090956dd1c7"
REVISION = "1090956dd1c79bc93ae98dcf395590369435ec91"
SMOL_MODEL = ROOT / "models/SmolVLM-256M-Instruct/7e3e67edbbed"
SMOL_REVISION = "7e3e67edbbed1bf9888184d9df282b700a323964"
RECORDS = ROOT / "data/processed/sar_clean_development_v0/records.jsonl"
for name, value in {
    "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1", "HF_HUB_DISABLE_TELEMETRY": "1",
    "HF_HOME": str(ROOT / ".cache/huggingface"),
    "HF_HUB_CACHE": str(ROOT / ".cache/huggingface/hub"),
    "TORCH_HOME": str(ROOT / ".cache/torch"), "TOKENIZERS_PARALLELISM": "false",
}.items():
    os.environ[name] = value


def load_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_records():
    task_path = ROOT / "experiments/protocols/sar_single_category_v0.json"
    screen_path = ROOT / "experiments/protocols/sar_clean_development_screen_v0.json"
    plan_path = ROOT / "resources/manifests/sar_clean_development_plan_20261105.json"
    plan = load_json(plan_path)
    acquisition = load_json(ROOT / "resources/manifests/sar_clean_development_acquisition_20261105.json")
    if (acquisition.get("status") != "passed" or sha256(plan_path) != acquisition["source_plan_sha256"] or
            sha256(RECORDS) != acquisition["records"]["sha256"]):
        raise ValueError("Samples or sampling plan differ from acquisition evidence")
    if sha256(task_path) != plan["task_protocol_sha256"] or sha256(screen_path) != plan["screen_protocol_sha256"]:
        raise ValueError("Frozen protocol changed after sampling")
    task, screen = load_json(task_path), load_json(screen_path)
    records = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(records) != 60 or {r["sample_id"] for r in records} != {r["sample_id"] for r in plan["selected"]}:
        raise ValueError("Records differ from the 60-sample plan")
    by_id = {r["sample_id"]: r for r in plan["selected"]}
    for record in records:
        planned = by_id[record["sample_id"]]
        for key in ("question", "reference_answer", "image_name", "author_split"):
            if record[key] != planned[key]:
                raise ValueError(f"Record differs from plan: {key}")
        if record["question"] != task["model_question"] or record["author_split"] != "train":
            raise ValueError("Unexpected model question or source split")
        path = (ROOT / record["asset"]["file"]).resolve()
        if not path.is_relative_to(ROOT / "data/raw/sardet_clean_dev"):
            raise ValueError("Unexpected raw asset path")
        if sha256(path) != record["asset"]["sha256"]:
            raise ValueError("Image content differs from acquired manifest")
    return records, task, screen


def get_inputs(processor, record, device, dtype, model_key="llava"):
    import numpy as np
    import torch
    from PIL import Image

    raw = (ROOT / record["asset"]["file"]).read_bytes()
    with Image.open(io.BytesIO(raw)) as image:
        image.load()
        array = np.asarray(image)
        if array.dtype != np.uint8 or image.mode not in ("L", "RGB") or not np.isfinite(array).all():
            raise ValueError("Unspecified input image representation")
        rgb = image.convert("RGB")
    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": record["question"]},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    try:
        batch = processor(text=prompt, images=[rgb], return_tensors="pt")
    finally:
        rgb.close()
    expected_keys = {"input_ids", "attention_mask", "pixel_values"}
    if model_key == "smol":
        expected_keys.add("pixel_attention_mask")
    if set(batch) != expected_keys:
        raise ValueError(f"Unexpected processor keys: {list(batch)}")
    expected_shape = [1, 3, 384, 384] if model_key == "llava" else [1, 1, 3, 512, 512]
    if list(batch["pixel_values"].shape) != expected_shape:
        raise ValueError("Unexpected vision shape; do not reuse SmolVLM resizing")
    token_id, token_count = (151646, 729) if model_key == "llava" else (49190, 64)
    if int((batch["input_ids"] == token_id).sum()) != token_count:
        raise ValueError("Image feature/token count does not match this checkpoint")
    result = {key: tensor.to(device=device, dtype=dtype if key == "pixel_values" else tensor.dtype)
              for key, tensor in batch.items()}
    if not torch.isfinite(result["pixel_values"]).all():
        raise ValueError("Nonfinite preprocessed image")
    return result


def generate_answer(model, processor, batch, max_new_tokens, model_key="llava"):
    import torch

    extra = {"num_logits_to_keep": 1} if model_key == "llava" else {}
    with torch.inference_mode():
        generated = model.generate(**batch, max_new_tokens=max_new_tokens,
                                   do_sample=False, use_cache=True, return_dict_in_generate=True,
                                   output_logits=True, **extra)
    if not generated.logits or not all(bool(torch.isfinite(value).all()) for value in generated.logits):
        raise RuntimeError("Nonfinite or absent generation logits")
    new = generated.sequences[:, batch["input_ids"].shape[1]:]
    return processor.batch_decode(new, skip_special_tokens=True)[0], int(new.shape[1])


def timed(report, key, function, device):
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
        return result
    finally:
        report[key] = {"seconds": time.perf_counter() - start, **result}
        if device == "cuda":
            report[key].update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                               peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20,
                               memory_scope="PyTorch allocator including resident model, not whole-device usage")
        print(json.dumps({key: report[key]}, ensure_ascii=True), flush=True)


def run(args, report):
    import torch
    from transformers import AutoProcessor, AutoModelForVision2Seq
    from evaluate_sar_clean_screen import score_samples

    torch.set_num_threads(4)
    torch.manual_seed(0)
    dtype = getattr(torch, args.dtype)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; no implicit CPU fallback")
    if args.device == "cpu" and dtype != torch.float32:
        raise ValueError("CPU compatibility uses FP32")
    resource_name = "llava_interface_resource_inventory_20261105.json" if args.model == "llava" else "verification_summary.json"
    model_path = MODEL if args.model == "llava" else SMOL_MODEL
    resource = load_json(ROOT / "resources/manifests" / resource_name)
    if resource.get("status") != "passed":
        raise RuntimeError("Model resource verification is not complete")
    if not (model_path / "model.safetensors").is_file() or (model_path / "model.safetensors.part").exists():
        raise RuntimeError("Model download incomplete")
    report["environment"] = {"python": platform.python_version(),
                             "versions": {name: importlib.metadata.version(name) for name in
                                          ("torch", "transformers", "numpy", "pillow", "safetensors")},
                             "cuda_runtime": torch.version.cuda, "cpu_threads": 4}
    if torch.cuda.is_available():
        report["environment"].update(gpu=torch.cuda.get_device_name(),
                                     compute_capability=list(torch.cuda.get_device_capability()),
                                     compiled_architectures=torch.cuda.get_arch_list(),
                                     physical_gpu_memory_mib=torch.cuda.get_device_properties(0).total_memory / 2**20)
    records, task, screen = read_records()
    processor = AutoProcessor.from_pretrained(model_path, local_files_only=True, trust_remote_code=False, use_fast=False)
    if args.model == "smol":
        processor.image_processor.size = {"longest_edge": 512}
        processor.image_processor.do_image_splitting = False
    # Use the verified saved template/processor settings; no natural-image sample URL.
    inputs = get_inputs(processor, records[0], args.device, dtype, args.model)
    report["probe"] = {"sample_id": records[0]["sample_id"],
                       "input_shapes": {key: list(value.shape) for key, value in inputs.items()},
                       "input_keys": sorted(inputs), "image_tokens": 729 if args.model == "llava" else 64,
                       "ground_truth_or_source_id_in_model_message": False}
    if args.phase == "processor":
        return
    holder = {}

    def load_model():
        model, info = AutoModelForVision2Seq.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=dtype, attn_implementation="eager", output_loading_info=True)
        info = {key: sorted(value) if isinstance(value, set) else value for key, value in info.items()}
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint loading issues: {info}")
        model.eval().requires_grad_(False)
        model.to(args.device)
        attention = {"root": model.config._attn_implementation,
                     "text": model.config.text_config._attn_implementation,
                     "vision": model.config.vision_config._attn_implementation}
        if set(attention.values()) != {"eager"}:
            raise ValueError(f"Unexpected attention: {attention}")
        holder["model"] = model
        return {"parameters": sum(p.numel() for p in model.parameters()),
                "trainable_parameters": sum(p.numel() for p in model.parameters() if p.requires_grad),
                "dtype": str(next(model.parameters()).dtype), "attention": attention,
                "checkpoint_loading_info": info}

    timed(report, "model_load", load_model, args.device)
    model = holder["model"]

    def generate(batch):
        return generate_answer(model, processor, batch, screen["generation"]["max_new_tokens"], args.model)

    if args.phase == "forward":
        def forward():
            with torch.inference_mode():
                extra = {"num_logits_to_keep": 1} if args.model == "llava" else {}
                output = model(**inputs, use_cache=False, **extra)
                if not torch.isfinite(output.logits).all():
                    raise RuntimeError("Nonfinite last-token logits")
                shape = list(output.logits.shape)
                del output
            answer, count = generate(inputs)
            return {"last_token_logits_finite": True, "logits_shape": shape,
                    "answer": answer, "generated_tokens": count,
                    "generation_logits_finite": True, "domain_capability_scored": False}
        timed(report, "forward", forward, args.device)
    elif args.phase == "gradient":
        def gradient():
            # Diagnostic continuation only, not the sample reference label.
            target = processor.tokenizer(" object", add_special_tokens=False, return_tensors="pt")["input_ids"].to(args.device)
            pixels = inputs["pixel_values"].detach().clone().requires_grad_(True)
            batch = {**inputs, "pixel_values": pixels,
                     "input_ids": torch.cat((inputs["input_ids"], target), dim=1),
                     "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(target)), dim=1)}
            # Last T+1 positions contain exactly the T causal predictors for the
            # continuation; discard the last unused next-token prediction.
            extra = {"num_logits_to_keep": target.shape[1] + 1} if args.model == "llava" else {}
            output = model(**batch, use_cache=False, **extra)
            predictors = output.logits[:, -(target.shape[1] + 1):-1]
            loss = torch.nn.functional.cross_entropy(predictors.float().reshape(-1, output.logits.shape[-1]), target.reshape(-1))
            grad, = torch.autograd.grad(loss, pixels)
            finite = bool(torch.isfinite(loss) and torch.isfinite(grad).all())
            nonzero = int(torch.count_nonzero(grad))
            parameter_grads = sum(p.grad is not None for p in model.parameters())
            if not finite or not nonzero or parameter_grads:
                raise RuntimeError("Invalid input gradient")
            return {"diagnostic_continuation": " object", "is_reference_answer": False,
                    "loss": float(loss.detach()), "target_tokens": target.tolist(),
                    "gradient_finite": finite, "nonzero_elements": nonzero, "elements": grad.numel(),
                    "gradient_l2": float(grad.float().norm()), "gradient_abs_max": float(grad.abs().max()),
                    "model_parameters_with_grad": parameter_grads, "weights_updated": False,
                    "gradient_domain": "Preprocessed normalized RGB pixels; not raw SAR pixels",
                    "num_logits_to_keep": extra.get("num_logits_to_keep", 0), "perturbation_generated": False}
        timed(report, "input_gradient", gradient, args.device)
    else:
        predictions = []
        for index, record in enumerate(records, 1):
            start = time.perf_counter()
            try:
                batch = get_inputs(processor, record, args.device, dtype, args.model)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                answer, count = generate(batch)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                prediction = {"sample_id": record["sample_id"], "status": "ok",
                              "answer": answer, "generated_tokens": count, "generation_logits_finite": True}
            except Exception as error:
                prediction = {"sample_id": record["sample_id"], "status": "error",
                              "answer": None, "error": str(error)}
            prediction["seconds_including_preprocessing"] = time.perf_counter() - start
            if args.device == "cuda":
                prediction.update(peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20,
                                  peak_reserved_mib=torch.cuda.max_memory_reserved() / 2**20)
            predictions.append(prediction)
            print(json.dumps({"progress": f"{index}/{len(records)}", **prediction}, ensure_ascii=True), flush=True)
            # Persist incomplete progress too; interrupted calls remain missing.
            report["predictions"] = predictions
            args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        report["evaluation"] = score_samples(records, predictions, task["labels"], screen)
        report["development_gate_passed"] = report["evaluation"]["gate"]["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("llava", "smol"), default="llava")
    parser.add_argument("--phase", choices=("processor", "forward", "gradient", "screen"), required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--dtype", choices=("float32", "float16"), default="float32")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if not args.output.is_relative_to(ROOT / "runs") or args.output.suffix != ".json":
        parser.error("Output must be a JSON path under runs/")
    if args.output.exists():
        parser.error("Preserve existing evidence; select a new output filename")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    report = {"started_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
              "model_id": "llava-hf/llava-interleave-qwen-0.5b-hf" if args.model == "llava" else "HuggingFaceTB/SmolVLM-256M-Instruct",
              "revision": REVISION if args.model == "llava" else SMOL_REVISION,
              "phase": args.phase, "device": args.device, "dtype": args.dtype,
              "status": "running", "scope": "Local offline compatibility/development only; no formal test, training or attack"}
    start = time.perf_counter()
    code = 0
    try:
        run(args, report)
        report["status"] = "completed" if args.phase == "screen" else "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
        code = 1
    finally:
        report["total_seconds"] = time.perf_counter() - start
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(f"{report['status']}: {args.output}", flush=True)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
