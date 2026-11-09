"""Check answer supervision and a single projector backward pass, without training."""

import argparse
import datetime
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from run_vlm_clean_screen import MODEL, REVISION, ROOT, get_inputs, load_json, read_records, sha256, timed


def verify_checkpoint():
    inventory = load_json(ROOT / "resources/manifests/llava_interface_resource_inventory_20261105.json")
    if inventory["status"] != "passed" or inventory["revision"] != REVISION:
        raise ValueError("Expected the verified local LLaVA revision")
    checked = {}
    for entry in inventory["files"]:
        path = (ROOT / entry["path"]).resolve()
        if not path.is_relative_to(MODEL) or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Invalid checkpoint file: {path.name}")
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Checkpoint hash mismatch: {path.name}")
        checked[path.name] = digest
    if not checked.get("model.safetensors") or list(MODEL.glob("*.part")):
        raise ValueError("Incomplete checkpoint")
    return checked


def answer_targets(processor, inputs, question, labels):
    import torch

    messages = [{"role": "user", "content": [
        {"type": "image"}, {"type": "text", "text": question},
    ]}]
    prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
    image_token = processor.image_token
    image_tokens = int((inputs["input_ids"] == 151646).sum())
    if prompt.count(image_token) != 1 or image_tokens != 729:
        raise ValueError("Unexpected single-image generation prefix")
    expanded_prompt = prompt.replace(image_token, image_token * image_tokens)
    tokenizer = processor.tokenizer
    prefix = inputs["input_ids"]
    if not torch.equal(tokenizer(expanded_prompt, return_tensors="pt")["input_ids"], prefix):
        raise ValueError("Tokenized generation prefix differs from the screen")

    targets = {}
    eos = tokenizer.eos_token
    if eos is None:
        raise ValueError("Missing checkpoint EOS token")
    for label in labels:
        target = tokenizer(label + eos, add_special_tokens=False, return_tensors="pt")["input_ids"]
        full = tokenizer(expanded_prompt + label + eos, return_tensors="pt")["input_ids"]
        if (target.shape[1] < 2 or target[0, -1].item() != tokenizer.eos_token_id
                or tokenizer.decode(target[0], skip_special_tokens=True) != label
                or not torch.equal(full, torch.cat((prefix, target), dim=1))):
            raise ValueError(f"Answer continuation changes the generation prefix: {label}")
        targets[label] = target
    if not bool(inputs["attention_mask"].all()):
        raise ValueError("Answer supervision expects an unpadded prompt")
    return targets


def answer_logits(model, inputs, target):
    import torch

    batch = {**inputs, "input_ids": torch.cat((inputs["input_ids"], target), dim=1),
             "attention_mask": torch.cat((inputs["attention_mask"], torch.ones_like(target)), dim=1)}
    target_count = target.shape[1]
    # With truncated logits, select each answer token's preceding position explicitly.
    output = model(**batch, use_cache=False, num_logits_to_keep=target_count + 1)
    if output.logits.shape[:2] != (1, target_count + 1):
        raise ValueError("Unexpected causal prediction positions")
    return output.logits[:, :-1].float()


def answer_loss(model, inputs, target):
    import torch.nn.functional as F

    return F.cross_entropy(answer_logits(model, inputs, target).flatten(0, 1), target.flatten())


def prepare_probe(report):
    import torch
    from transformers import AutoProcessor

    records, task, _ = read_records()
    # Use one predetermined engineering probe, never a training or test sample.
    record = records[0]
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    inputs = get_inputs(processor, {"asset": record["asset"], "question": record["question"]},
                        "cpu", torch.float32)
    targets = answer_targets(processor, inputs, record["question"], task["labels"])
    prefix = inputs["input_ids"]
    answer = record["reference_answer"]
    target = targets[answer]
    report["probe"] = {
        "sample_id": record["sample_id"], "image_sha256": record["asset"]["sha256"],
        "sample_role": "fixed first record from the existing development screen; engineering only",
        "prefix_tokens": prefix.shape[1], "image_tokens": int((prefix == 151646).sum()),
        "pixel_shape": list(inputs["pixel_values"].shape),
        "answer_target": answer, "target_ids": target[0].tolist(), "eos_token_id": processor.tokenizer.eos_token_id,
        "all_six_label_prefix_checks_passed": True,
        "target_ids_by_label": {label: ids[0].tolist() for label, ids in targets.items()},
        "supervised_tokens": target.shape[1], "loss_excludes_prompt_and_image_tokens": True,
        "first_answer_predicted_at_zero_based_position": prefix.shape[1] - 1,
        "reference_in_generation_prompt": False,
    }
    return inputs, target


def run_backward(inputs, target, report):
    import torch
    from transformers import LlavaForConditionalGeneration

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for the FP16 preflight; no silent CPU fallback")
    physical_mib = torch.cuda.get_device_properties(0).total_memory / 2**20
    report["gpu"] = {"name": torch.cuda.get_device_name(),
                     "compute_capability": list(torch.cuda.get_device_capability()),
                     "physical_memory_mib": physical_mib,
                     "free_before_load_mib": torch.cuda.mem_get_info()[0] / 2**20}
    holder = {}

    def load_model():
        model, info = LlavaForConditionalGeneration.from_pretrained(
            MODEL, local_files_only=True, trust_remote_code=False, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="eager", output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint loading issue: {info}")
        configs = (model.config, model.config.vision_config, model.config.text_config)
        if any(config._attn_implementation != "eager" for config in configs):
            raise ValueError("This preflight requires eager attention")
        model.eval().requires_grad_(False)
        model.multi_modal_projector.requires_grad_(True)
        model.to("cuda")
        holder["model"] = model
        total = sum(p.numel() for p in model.parameters())
        active = sum(p.numel() for p in model.parameters() if p.requires_grad)
        return {"parameters": total, "parameters_enabled_for_gradient": active,
                "percentage_enabled_for_gradient": 100 * active / total,
                "mode": "eval", "dtype": "float16", "attention": "eager"}

    timed(report, "model_load", load_model, "cuda")
    model = holder["model"]
    active = {name: p for name, p in model.named_parameters() if p.requires_grad}
    if not active or any(not name.startswith("multi_modal_projector.") for name in active):
        raise ValueError("Only the vision-to-language projector may require gradients")
    before = {name: p.detach().cpu().clone() for name, p in active.items()}
    device_inputs = {key: value.to("cuda", dtype=torch.float16 if key == "pixel_values" else value.dtype)
                     for key, value in inputs.items()}
    target = target.to("cuda")

    def backward():
        loss = answer_loss(model, device_inputs, target)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError("Nonfinite answer loss")
        loss.backward()
        gradients = {}
        for name, parameter in active.items():
            gradient = parameter.grad
            if gradient is None or not bool(torch.isfinite(gradient).all()):
                raise RuntimeError(f"Missing or nonfinite projector gradient: {name}")
            gradients[name] = {"elements": gradient.numel(), "nonzero": int(torch.count_nonzero(gradient)),
                               "l2": float(gradient.float().norm())}
        if not any(item["nonzero"] for item in gradients.values()):
            raise RuntimeError("All projector gradients are zero")
        if any(p.grad is not None for p in model.parameters() if not p.requires_grad):
            raise RuntimeError("A frozen parameter received a gradient")
        if device_inputs["pixel_values"].requires_grad or device_inputs["pixel_values"].grad is not None:
            raise RuntimeError("Image pixels must remain fixed in this preflight")
        if any(not torch.equal(p.detach().cpu(), before[name]) for name, p in active.items()):
            raise RuntimeError("The projector changed during the backward-only probe")
        return {"answer_loss": float(loss.detach()), "projector_gradients": gradients,
                "frozen_parameters_with_gradients": 0, "image_requires_gradient": False,
                "projector_values_unchanged": True, "optimizer_created": False, "optimizer_steps": 0,
                "free_after_backward_mib": torch.cuda.mem_get_info()[0] / 2**20}

    timed(report, "backward", backward, "cuda")
    if report["backward"]["peak_reserved_mib"] > physical_mib:
        raise RuntimeError("Backward allocator reservation exceeded physical GPU memory")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "backward"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON output path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    report = {"status": "running", "phase": args.phase, "model_revision": REVISION,
              "started_at": datetime.datetime.now().astimezone().isoformat(timespec="seconds"),
              "scope": "Offline engineering preflight; one sample, no weight update, no accuracy or attack"}
    try:
        report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pillow")}
        import torch

        torch.set_num_threads(4)
        torch.manual_seed(0)
        checked = verify_checkpoint()
        report["checkpoint"] = {"files_verified": len(checked), "weight_sha256": checked["model.safetensors"]}
        print(f"Verified {len(checked)} checkpoint files", flush=True)
        inputs, target = prepare_probe(report)
        print(f"Verified all six answer continuations; probe {report['probe']['sample_id']}", flush=True)
        if args.phase == "backward":
            run_backward(inputs, target, report)
            report["checkpoint"]["weight_file_unchanged"] = (
                sha256(MODEL / "model.safetensors") == checked["model.safetensors"])
            if not report["checkpoint"]["weight_file_unchanged"]:
                raise RuntimeError("The checkpoint file changed during the probe")
        report["status"] = "passed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error),
                      traceback=traceback.format_exc())
        print(report["traceback"], flush=True)
    report["total_seconds"] = time.perf_counter() - started
    with output.open("x", encoding="utf-8") as stream:
        stream.write(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
