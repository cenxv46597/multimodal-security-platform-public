"""Probe the pinned InternVL snapshot, then score the fixed grouped SAR panel."""

import argparse
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from plan_sar_adaptation_data import read_verified
from run_vlm_clean_screen import ROOT, load_json, sha256, timed

PROTOCOL = ROOT / "experiments/protocols/sar_proxy_selection_v0.json"
ACQUISITION = ROOT / "resources/manifests/sar_internvl_acquisition_20261108.json"
MODEL_ID = "OpenGVLab/InternVL2_5-1B"
IMAGE_SIZE = 448
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)
# Reviewed configuration and executable files from the pinned Hub revision.
REVIEWED_FILES = {
    "config.json": "7e35f990b9383800fac0383daf804e9243c45d88d93221a60fdb2d42a46e75dc",
    "configuration_intern_vit.py": "22dd687a5f142985407675fc6b1dc688c2ded93e54192543ecf28ef495a6efa8",
    "configuration_internvl_chat.py": "6524e6deae6c318543d1022a599f011120f5a970eaa25f1118e7f5d0e346710c",
    "conversation.py": "b1b9f280e9cb9211e519e2b8071002b57cc742aadb2f49b05fa674ee3c7d83ee",
    "modeling_intern_vit.py": "1978e330ea5533dda4d1379096aded271fc741347fc48cb4924846b56a2a6a4b",
    "modeling_internvl_chat.py": "5d8f26ccfeb8434c0bcc295050725add4b2f2b0ffb4d1ab550c877a648386597",
}


def verified_evidence(evidence):
    path = (ROOT / evidence["path"]).resolve()
    if not path.is_relative_to(ROOT):
        raise ValueError("Fixed evidence must be inside the workspace")
    return read_verified(path, evidence["sha256"])


def verify_snapshot(candidate):
    revision = candidate["revision"]
    snapshot = (ROOT / "models" / "InternVL2_5-1B" / revision[:12]).resolve()
    if not snapshot.is_relative_to(ROOT / "models") or snapshot.relative_to(ROOT).as_posix() != candidate["local_snapshot"]:
        raise ValueError("Unexpected InternVL snapshot path")
    inventory_data = ACQUISITION.read_bytes()
    inventory = json.loads(inventory_data)
    expected = candidate["cached_inventory"]
    if (inventory["status"] != "passed" or inventory["model"] != MODEL_ID or inventory["revision"] != revision
            or inventory["total_files"] != expected["files"] or inventory["total_bytes"] != expected["total_bytes"]
            or len(inventory["files"]) != inventory["total_files"]
            or sum(entry["bytes"] for entry in inventory["files"]) != inventory["total_bytes"]):
        raise ValueError("The acquisition inventory does not match the selected revision")
    files = {}
    for entry in inventory["files"]:
        name = entry["path"]
        path = snapshot / name
        if path.resolve().parent != snapshot or name in files or not path.is_file():
            raise ValueError(f"Missing or invalid fixed snapshot file: {name}")
        if path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Snapshot size differs from the fixed inventory: {name}")
        if entry["upstream_sha256"]:
            digest = sha256(path)
            if digest != entry["upstream_sha256"]:
                raise ValueError(f"Snapshot SHA256 differs from the fixed inventory: {name}")
        else:
            data = path.read_bytes()
            git_blob = hashlib.sha1(f"blob {len(data)}\0".encode() + data).hexdigest()
            digest = hashlib.sha256(data).hexdigest()
            if git_blob != entry["upstream_git_blob_sha1"]:
                raise ValueError(f"Snapshot content differs from the reviewed inventory: {name}")
        if digest != entry["sha256"]:
            raise ValueError(f"Snapshot SHA256 differs from the verified acquisition: {name}")
        files[name] = {"bytes": entry["bytes"], "sha256": digest}
    weight = candidate["weight"]
    if files[weight["path"]] != {"bytes": weight["bytes"], "sha256": weight["sha256"]}:
        raise ValueError("The verified snapshot weight does not match the selected weight")
    if any(files.get(name, {}).get("sha256") != digest for name, digest in REVIEWED_FILES.items()):
        raise ValueError("The snapshot's model code or configuration differs from reviewed content")
    return snapshot, files, hashlib.sha256(inventory_data).hexdigest()


def prepare_prompt(model, tokenizer, question):
    # The pinned chat() template, including its separator and visual placeholders.
    template = model.conv_template.copy()
    template.system_message = model.system_message
    template.append_message(template.roles[0], "<image>\n" + question)
    template.append_message(template.roles[1], None)
    rendered = template.get_prompt()
    if rendered.count("<image>") != 1 or model.num_image_token != 256:
        raise ValueError("Expected one InternVL tile and 256 image tokens")
    context = "<IMG_CONTEXT>"
    prompt = rendered.replace("<image>", "<img>" + context * model.num_image_token + "</img>", 1)
    inputs = tokenizer(prompt, return_tensors="pt")
    context_id = tokenizer.convert_tokens_to_ids(context)
    eos_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
    if (context_id in (None, tokenizer.unk_token_id) or eos_id in (None, tokenizer.unk_token_id)
            or int((inputs["input_ids"] == context_id).sum()) != model.num_image_token):
        raise ValueError("The tokenizer does not match the fixed InternVL template")
    model.img_context_token_id = context_id
    metadata = {"name": model.template, "rendered_before_image_token_expansion": rendered,
                "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                "image_context_token_id": context_id, "image_tokens": model.num_image_token,
                "input_tokens": int(inputs["input_ids"].shape[1]), "eos_token_id": eos_id}
    return inputs, template.sep.strip(), metadata


def predict(model, tokenizer, inputs, separator, transform, image_path, generation):
    import torch
    from PIL import Image

    with Image.open(image_path) as image:
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"Unsupported SAR image representation: {image.mode}")
        with image.convert("RGB") as rgb:
            pixels = transform(rgb).unsqueeze(0).to(device="cuda", dtype=torch.float16)
    if list(pixels.shape) != [1, 3, IMAGE_SIZE, IMAGE_SIZE] or not torch.isfinite(pixels).all():
        raise ValueError("Unexpected or nonfinite single-tile pixel tensor")
    with torch.inference_mode():
        generated = model.generate(pixel_values=pixels, **inputs, **generation,
                                   return_dict_in_generate=True, output_logits=True)
    if not generated.logits or not all(bool(torch.isfinite(value).all()) for value in generated.logits):
        raise RuntimeError("Nonfinite or absent generation logits")
    # The pinned implementation supplies inputs_embeds and returns continuation IDs only.
    raw = tokenizer.batch_decode(generated.sequences, skip_special_tokens=True)[0]
    return {"answer": raw.split(separator, 1)[0], "raw_decoded_answer": raw,
            "generated_tokens": int(generated.sequences.shape[1]), "generation_logits_finite": True}


def check_probe(path, report):
    path = path.resolve()
    if not path.is_relative_to(ROOT / "runs") or path.suffix != ".json":
        raise ValueError("Expected a saved JSON probe report under runs/")
    probe = load_json(path)
    if (probe.get("phase") != "probe" or probe.get("status") != "passed"
            or probe.get("probe", {}).get("generation_logits_finite") is not True
            or any(probe.get(key) != report[key] for key in (
                "model_id", "revision", "protocol_sha256", "runner_sha256", "acquisition", "environment", "execution"))):
        raise ValueError("The one-image probe did not pass under this exact configuration")
    report["source_probe"] = {"path": path.relative_to(ROOT).as_posix(), "sha256": sha256(path)}


def run(args, report, persist):
    protocol = load_json(PROTOCOL)
    candidate = next(row for row in protocol["candidate_order"] if row["model"] == MODEL_ID)
    report.update(model_id=MODEL_ID, revision=candidate["revision"], stage="snapshot_verification")
    snapshot, files, acquisition_hash = verify_snapshot(candidate)
    report.update(snapshot=candidate["local_snapshot"], snapshot_files=files,
                  acquisition={"path": ACQUISITION.relative_to(ROOT).as_posix(), "sha256": acquisition_hash},
                  stage="panel_verification")

    evidence = protocol["fixed_evidence"]
    task = json.loads(verified_evidence(evidence["task"]))
    verified_evidence(evidence["scoring"])
    verified_evidence(evidence["llava_reference"])
    meta, _, _, validation, _, _, labels, screen = load_panel(evidence["grouped_plan"]["sha256"])
    if any(meta[key] != evidence["grouped_records"][key] for key in ("path", "sha256")):
        raise ValueError("Grouped records do not match the candidate selection protocol")
    report.update(fixed_evidence=evidence, records=meta, labels=labels,
                  panel_order=[row["sample_id"] for row in validation], stage="environment_check")
    persist()

    # run_vlm_clean_screen pins HF offline mode before Transformers is imported.
    os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules" / candidate["revision"])
    packages = ("torch", "transformers", "torchvision", "timm", "einops", "numpy", "pillow", "safetensors")
    versions = {name: importlib.metadata.version(name) for name in packages}
    import torch
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    from transformers import AutoModel, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the pinned FP16 probe and screen")
    torch.set_num_threads(4)
    torch.manual_seed(0)
    report["environment"] = {"python": platform.python_version(), "versions": versions,
                             "cuda_runtime": torch.version.cuda, "cpu_threads": 4,
                             "gpu": torch.cuda.get_device_name(),
                             "compute_capability": list(torch.cuda.get_device_capability()),
                             "compiled_architectures": torch.cuda.get_arch_list(),
                             "physical_gpu_memory_mib": torch.cuda.get_device_properties(0).total_memory / 2**20}
    report["execution"] = {"device": "cuda", "dtype": "float16", "batch_size": 1, "attention": "eager",
                           "tiles_per_image": 1, "tile_size": [IMAGE_SIZE, IMAGE_SIZE], "thumbnail": False,
                           "resize": "PIL RGB bicubic square", "mean": list(IMAGE_MEAN), "std": list(IMAGE_STD),
                           "local_files_only": True, "trust_verified_snapshot_code": True,
                           "use_flash_attn": False, "generation": screen["generation"]}
    if args.phase == "screen":
        check_probe(args.probe_report, report)
    report["stage"] = "model_load"
    persist()

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=False)
    transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                         ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
    holder = {}

    def load_model():
        model, info = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="eager", use_flash_attn=False,
            output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint loading issues: {info}")
        model.eval().requires_grad_(False).to("cuda")
        attention = {"root": model.config._attn_implementation,
                     "text": model.language_model.config._attn_implementation,
                     "vision": "flash_attention_2" if model.vision_model.config.use_flash_attn else "eager"}
        if set(attention.values()) != {"eager"} or next(model.parameters()).dtype != torch.float16:
            raise ValueError(f"The loaded model did not use FP16 eager attention: {attention}")
        holder["model"] = model
        return {"parameters": sum(p.numel() for p in model.parameters()), "attention": attention,
                "dtype": str(next(model.parameters()).dtype), "weights_frozen": True}

    timed(report, "model_load", load_model, "cuda")
    model = holder["model"]
    inputs, separator, template = prepare_prompt(model, tokenizer, task["model_question"])
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    generation = {**screen["generation"], "eos_token_id": template["eos_token_id"]}
    report.update(template=template, stage="generation")

    def infer(row):
        return predict(model, tokenizer, inputs, separator, transform, ROOT / row["asset"]["file"], generation)

    if args.phase == "probe":
        report["probe_sample_id"] = validation[0]["sample_id"]
        timed(report, "probe", lambda: infer(validation[0]), "cuda")
        return

    predictions = report["predictions"] = []
    try:
        for index, row in enumerate(validation, 1):
            print(f"Grouped development sample {index}/{len(validation)}", flush=True)
            prediction = {"sample_id": row["sample_id"], "status": "ok"}
            try:
                timed(prediction, "generation", lambda: infer(row), "cuda")
            except Exception as error:
                prediction.update(status="error", answer=None, error_type=type(error).__name__, error=str(error))
            prediction.update(prediction.pop("generation", {}))
            predictions.append(prediction)
            persist()
    finally:
        report["evaluation"] = score_samples(validation, predictions, labels, screen)
        report["development_gate_passed"] = report["evaluation"]["gate"]["passed"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "screen"), required=True)
    parser.add_argument("--probe-report", type=Path, help="Passing probe JSON required for screen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.phase == "screen") != (args.probe_report is not None):
        parser.error("Supply --probe-report for screen only")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": args.phase, "stage": "protocol_load",
              "started_at": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
              "protocol": PROTOCOL.relative_to(ROOT).as_posix(), "protocol_sha256": sha256(PROTOCOL),
              "runner_sha256": sha256(Path(__file__)), "scope": "Offline SAR grouped development selection"}
    started = time.perf_counter()

    def persist():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
        temporary.replace(output)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    try:
        run(args, report, persist)
        report.update(status="passed" if args.phase == "probe" else "completed", stage="finished")
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    finally:
        report["total_seconds"] = time.perf_counter() - started
        persist()
    print(f"{report['status']}: {output}", flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
