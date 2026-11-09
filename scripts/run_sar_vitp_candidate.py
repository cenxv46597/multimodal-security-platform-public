"""Run the ViTP remote-sensing checkpoint with the frozen InternVL SAR inputs."""

import argparse
from collections import Counter
import datetime
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import time
import traceback

import run_sar_proxy_candidate as internvl
from run_vlm_clean_screen import ROOT, load_json, sha256, timed

PROTOCOL = ROOT / "experiments/protocols/sar_vitp_candidate_v0.json"
ACQUISITION = ROOT / "resources/manifests/sar_vitp_acquisition_20261108.json"
MODEL_ID = "GreatBird/ViTP"
REVISION = "2d26fd26d65870c2daa169a8f319ff499668abe8"
SNAPSHOT = ROOT / "models/ViTP_InternVL_1B_rs" / REVISION[:12]
BASE_ACQUISITION_SHA256 = "9c86b819c61b3227393373df5afb66b514e7a111e451bbebbf1f37d157c70df2"
BASE_RUNNER_SHA256 = "13eaf2c321e5e3986f8b3fcbee78da0f06b5cc1d527e49b379d485bde245a181"
# Canonical names, shapes and storage dtypes of the pinned InternVL weight.
BASE_TENSOR_SCHEMA_SHA256 = "5cfdaefae11a5f114455cb39bcc8c1dfce9d7a1f5534bcba90cd1774cfee3810"


def verify_tensor_schema(weight_path, base_count):
    with weight_path.open("rb") as stream:
        header_size = int.from_bytes(stream.read(8), "little")
        if not 0 < header_size <= 2**20:
            raise ValueError("Unexpected safetensors header size")
        header = json.loads(stream.read(header_size))
    schema = {name: [entry["dtype"], entry["shape"]] for name, entry in header.items() if name != "__metadata__"}
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(canonical).hexdigest()
    if len(schema) != base_count or digest != BASE_TENSOR_SCHEMA_SHA256:
        raise ValueError("ViTP tensor names, shapes or dtypes differ from the fixed InternVL schema")
    return {"count": len(schema), "name_shape_dtype_sha256": digest}


def verify_snapshot(candidate, base_reference):
    snapshot = SNAPSHOT.resolve()
    if snapshot != SNAPSHOT or candidate["local_snapshot"] != SNAPSHOT.relative_to(ROOT).as_posix():
        raise ValueError("Unexpected ViTP snapshot path")
    if base_reference != {"path": internvl.ACQUISITION.relative_to(ROOT).as_posix(), "sha256": BASE_ACQUISITION_SHA256}:
        raise ValueError("Unexpected base acquisition evidence")
    base = json.loads(internvl.verified_evidence(base_reference))
    base_files = {row["path"]: row for row in base["files"]}
    data = ACQUISITION.read_bytes()
    acquisition = json.loads(data)
    expected = candidate["cached_inventory"]
    inventory = {row["path"]: row for row in acquisition["files"]}
    if (acquisition["status"] != "passed" or acquisition["model"] != MODEL_ID
            or acquisition["revision"] != REVISION or acquisition["total_files"] != expected["files"]
            or len(inventory) != len(acquisition["files"]) or len(inventory) != expected["files"]
            or acquisition["total_bytes"] != expected["total_bytes"]
            or sum(row["bytes"] for row in inventory.values()) != expected["total_bytes"]
            or inventory.keys() != base_files.keys()):
        raise ValueError("Acquisition inventory does not match the pinned ViTP snapshot")

    weight = candidate["weight"]
    if weight["path"] != "model.safetensors":
        raise ValueError("Unexpected ViTP weight filename")
    files = {}
    for name, entry in inventory.items():
        path = snapshot / name
        if path.resolve() != path or not path.is_file() or path.stat().st_size != entry["bytes"]:
            raise ValueError(f"Missing or invalid fixed snapshot file: {name}")
        digest = sha256(path)
        if digest != entry["sha256"]:
            raise ValueError(f"Snapshot file differs from verified acquisition: {name}")
        if name == weight["path"]:
            if digest != weight["sha256"] or digest != entry["upstream_sha256"] or entry["bytes"] != weight["bytes"]:
                raise ValueError("The ViTP weight differs from the pinned upstream checkpoint")
        elif any(entry[key] != base_files[name][key] for key in ("bytes", "sha256", "upstream_git_blob_sha1")):
            raise ValueError(f"Reused file differs from verified InternVL source: {name}")
        files[name] = {"bytes": entry["bytes"], "sha256": digest}
    tensor_schema = verify_tensor_schema(snapshot / weight["path"], base["safetensors"]["count"])
    return snapshot, files, hashlib.sha256(data).hexdigest(), tensor_schema, base


def compare_reference(report, reference, validation, labels, screen):
    if reference is None:
        return
    path = (ROOT / reference["path"]).resolve()
    if not path.is_relative_to(ROOT / "runs"):
        raise ValueError("Expected the optional reference under runs/")
    if not path.is_file():
        report["internvl_reference"] = {**reference, "status": "unavailable_locally"}
        return
    old = json.loads(internvl.verified_evidence(reference))
    if (reference["model"] != internvl.MODEL_ID or old["status"] != "completed"
            or old["model_id"] != reference["model"] or old["revision"] != reference["revision"]
            or old["runner_sha256"] != BASE_RUNNER_SHA256
            or any(old[key] != report[key] for key in ("panel_order", "execution", "template", "environment"))):
        raise ValueError("The archived InternVL reference used different evaluation inputs or settings")
    before = internvl.score_samples(validation, old["predictions"], labels, screen)
    after = report["evaluation"]
    outcome_names = {(True, True): "both_correct", (False, True): "became_correct",
                     (True, False): "became_incorrect", (False, False): "both_incorrect"}
    counts = Counter(dict.fromkeys(outcome_names.values(), 0))
    pairs = []
    for previous, current in zip(before["sample_results"], after["sample_results"], strict=True):
        outcome = outcome_names[previous["correct"], current["correct"]]
        counts[outcome] += 1
        pairs.append({"sample_id": current["sample_id"], "reference": current["reference"],
                      "internvl_label": previous["parsed_label"], "vitp_label": current["parsed_label"],
                      "outcome": outcome})
    report["internvl_reference"] = {**reference, "status": "paired", "correct": before["correct"],
                                   "per_class": before["per_class"], "outcomes": dict(counts), "pairs": pairs}


def run(args, report, persist):
    shared_path = Path(internvl.__file__)
    shared_hash = sha256(shared_path)
    if shared_hash != BASE_RUNNER_SHA256:
        raise ValueError("The shared inference source differs from the archived InternVL runner")
    report["shared_runner"] = {"path": shared_path.relative_to(ROOT).as_posix(), "sha256": shared_hash}
    protocol = load_json(PROTOCOL)
    candidate = next(row for row in protocol["candidate_order"] if row["model"] == MODEL_ID)
    if candidate["revision"] != REVISION or protocol["fixed_evidence"] != load_json(internvl.PROTOCOL)["fixed_evidence"]:
        raise ValueError("Expected the pinned ViTP revision and unchanged grouped screening evidence")
    report.update(stage="snapshot_verification")
    snapshot, files, acquisition_hash, tensor_schema, base = verify_snapshot(candidate, protocol["base_acquisition"])
    report.update(snapshot=candidate["local_snapshot"], snapshot_files=files,
                  acquisition={"path": ACQUISITION.relative_to(ROOT).as_posix(), "sha256": acquisition_hash},
                  safetensors=tensor_schema,
                  configuration_source={"model": base["model"], "revision": base["revision"],
                                        "manifest_sha256": BASE_ACQUISITION_SHA256,
                                        "scope": "Controlled reuse of verified non-weight InternVL files. "
                                                 "The publisher's ViTP inference configuration is unverified."},
                  stage="panel_verification")

    evidence = protocol["fixed_evidence"]
    task = json.loads(internvl.verified_evidence(evidence["task"]))
    internvl.verified_evidence(evidence["scoring"])
    internvl.verified_evidence(evidence["llava_reference"])
    meta, _, _, validation, _, _, labels, screen = internvl.load_panel(evidence["grouped_plan"]["sha256"])
    if any(meta[key] != evidence["grouped_records"][key] for key in ("path", "sha256")):
        raise ValueError("Grouped records do not match the fixed screen")
    report.update(fixed_evidence=evidence, records=meta, labels=labels,
                  panel_order=[row["sample_id"] for row in validation], stage="environment_check")
    persist()

    os.environ["HF_MODULES_CACHE"] = str(ROOT / ".cache/huggingface/modules" / REVISION)
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
    size, mean, std = internvl.IMAGE_SIZE, internvl.IMAGE_MEAN, internvl.IMAGE_STD
    report["execution"] = {"device": "cuda", "dtype": "float16", "batch_size": 1, "attention": "eager",
                           "tiles_per_image": 1, "tile_size": [size, size], "thumbnail": False,
                           "resize": "PIL RGB bicubic square", "mean": list(mean), "std": list(std),
                           "local_files_only": True, "trust_verified_snapshot_code": True,
                           "use_flash_attn": False, "generation": screen["generation"]}
    if args.phase == "screen":
        internvl.check_probe(args.probe_report, report)
        if load_json(args.probe_report).get("shared_runner") != report["shared_runner"]:
            raise ValueError("The passing probe used a different shared runner")
    report["stage"] = "model_load"
    persist()

    tokenizer = AutoTokenizer.from_pretrained(snapshot, local_files_only=True, trust_remote_code=False, use_fast=False)
    transform = Compose([Resize((size, size), interpolation=InterpolationMode.BICUBIC), ToTensor(), Normalize(mean, std)])
    holder = {}

    def load_model():
        model, info = AutoModel.from_pretrained(
            snapshot, local_files_only=True, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="eager", use_flash_attn=False,
            output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"ViTP weight loading issues: {info}")
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
    inputs, separator, template = internvl.prepare_prompt(model, tokenizer, task["model_question"])
    inputs = {name: tensor.to("cuda") for name, tensor in inputs.items()}
    generation = {**screen["generation"], "eos_token_id": template["eos_token_id"]}
    report.update(template=template, stage="generation")

    def infer(row):
        return internvl.predict(model, tokenizer, inputs, separator, transform, ROOT / row["asset"]["file"], generation)

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
        report["evaluation"] = internvl.score_samples(validation, predictions, labels, screen)
        report["development_gate_passed"] = report["evaluation"]["gate"]["passed"]
    compare_reference(report, protocol.get("paired_reference"), validation, labels, screen)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("probe", "screen"), required=True)
    parser.add_argument("--probe-report", type=Path, help="Passing ViTP probe JSON required for screen")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.phase == "screen") != (args.probe_report is not None):
        parser.error("Supply --probe-report for screen only")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "phase": args.phase, "stage": "protocol_load",
              "model_id": MODEL_ID, "revision": REVISION, "checkpoint_variant": "InternVL 1B remote sensing",
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
