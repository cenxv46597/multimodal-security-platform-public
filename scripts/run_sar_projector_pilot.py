"""Run one local SAR projector update probe and a fixed development pilot."""

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import random
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from preflight_sar_adaptation import answer_logits, answer_loss, answer_targets, verify_checkpoint
from run_vlm_clean_screen import MODEL, REVISION, ROOT, generate_answer, get_inputs, load_json, sha256

MANIFESTS = ROOT / "resources/manifests"
PROTOCOL = ROOT / "experiments/protocols/sar_projector_pilot_v0.json"
CATEGORY_PROTOCOL = ROOT / "experiments/protocols/sar_category_supervision_v0.json"
PILOT = MANIFESTS / "sar_projector_pilot_results_20261107.json"
OVERLAP_AUDIT = MANIFESTS / "sar_adaptation_near_duplicate_audit_20261107.json"
PLAN = MANIFESTS / "sar_adaptation_data_plan_20261107.json"
ACQUISITION = MANIFESTS / "sar_adaptation_data_acquisition_20261107.json"
RECORDS = ROOT / "data/processed/sar_adaptation_v0/records.jsonl"
RAW = ROOT / "data/raw/sardet_adaptation_v0"


def checked_local_file(entry, suffix):
    path = (ROOT / entry["path"]).resolve()
    if not path.is_relative_to(ROOT / "runs") or path.suffix != suffix or sha256(path) != entry["sha256"]:
        raise ValueError(f"Pilot artifact does not match the saved evidence: {path.name}")
    return path


def verified_partitions(protocol):
    plan_hash = sha256(PLAN)
    acquisition = load_json(ACQUISITION)
    if (plan_hash != protocol["data_plan_sha256"] or acquisition["source_plan_sha256"] != plan_hash
            or acquisition["status"] != "passed"):
        raise ValueError("Acquisition does not pass the fixed data plan")
    records_meta = acquisition["records"]
    if (records_meta["path"] != RECORDS.relative_to(ROOT).as_posix()
            or sha256(RECORDS) != records_meta["sha256"]):
        raise ValueError("Acquired records differ from their verification report")
    plan = load_json(PLAN)
    records = [json.loads(line) for line in RECORDS.read_text(encoding="utf-8").splitlines() if line]
    if len(records) != len(plan["selected"]) or len(records) != records_meta["count"]:
        raise ValueError("Incomplete acquired records")
    file_hashes, pixel_hashes = set(), set()
    for record, selected in zip(records, plan["selected"]):
        for field in ("sample_id", "question", "reference_answer", "proposed_partition"):
            if record[field] != selected[field]:
                raise ValueError(f"Acquisition changed a fixed field: {field}")
        asset = record["asset"]
        path = (ROOT / asset["file"]).resolve()
        if not path.is_relative_to(RAW) or sha256(path) != asset["sha256"]:
            raise ValueError(f"Image verification failed: {record['sample_id']}")
        if asset["sha256"] in file_hashes or asset["pixel_sha256"] in pixel_hashes:
            raise ValueError("Duplicate acquired image")
        file_hashes.add(asset["sha256"])
        pixel_hashes.add(asset["pixel_sha256"])
    labels = plan["labels"]
    partitions = []
    for partition, per_class in (("adapt_train", protocol["train_per_class"]),
                                 ("adapt_validation", protocol["validation_per_class"])):
        members = [record for record in records if record["proposed_partition"] == partition]
        if Counter(record["reference_answer"] for record in members) != Counter(dict.fromkeys(labels, per_class)):
            raise ValueError(f"Unexpected class counts in {partition}")
        partitions.append(members)
    if sum(map(len, partitions)) != len(records) or len({row["sample_id"] for row in records}) != len(records):
        raise ValueError("Invalid partition membership")
    return partitions[0], partitions[1], labels


def frozen_digest(parameters):
    digest = hashlib.sha256()
    for name, parameter in parameters:
        if parameter.requires_grad or parameter.grad is not None:
            raise RuntimeError(f"Frozen parameter has gradient enabled or populated: {name}")
        digest.update(name.encode() + b"\0")
        digest.update(memoryview(parameter.detach().cpu().contiguous().numpy()).cast("B"))
    return digest.hexdigest()


def memory_check(protocol):
    import torch

    torch.cuda.synchronize()
    stats = {"peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
             "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
             "free_mib_at_check": torch.cuda.mem_get_info()[0] / 2**20}
    if (stats["peak_reserved_mib"] > protocol["peak_reserved_limit_mib"]
            or stats["free_mib_at_check"] < protocol["minimum_free_mib"]):
        raise RuntimeError(f"GPU memory guard reached: {stats}")
    return stats


def parameter_change(projector, initial):
    import torch

    change = 0.0
    for name, parameter in projector.named_parameters():
        updated = parameter.detach().cpu()
        if not bool(torch.isfinite(updated).all()):
            raise RuntimeError(f"Nonfinite updated parameter: {name}")
        change = max(change, float((updated - initial[name]).abs().max()))
    if change == 0:
        raise RuntimeError("Projector parameters did not change")
    return change


def check_optimizer(optimizer, projector, expected_steps):
    import torch

    for parameter in projector.parameters():
        state = optimizer.state[parameter]
        if int(state["step"].item()) != expected_steps:
            raise RuntimeError("Unexpected AdamW step count")
        for key in ("exp_avg", "exp_avg_sq"):
            if state[key].dtype != torch.float32 or not bool(torch.isfinite(state[key]).all()):
                raise RuntimeError(f"Invalid AdamW state: {key}")


def supervised_loss(model, inputs, target, category):
    if category is None:
        return answer_loss(model, inputs, target), {}
    import torch.nn.functional as F

    first_tokens, label_index, weight = category
    logits = answer_logits(model, inputs, target)
    answer_ce = F.cross_entropy(logits.flatten(0, 1), target.flatten())
    category_ce = F.cross_entropy(logits[:, 0, :].index_select(-1, first_tokens), label_index)
    return answer_ce + weight * category_ce, {
        "answer_cross_entropy": float(answer_ce.detach()),
        "category_cross_entropy": float(category_ce.detach()),
    }


def update(model, inputs, target, optimizer, protocol, category=None):
    import torch

    optimizer.zero_grad(set_to_none=True)
    loss, components = supervised_loss(model, inputs, target, category)
    value = float(loss.detach())
    if not bool(torch.isfinite(loss)):
        raise RuntimeError("Nonfinite training loss")
    loss.backward()
    parameters = list(model.multi_modal_projector.parameters())
    if any(parameter.grad is None for parameter in parameters):
        raise RuntimeError("Missing projector gradient")
    norm = float(torch.nn.utils.clip_grad_norm_(
        parameters, protocol["gradient_clip_norm"], error_if_nonfinite=True))
    if norm <= 0:
        raise RuntimeError("Zero projector gradient")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    return {"loss": value, "gradient_norm_before_clip": norm, **components}


def evaluate(model, processor, records, labels, screen):
    import torch

    predictions = []
    for record in records:
        started = time.perf_counter()
        row = {"sample_id": record["sample_id"]}
        try:
            inputs = get_inputs(processor, record, "cuda", torch.float16)
            answer, tokens = generate_answer(model, processor, inputs, screen["generation"]["max_new_tokens"])
            row.update(status="ok", answer=answer, generated_tokens=tokens)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as error:
            row.update(status="error", answer=None, error=f"{type(error).__name__}: {error}")
        predictions.append({**row, "seconds": time.perf_counter() - started})
    return {"predictions": predictions, "score": score_samples(records, predictions, labels, screen)}


def load_frozen_projector_model(device="cuda"):
    import torch
    from transformers import LlavaForConditionalGeneration

    model, info = LlavaForConditionalGeneration.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_safetensors=True,
        torch_dtype=torch.float16, attn_implementation="eager", output_loading_info=True)
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Checkpoint loading issue: {info}")
    model.eval().requires_grad_(False).to(device)
    model.multi_modal_projector.float()
    model.multi_modal_projector.register_forward_pre_hook(lambda _, args: (args[0].float(),))
    return model


def load_projector_weights(projector, checkpoint):
    import torch
    from safetensors.torch import load_file

    weights = load_file(checkpoint)
    expected = projector.state_dict()
    if (set(weights) != set(expected) or any(
            value.dtype != torch.float32 or expected[name].dtype != torch.float32
            or value.shape != expected[name].shape or not bool(torch.isfinite(value).all())
            for name, value in weights.items())):
        raise ValueError("Unexpected projector tensors")
    projector.load_state_dict(weights, strict=True)


def run(report, persist, output, probe_only, category_supervision):
    import torch
    from safetensors.torch import load_file, save_file
    from transformers import AutoProcessor

    protocol = load_json(PROTOCOL)
    if (protocol["optimizer"], protocol["epochs"], protocol["batch_size"]) != ("AdamW", 1, 1):
        raise ValueError("This pilot implements one pass of single-image AdamW updates")
    train, validation, labels = verified_partitions(protocol)
    screen_path = ROOT / "experiments/protocols/sar_clean_development_screen_v0.json"
    screen_hash = sha256(screen_path)
    original_screen = load_json(MANIFESTS / "sar_clean_development_plan_20261105.json")
    if screen_hash != original_screen["screen_protocol_sha256"]:
        raise ValueError("The original development gate or generation settings changed")
    screen = load_json(screen_path)
    random.Random(protocol["seed"]).shuffle(train)
    torch.manual_seed(protocol["seed"])
    torch.set_num_threads(4)
    weight_hash = verify_checkpoint()["model.safetensors"]
    report.update(protocol=protocol, protocol_sha256=sha256(PROTOCOL),
                  screen_sha256=screen_hash, source_records_sha256=sha256(RECORDS),
                  base_weight_sha256=weight_hash, train_order=[row["sample_id"] for row in train],
                  validation_order=[row["sample_id"] for row in validation])
    category_protocol = None
    standard_run = None
    excluded_ids = set()
    if category_supervision:
        category_protocol = load_json(CATEGORY_PROTOCOL)
        if (category_protocol["base_protocol_sha256"] != report["protocol_sha256"]
                or sha256(PILOT) != category_protocol["standard_pilot_summary_sha256"]
                or sha256(OVERLAP_AUDIT) != category_protocol["overlap_audit_sha256"]):
            raise ValueError("Category protocol differs from the frozen standard pilot")
        audit = load_json(OVERLAP_AUDIT)
        excluded = audit["excluded_validation_ids"]
        excluded_ids = set(excluded)
        if (audit["status"] != "completed" or audit["source_records_sha256"] != report["source_records_sha256"]
                or len(excluded) != len(excluded_ids) or not excluded_ids < set(report["validation_order"])):
            raise ValueError("The overlap audit does not match the fixed validation panel")
        source = load_json(PILOT)["source_run"]
        standard_run = load_json(checked_local_file(source, ".json"))
        fields = ("model_revision", "protocol_sha256", "screen_sha256", "source_records_sha256",
                  "base_weight_sha256", "train_order", "validation_order")
        if standard_run["status"] != "completed" or any(standard_run[key] != report[key] for key in fields):
            raise ValueError("The two pilots use different data, initialization or scoring")
        report.update(category_protocol=category_protocol, category_protocol_sha256=sha256(CATEGORY_PROTOCOL),
                      source_standard_pilot=source,
                      overlap_audit={"path": OVERLAP_AUDIT.relative_to(ROOT).as_posix(),
                                     "sha256": category_protocol["overlap_audit_sha256"],
                                     "excluded_validation_ids": excluded})
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this pilot")
    report["gpu"] = torch.cuda.get_device_name()
    report["physical_mib"] = torch.cuda.get_device_properties(0).total_memory / 2**20
    print(f"Verified {len(train)} training and {len(validation)} validation images. Loading the local model.", flush=True)
    processor = AutoProcessor.from_pretrained(
        MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    model = load_frozen_projector_model()
    projector = model.multi_modal_projector.requires_grad_(True)
    initial = {name: parameter.detach().cpu().clone() for name, parameter in projector.named_parameters()}
    frozen = [(name, parameter) for name, parameter in model.named_parameters()
              if not name.startswith("multi_modal_projector.")]
    report["projector_parameters"] = sum(parameter.numel() for parameter in projector.parameters())
    frozen_before = frozen_digest(frozen)
    report["frozen_initial_sha256"] = frozen_before
    if standard_run is not None and standard_run["frozen_initial_sha256"] != frozen_before:
        raise ValueError("The frozen backbones differ from the standard pilot")

    def new_optimizer():
        return torch.optim.AdamW(projector.parameters(), lr=protocol["learning_rate"],
                                 weight_decay=protocol["weight_decay"], foreach=False, fused=False)

    first = train[0]
    cpu_inputs = get_inputs(processor, first, "cpu", torch.float32)
    targets = {label: target.to("cuda") for label, target in
               answer_targets(processor, cpu_inputs, first["question"], labels).items()}
    category_targets = {}
    if category_protocol is not None:
        token_ids = [targets[label][0, 0].item() for label in labels]
        if len(set(token_ids)) != 6 or processor.tokenizer.eos_token_id in token_ids:
            raise ValueError("The fixed six answers must start with distinct non-EOS tokens")
        tokens = torch.tensor(token_ids, dtype=torch.long, device="cuda")
        indices = torch.arange(len(labels), dtype=torch.long, device="cuda")
        category_targets = {
            label: (tokens, indices[index:index + 1], category_protocol["category_first_token_loss_weight"])
            for index, label in enumerate(labels)
        }
        report["category_first_token_ids"] = dict(zip(labels, token_ids))
    inputs = get_inputs(processor, first, "cuda", torch.float16)
    target = targets[first["reference_answer"]]
    category = category_targets.get(first["reference_answer"])
    optimizer = new_optimizer()
    torch.cuda.reset_peak_memory_stats()
    started = time.perf_counter()
    probe = update(model, inputs, target, optimizer, protocol, category)
    probe.update(sample_id=first["sample_id"], optimizer_steps=1)
    report["probe"] = probe
    stats = memory_check(protocol)
    with torch.no_grad():
        after_loss, after_components = supervised_loss(model, inputs, target, category)
    if not bool(torch.isfinite(after_loss)):
        raise RuntimeError("Nonfinite post-update loss")
    after = float(after_loss)
    check_optimizer(optimizer, projector, 1)
    probe.update(loss_after=after, seconds=time.perf_counter() - started,
                 max_abs_parameter_change=parameter_change(projector, initial), **stats,
                 frozen_sha256=frozen_digest(frozen))
    probe.update({f"{key}_after": value for key, value in after_components.items()})
    if probe["frozen_sha256"] != frozen_before or not after < probe["loss"]:
        raise RuntimeError("Single-update probe failed its frozen-weight or same-image loss check")
    if sha256(MODEL / "model.safetensors") != weight_hash:
        raise RuntimeError("Original model weight file changed")
    probe["passed"] = True
    persist()
    print(f"Optimizer probe passed. Loss {probe['loss']:.5f} -> {after:.5f}, peak {stats['peak_allocated_mib']:.0f} MiB.",
          flush=True)
    if probe_only:
        return

    projector.load_state_dict(initial, strict=True)
    optimizer = new_optimizer()
    if standard_run is None:
        print("Running the unadapted model on the fixed validation images.", flush=True)
        report["before"] = evaluate(model, processor, validation, labels, screen)
    else:
        print("Using the verified unadapted predictions from the standard pilot.", flush=True)
        report["before"] = standard_run["before"]
    persist()
    started = time.perf_counter()
    print("Training one fixed pass over 120 images.", flush=True)
    updates = report["training_updates"] = []
    for number, record in enumerate(train, 1):
        if time.perf_counter() - started >= protocol["training_time_limit_seconds"]:
            raise TimeoutError("Training time limit reached")
        torch.cuda.reset_peak_memory_stats()
        step_start = time.perf_counter()
        inputs = get_inputs(processor, record, "cuda", torch.float16)
        label = record["reference_answer"]
        result = update(model, inputs, targets[label], optimizer, protocol, category_targets.get(label))
        result.update(sample_id=record["sample_id"], step=number)
        updates.append(result)
        result.update(memory_check(protocol))
        result["seconds"] = time.perf_counter() - step_start
        if number == 1 or number % 10 == 0:
            persist()
            print(f"Training {number}/{len(train)}. Loss {result['loss']:.4f}.", flush=True)
    report["training_seconds"] = time.perf_counter() - started
    check_optimizer(optimizer, projector, len(train))
    report["pilot_max_abs_parameter_change"] = parameter_change(projector, initial)
    report["frozen_final_sha256"] = frozen_digest(frozen)
    if report["frozen_final_sha256"] != frozen_before or sha256(MODEL / "model.safetensors") != weight_hash:
        raise RuntimeError("Frozen weights changed")
    checkpoint = output.with_suffix(".safetensors")
    state = {name: parameter.detach().cpu().contiguous() for name, parameter in projector.named_parameters()}
    temporary_checkpoint = checkpoint.with_suffix(".safetensors.part")
    protocol_id = category_protocol["id"] if category_protocol is not None else protocol["id"]
    save_file(state, temporary_checkpoint, metadata={"base_revision": REVISION, "protocol": protocol_id})
    restored = load_file(temporary_checkpoint)
    if set(restored) != set(state) or any(not torch.equal(restored[name], value) for name, value in state.items()):
        raise RuntimeError("Saved projector did not round-trip")
    temporary_checkpoint.replace(checkpoint)
    report["projector_checkpoint"] = {"path": checkpoint.relative_to(ROOT).as_posix(),
                                      "sha256": sha256(checkpoint)}
    persist()
    print("Running the adapted model on the same validation images.", flush=True)
    report["after"] = evaluate(model, processor, validation, labels, screen)
    before = {row["sample_id"]: row["correct"] for row in report["before"]["score"]["sample_results"]}
    after = {row["sample_id"]: row["correct"] for row in report["after"]["score"]["sample_results"]}
    report["paired"] = {
        "both_correct": sum(before[key] and after[key] for key in before),
        "became_correct": sum(not before[key] and after[key] for key in before),
        "became_incorrect": sum(before[key] and not after[key] for key in before),
        "both_incorrect": sum(not before[key] and not after[key] for key in before),
    }
    if standard_run is not None:
        retained = [record for record in validation if record["sample_id"] not in excluded_ids]
        comparison = {"unadapted": standard_run["before"], "standard_supervision": standard_run["after"],
                      "category_supervision": report["after"]}
        report["overlap_sensitivity"] = {}
        for name, result in comparison.items():
            predictions = [row for row in result["predictions"] if row["sample_id"] not in excluded_ids]
            score = score_samples(retained, predictions, labels, screen)
            fields = ("planned_samples", "correct", "accuracy", "valid_answer_fraction", "per_class")
            report["overlap_sensitivity"][name] = {key: score[key] for key in fields}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--probe-only", action="store_true")
    parser.add_argument("--category-supervision", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if (not output.is_relative_to(ROOT / "runs") or output.suffix != ".json"
            or output.exists() or output.with_suffix(".safetensors").exists()):
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "model_revision": REVISION, "probe_only": args.probe_only}
    started = time.perf_counter()

    def persist():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(output)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    try:
        report["versions"] = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pillow")}
        run(report, persist, output, args.probe_only, args.category_supervision)
        report["status"] = "passed" if args.probe_only else "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    report["total_seconds"] = time.perf_counter() - started
    persist()
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["status"] in ("passed", "completed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
