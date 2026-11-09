"""Compare the two saved SAR projectors on the grouped development panel."""

import argparse
from collections import Counter
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from plan_sar_adaptation_data import read_verified
from preflight_sar_adaptation import verify_checkpoint
from run_sar_projector_pilot import (
    MANIFESTS, PILOT, PROTOCOL, RECORDS, checked_local_file, evaluate,
    load_frozen_projector_model, load_projector_weights, verified_partitions,
)
from run_vlm_clean_screen import MODEL, REVISION, ROOT, load_json, read_records, sha256

GROUPED_PLAN = MANIFESTS / "sar_grouped_development_plan_20261107.json"
CATEGORY_SUMMARY = MANIFESTS / "sar_category_supervision_results_20261107.json"
SCREEN = ROOT / "experiments/protocols/sar_clean_development_screen_v0.json"


def fixed_input(row):
    return (row["question"], row["reference_answer"], row["asset"]["file"], row["asset"]["sha256"])


def load_panel(plan_sha256):
    plan = json.loads(read_verified(GROUPED_PLAN, plan_sha256))
    original_hash = sha256(RECORDS)
    if plan["status"] != "completed" or plan["original_records_sha256"] != original_hash:
        raise ValueError("The grouped plan does not match the original adaptation records")
    train, original, labels = verified_partitions(load_json(PROTOCOL))
    historical, task, screen = read_records()
    if plan["labels"] != labels or task["labels"] != labels or len(labels) != 6:
        raise ValueError("The grouped plan changed the fixed vocabulary")
    meta = plan["records"]
    path = (ROOT / meta["path"]).resolve()
    if not path.is_relative_to(ROOT / "data/processed") or path.suffix != ".jsonl":
        raise ValueError("Expected the grouped records JSONL under data/processed")
    rows = [json.loads(line) for line in read_verified(path, meta["sha256"]).splitlines() if line.strip()]
    if (len(rows) != 180 or meta["count"] != 180
            or len({row["sample_id"] for row in rows}) != 180
            or any(not isinstance(row["observed_source_group"], str) or not row["observed_source_group"].strip()
                   for row in rows)):
        raise ValueError("Expected 180 distinct rows with observed source groups")
    grouped_train = [row for row in rows if row["grouped_partition"] == "train"]
    validation = [row for row in rows if row["grouped_partition"] == "validation"]
    if (len(grouped_train) != 120 or len(validation) != 60
            or Counter(row["reference_answer"] for row in validation) != Counter(dict.fromkeys(labels, 10))):
        raise ValueError("Unexpected grouped partition sizes or validation class counts")
    train_by_id = {row["sample_id"]: row for row in train}
    original_by_id = {row["sample_id"]: row for row in original}
    historical_by_id = {row["sample_id"]: row for row in historical}
    if {row["sample_id"] for row in grouped_train} != set(train_by_id):
        raise ValueError("The grouped plan changed the fixed training images")
    for row in grouped_train:
        if fixed_input(row) != fixed_input(train_by_id[row["sample_id"]]):
            raise ValueError(f"The grouped plan changed a training record: {row['sample_id']}")
    retained, replacements = [], []
    for row in validation:
        key = row["sample_id"]
        if key in original_by_id:
            source = original_by_id[key]
            retained.append(row)
        else:
            source = historical_by_id.get(key)
            replacements.append(row)
        if source is None or fixed_input(row) != fixed_input(source) or row["question"] != task["model_question"]:
            raise ValueError(f"The grouped validation input does not match fixed evidence: {key}")
    train_groups = {row["observed_source_group"] for row in grouped_train}
    validation_groups = {row["observed_source_group"] for row in validation}
    if len(validation_groups) != len(validation) or train_groups & validation_groups:
        raise ValueError("Observed groups overlap or are repeated in validation")
    return meta, train, original, validation, retained, replacements, labels, screen


def checked_pilot(summary_path, original_ids, original_hash, screen_hash, base_hash):
    summary = load_json(summary_path)
    source = summary["source_run"]
    checkpoint = summary["projector_checkpoint"]
    prior = load_json(checked_local_file(source, ".json"))
    checkpoint_path = checked_local_file(checkpoint, ".safetensors")
    predictions = prior["after"]["predictions"]
    if (summary["status"] != "completed" or prior["status"] != "completed"
            or prior["model_revision"] != REVISION or prior["source_records_sha256"] != original_hash
            or prior["screen_sha256"] != screen_hash or prior["base_weight_sha256"] != base_hash
            or prior["validation_order"] != original_ids or [row["sample_id"] for row in predictions] != original_ids
            or any(prior["projector_checkpoint"][key] != checkpoint[key] for key in ("path", "sha256"))):
        raise ValueError(f"Saved predictions or projector do not match the original evaluation: {summary_path.name}")
    return prior, source, checkpoint, checkpoint_path


def run(report, persist):
    meta, train, original, validation, retained, replacements, labels, screen = load_panel(report["plan_sha256"])
    screen_hash = sha256(SCREEN)
    original_hash = sha256(RECORDS)
    base_hash = verify_checkpoint()["model.safetensors"]
    original_ids = [row["sample_id"] for row in original]
    pilots = {name: checked_pilot(path, original_ids, original_hash, screen_hash, base_hash)
              for name, path in (("standard", PILOT), ("category_supervision", CATEGORY_SUMMARY))}
    standard, category = (pilots[name][0] for name in pilots)
    train_order = standard["train_order"]
    if (len(train_order) != 120 or set(train_order) != {row["sample_id"] for row in train}
            or category["source_standard_pilot"] != pilots["standard"][1]
            or any(category[key] != standard[key] for key in ("train_order", "protocol_sha256", "frozen_final_sha256"))):
        raise ValueError("The saved pilots do not share the same fixed training inputs and backbone")
    retained_ids = {row["sample_id"] for row in retained}
    report.update(records=meta, original_records_sha256=original_hash, screen_sha256=screen_hash,
                  base_weight_sha256=base_hash, model_revision=REVISION, labels=labels,
                  retained_count=len(retained), replacement_count=len(replacements),
                  retained_ids=[row["sample_id"] for row in retained],
                  replacement_ids=[row["sample_id"] for row in replacements],
                  removed_ids=[key for key in original_ids if key not in retained_ids])
    persist()
    print(f"Verified 60 grouped development images: {len(retained)} retained, {len(replacements)} replacements.", flush=True)

    if replacements:
        import torch
        from transformers import AutoProcessor

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required to evaluate the historical replacement images")
        versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pillow")}
        if any(prior["versions"].get(key) != value for prior, *_ in pilots.values() for key, value in versions.items()):
            raise ValueError("The installed evaluation dependencies differ from the original pilots")
        torch.set_num_threads(4)
        report.update(versions=versions, gpu=torch.cuda.get_device_name())
        processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
        model = load_frozen_projector_model()

    evaluations = report["evaluations"] = {}
    for name, (prior, source, checkpoint, checkpoint_path) in pilots.items():
        predictions = {row["sample_id"]: {**row, "prediction_source": "reused_original_validation"}
                       for row in prior["after"]["predictions"] if row["sample_id"] in retained_ids}
        if replacements:
            print(f"Evaluating {len(replacements)} replacements with {name}.", flush=True)
            load_projector_weights(model.multi_modal_projector, checkpoint_path)
            result = evaluate(model, processor, replacements, labels, screen)
            for row in result["predictions"]:
                predictions[row["sample_id"]] = {**row, "prediction_source": "new_replacement_inference"}
        ordered = [predictions[row["sample_id"]] for row in validation]
        score = score_samples(validation, ordered, labels, screen)
        del score["gate"]
        subsets = {}
        for subset_name, members in (("retained", retained), ("replacements", replacements)):
            ids = {row["sample_id"] for row in members}
            subsets[subset_name] = {"samples": len(ids), "correct": sum(
                row["sample_id"] in ids and row["correct"] for row in score["sample_results"])}
        evaluations[name] = {"source_run": source, "projector_checkpoint": checkpoint,
                             "predictions": ordered, "score": score, "subsets": subsets}
        persist()

    standard_results = evaluations["standard"]["score"]["sample_results"]
    category_by_id = {row["sample_id"]: row for row in evaluations["category_supervision"]["score"]["sample_results"]}
    validation_by_id = {row["sample_id"]: row for row in validation}
    pairs = []
    counts = dict.fromkeys(("both_correct", "became_correct", "became_incorrect", "both_incorrect"), 0)
    for standard_row in standard_results:
        key = standard_row["sample_id"]
        category_row = category_by_id[key]
        before, after = standard_row["correct"], category_row["correct"]
        outcome = {(True, True): "both_correct", (False, True): "became_correct",
                   (True, False): "became_incorrect", (False, False): "both_incorrect"}[before, after]
        counts[outcome] += 1
        pairs.append({"sample_id": key, "reference": standard_row["reference"],
                      "observed_source_group": validation_by_id[key]["observed_source_group"],
                      "panel_source": "retained" if key in retained_ids else "replacement",
                      "standard_label": standard_row["parsed_label"], "category_label": category_row["parsed_label"],
                      "standard_correct": before, "category_correct": after, "outcome": outcome})
    report["paired"] = {**counts, "direction": "standard to category supervision", "rows": pairs}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan-sha256", required=True, help="SHA256 of the completed grouped plan")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    plan_hash = args.plan_sha256.lower()
    if len(plan_hash) != 64 or any(char not in "0123456789abcdef" for char in plan_hash):
        parser.error("--plan-sha256 must be a 64-character hexadecimal SHA256")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "plan": GROUPED_PLAN.relative_to(ROOT).as_posix(), "plan_sha256": plan_hash,
        "scope": "Exploratory comparison on a reused development panel separated by observed source groups",
        "scene_lineage": "Original scenes and model pretraining exposure are unresolved",
    }
    started = time.perf_counter()

    def persist():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(output)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    try:
        run(report, persist)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    persist()
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
