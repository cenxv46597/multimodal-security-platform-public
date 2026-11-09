"""Compare the fixed SAR question with an explicit storage tank definition."""

import argparse
import importlib.metadata
import json
from pathlib import Path
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import SCREEN, load_panel
from plan_sar_adaptation_data import read_verified
from preflight_sar_adaptation import verify_checkpoint
from run_sar_projector_pilot import (
    MANIFESTS, checked_local_file, evaluate, load_frozen_projector_model, load_projector_weights,
)
from run_vlm_clean_screen import MODEL, REVISION, ROOT, get_inputs, load_json, sha256

PROTOCOL = ROOT / "experiments/protocols/sar_storage_tank_prompt_v0.json"
GROUPED_RESULT = MANIFESTS / "sar_grouped_development_results_20261107.json"


def tank_counts(score):
    true = score["per_class"]["tank"]
    false_positives = sum(counts["tank"] for label, counts in score["confusion"].items() if label != "tank")
    return {"correct": true["correct"], "support": true["support"], "false_positives": false_positives,
            "tank_predictions": true["correct"] + false_positives}


def run(report, persist):
    import torch
    from transformers import AutoProcessor

    protocol = load_json(PROTOCOL)
    plan_hash = protocol["grouped_plan_sha256"]
    meta, _, _, validation, _, _, labels, screen = load_panel(plan_hash)
    summary = json.loads(read_verified(GROUPED_RESULT, protocol["grouped_result_sha256"]))
    if summary["status"] != "completed" or summary["source_plan"]["sha256"] != plan_hash:
        raise ValueError("The baseline is not from the fixed grouped development plan")
    prior = load_json(checked_local_file(summary["source_run"], ".json"))
    baseline = prior["evaluations"][protocol["model_condition"]]
    checkpoint = checked_local_file(baseline["projector_checkpoint"], ".safetensors")
    ids = [row["sample_id"] for row in validation]
    if (prior["status"] != "completed" or prior["plan_sha256"] != plan_hash
            or prior["model_revision"] != REVISION or prior["records"] != meta
            or prior["screen_sha256"] != sha256(SCREEN)
            or [row["sample_id"] for row in baseline["predictions"]] != ids):
        raise ValueError("The grouped baseline differs from the fixed inputs or scoring settings")
    questions = {row["question"] for row in validation}
    if len(questions) != 1:
        raise ValueError("Expected one original question")
    original = questions.pop()
    anchor = protocol["insert_before"]
    if original.count(anchor) != 1:
        raise ValueError("Could not place the definition in the original question")
    clarified = original.replace(anchor, protocol["definition"] + " " + anchor, 1)
    variants = [{**row, "question": clarified} for row in validation]
    versions = {name: importlib.metadata.version(name) for name in ("torch", "transformers", "pillow")}
    if versions != prior["versions"] or verify_checkpoint()["model.safetensors"] != prior["base_weight_sha256"]:
        raise ValueError("The inference environment or model differs from the baseline")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required to reproduce the fixed inference precision")
    torch.set_num_threads(4)
    report.update(protocol=protocol, protocol_sha256=sha256(PROTOCOL), source_run=summary["source_run"],
                  records=meta, base_weight_sha256=prior["base_weight_sha256"],
                  projector_checkpoint=baseline["projector_checkpoint"],
                  model_revision=REVISION, versions=versions, labels=labels,
                  original_question=original, clarified_question=clarified)
    baseline_score = score_samples(validation, baseline["predictions"], labels, screen)
    baseline_score.pop("gate")
    if baseline_score != baseline["score"]:
        raise ValueError("Could not reproduce the original scores")
    report["original"] = {"score": baseline_score, "tank": tank_counts(baseline_score)}
    persist()

    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    original_inputs = get_inputs(processor, validation[0], "cpu", torch.float32)
    clarified_inputs = get_inputs(processor, variants[0], "cpu", torch.float32)
    if not torch.equal(original_inputs["pixel_values"], clarified_inputs["pixel_values"]):
        raise ValueError("The question edit affected the processed image")
    report["input_check"] = {
        "identical_processed_image": True,
        "original_prompt_tokens": original_inputs["input_ids"].shape[1],
        "clarified_prompt_tokens": clarified_inputs["input_ids"].shape[1],
    }
    model = load_frozen_projector_model()
    load_projector_weights(model.multi_modal_projector, checkpoint)
    print(f"Comparing the clarified question on the same {len(variants)} development images.", flush=True)
    changed = evaluate(model, processor, variants, labels, screen)
    changed["score"].pop("gate")
    changed["tank"] = tank_counts(changed["score"])
    report["clarified"] = changed
    before = baseline_score["sample_results"]
    after = changed["score"]["sample_results"]
    if [(row["sample_id"], row["reference"]) for row in before] != [
            (row["sample_id"], row["reference"]) for row in after]:
        raise ValueError("The two scores use different image/reference pairs")
    report["paired"] = {
        "both_correct": sum(a["correct"] and b["correct"] for a, b in zip(before, after)),
        "became_correct": sum(not a["correct"] and b["correct"] for a, b in zip(before, after)),
        "became_incorrect": sum(a["correct"] and not b["correct"] for a, b in zip(before, after)),
        "both_incorrect": sum(not a["correct"] and not b["correct"] for a, b in zip(before, after)),
        "same_parsed_label": sum(a["parsed_label"] == b["parsed_label"] for a, b in zip(before, after)),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "Prompt wording comparison on reused grouped development images"}
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
