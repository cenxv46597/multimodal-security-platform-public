"""Score six fixed answer strings and compare matched and swapped SAR images."""

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean, median
import time
import traceback

from evaluate_sar_clean_screen import parse_label
from preflight_sar_adaptation import answer_logits, answer_targets, verify_checkpoint
from run_sar_projector_pilot import (
    PILOT, PROTOCOL, RECORDS, checked_local_file, load_frozen_projector_model, load_projector_weights, verified_partitions,
)
from run_vlm_clean_screen import MODEL, REVISION, ROOT, get_inputs, load_json, sha256

def measure(rows, labels):
    by_class = {label: [row for row in rows if row["reference"] == label] for label in labels}
    per_class = {}
    swapped_correct = 0
    comparisons = []
    for label, members in by_class.items():
        margins = []
        label_margins = []
        eos_margins = []
        swaps = []
        label_swaps = []
        eos_swaps = []
        ranks = []
        for index, row in enumerate(members):
            ref = row["scores"][label]
            ranks.append(row["candidate_ranking"].index(label) + 1)
            strongest_other = next(other for other in row["candidate_ranking"] if other != label)
            alternative = row["scores"][strongest_other]
            margins.append(ref["total"] - alternative["total"])
            label_margins.append(ref["label"] - alternative["label"])
            eos_margins.append(ref["eos"] - alternative["eos"])
            for other in labels:
                if other == label:
                    continue
                # One-to-one within each class pair at the fixed validation ordinal.
                replacement = by_class[other][index]
                wrong = replacement["scores"][label]
                total_delta = ref["total"] - wrong["total"]
                label_delta = ref["label"] - wrong["label"]
                eos_delta = ref["eos"] - wrong["eos"]
                swaps.append(total_delta)
                label_swaps.append(label_delta)
                eos_swaps.append(eos_delta)
                new_correct = replacement["candidate_ranking"][0] == label
                swapped_correct += new_correct
                comparisons.append({
                    "reference_sample": row["sample_id"], "replacement_image": replacement["sample_id"],
                    "reference_class": label, "replacement_class": other,
                    "matched_minus_replacement_logprob": total_delta, "replacement_top1_correct": new_correct,
                })
        per_class[label] = {
            "support": len(members),
            "free_correct": sum(row["free_label"] == label for row in members),
            "candidate_correct": ranks.count(1),
            "rank_histogram": dict(sorted(Counter(ranks).items())),
            "median_reference_rank": median(ranks),
            "mean_margin_over_best_other": mean(margins),
            "mean_label_margin_over_best_other": mean(label_margins),
            "mean_eos_margin_over_best_other": mean(eos_margins),
            "mean_matched_minus_swapped_logprob": mean(swaps),
            "mean_matched_minus_swapped_label_logprob": mean(label_swaps),
            "mean_matched_minus_swapped_eos_logprob": mean(eos_swaps),
            "matched_score_higher_count": sum(delta > 0 for delta in swaps),
            "swaps": len(swaps),
        }
    return {
        "samples": len(rows),
        "free_correct": sum(item["free_correct"] for item in per_class.values()),
        "candidate_correct": sum(item["candidate_correct"] for item in per_class.values()),
        "candidate_agrees_with_free": sum(row["candidate_ranking"][0] == row["free_label"] for row in rows),
        "candidate_correct_free_wrong": sum(row["candidate_ranking"][0] == row["reference"]
                                            and row["free_label"] != row["reference"] for row in rows),
        "candidate_wrong_free_correct": sum(row["candidate_ranking"][0] != row["reference"]
                                            and row["free_label"] == row["reference"] for row in rows),
        "swapped_candidate_correct": swapped_correct, "swap_comparisons": len(comparisons),
        "per_class": per_class, "swaps": comparisons,
    }


def run(report, persist):
    import torch
    from transformers import AutoProcessor

    summary = load_json(PILOT)
    if summary["status"] != "completed" or summary["model_revision"] != REVISION:
        raise ValueError("Expected the completed fixed-model pilot")
    prior_path = checked_local_file(summary["source_run"], ".json")
    checkpoint = checked_local_file(summary["projector_checkpoint"], ".safetensors")
    prior = load_json(prior_path)
    _, records, labels = verified_partitions(load_json(PROTOCOL))
    ids = [row["sample_id"] for row in records]
    previous_predictions = prior["after"]["predictions"]
    if (prior["status"] != "completed" or prior["model_revision"] != REVISION
            or prior["projector_checkpoint"]["sha256"] != summary["projector_checkpoint"]["sha256"]
            or sha256(RECORDS) != prior["source_records_sha256"]
            or ids != prior["validation_order"] or ids != [row["sample_id"] for row in previous_predictions]):
        raise ValueError("The free generation and scoring inputs do not match")
    questions = {row["question"] for row in records}
    if len(questions) != 1 or len(records) != 60 or len(labels) != 6:
        raise ValueError("The image swap requires a common question and the balanced six-class panel")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this fixed FP16 diagnostic")
    torch.set_num_threads(4)
    if verify_checkpoint()["model.safetensors"] != prior["base_weight_sha256"]:
        raise ValueError("Base checkpoint differs from the pilot")
    report.update(source_run=summary["source_run"], source_records_sha256=sha256(RECORDS),
                  projector_checkpoint=summary["projector_checkpoint"], model_revision=REVISION,
                  labels=labels, gpu=torch.cuda.get_device_name())
    print("Verified the same projector, question and 60 validation images.", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    model = load_frozen_projector_model()
    load_projector_weights(model.multi_modal_projector, checkpoint)
    first = get_inputs(processor, records[0], "cpu", torch.float32)
    targets = {label: target.to("cuda") for label, target in
               answer_targets(processor, first, records[0]["question"], labels).items()}
    report["target_ids"] = {label: target[0].tolist() for label, target in targets.items()}
    rows = report["rows"] = []
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, (record, previous) in enumerate(zip(records, previous_predictions), 1):
            inputs = get_inputs(processor, record, "cuda", torch.float16)
            scores = {}
            for label, target in targets.items():
                logits = answer_logits(model, inputs, target)
                token_scores = logits.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1)).flatten()
                if not bool(torch.isfinite(token_scores).all()):
                    raise RuntimeError(f"Nonfinite candidate score: {record['sample_id']}, {label}")
                values = token_scores.tolist()
                scores[label] = {"total": sum(values), "label": sum(values[:-1]), "eos": values[-1],
                                 "token_logprobs": values}
            # Python's stable sort preserves the fixed task label order for exact ties.
            ranking = sorted(labels, key=lambda label: scores[label]["total"], reverse=True)
            rows.append({"sample_id": record["sample_id"], "reference": record["reference_answer"],
                         "free_label": parse_label(previous.get("answer"), labels) if previous["status"] == "ok" else None,
                         "scores": scores, "candidate_ranking": ranking})
            if index % 10 == 0:
                persist()
                print(f"Scored {index}/{len(records)} validation images.", flush=True)
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    report["metrics"] = measure(rows, labels)
    if report["metrics"]["free_correct"] != prior["after"]["score"]["correct"]:
        raise ValueError("Free generation accuracy differs from the original paired evaluation")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {"status": "running", "scope": "Image-level development diagnosis",
              "candidate_score": "Sum of full-vocabulary natural log probabilities of answer tokens and EOS",
              "swap_rule": "Within-class validation ordinal matched to the same ordinal in each of the other five classes"}
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
