"""Strict label scoring for the frozen clean development screen."""
import argparse
from collections import Counter
import json
from pathlib import Path


def parse_label(answer, labels):
    if not isinstance(answer, str):
        return None
    candidate = answer.strip().lower()
    return candidate if candidate in labels else None


def score_samples(samples, predictions, labels, screen):
    if not samples:
        raise ValueError("Cannot evaluate an empty sample list")
    expected = {sample["sample_id"]: sample for sample in samples}
    if len(expected) != len(samples):
        raise ValueError("Duplicate planned sample IDs")
    if any(sample["reference_answer"] not in labels for sample in samples):
        raise ValueError("Reference label is outside the task vocabulary")
    returned = {prediction["sample_id"]: prediction for prediction in predictions}
    if len(returned) != len(predictions):
        raise ValueError("Duplicate prediction IDs")
    if set(returned) - set(expected):
        raise ValueError("Unknown prediction ID")
    confusion = {label: {name: 0 for name in [*labels, "__invalid_or_missing__"]} for label in labels}
    statuses = Counter()
    results = []
    for sample_id, sample in expected.items():
        prediction = returned.get(sample_id)
        if prediction is None:
            label, status = None, "missing"
        elif prediction.get("status") != "ok":
            label, status = None, "call_error"
        else:
            label = parse_label(prediction.get("answer"), labels)
            status = "valid" if label is not None else "invalid_answer"
        reference = sample["reference_answer"]
        correct = label == reference
        statuses[status] += 1
        confusion[reference][label if label is not None else "__invalid_or_missing__"] += 1
        results.append({"sample_id": sample_id, "reference": reference,
                        "parsed_label": label, "status": status, "correct": correct})
    per_class = {}
    for label in labels:
        support = sum(confusion[label].values())
        correct = confusion[label][label]
        per_class[label] = {"support": support, "correct": correct,
                            "recall": correct / support if support else None}
    total = len(samples)
    correct = sum(result["correct"] for result in results)
    accuracy = correct / total
    valid_fraction = statuses["valid"] / total
    complete_classes = all(value["support"] > 0 for value in per_class.values())
    macro_recall = sum(value["recall"] for value in per_class.values()) / len(labels) if complete_classes else None
    gate = screen["development_gate"]
    decisions = {
        "complete_planned_size": total == screen["sampling"]["total"],
        "complete_prediction_accounting": len(returned) == total,
        "class_support": all(value["support"] >= gate["all_six_classes_have_at_least"] for value in per_class.values()),
        "valid_answer_fraction": valid_fraction >= gate["minimum_valid_answer_fraction"],
        "accuracy": accuracy >= gate["minimum_accuracy"],
        "per_class_recall": all(value["recall"] is not None and value["recall"] >= gate["minimum_per_class_recall"] for value in per_class.values()),
    }
    return {"scope": "Exploratory development screen, not a formal test",
            "planned_samples": total, "correct": correct, "accuracy": accuracy,
            "valid_answer_fraction": valid_fraction, "status_counts": dict(statuses),
            "per_class": per_class, "six_class_macro_recall": macro_recall,
            "confusion": confusion, "gate": {"criteria": decisions, "passed": all(decisions.values())},
            "sample_results": results,
            "denominator_rule": "All planned samples including missing, invalid and failed responses"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    from run_vlm_clean_screen import read_records, sha256, RECORDS
    samples, task, screen = read_records()
    if sha256(args.samples) != sha256(RECORDS):
        parser.error("Samples must match the frozen acquired 60-record manifest")
    raw = args.predictions.read_text(encoding="utf-8")
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        predictions = [json.loads(line) for line in raw.splitlines() if line.strip()]
        source_status = "JSONL"
    else:
        if isinstance(decoded, dict) and isinstance(decoded.get("predictions"), list):
            predictions, source_status = decoded["predictions"], decoded.get("status")
        elif isinstance(decoded, list):
            predictions, source_status = decoded, "JSON array"
        else:
            parser.error("Expected prediction JSONL, array, or runner report with predictions")
    result = score_samples(samples, predictions, task["labels"], screen)
    result["source_prediction_status"] = source_status
    output = args.output.resolve()
    if not output.is_relative_to(root / "runs") or output.suffix != ".json":
        parser.error("Output must be a JSON report under project runs/")
    if output.exists():
        parser.error("Preserve prior evidence; choose a new output path")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("planned_samples", "correct", "accuracy", "gate")}))


if __name__ == "__main__":
    main()
