"""Check sanitized experiment records without loading a model or dataset."""

import json
from pathlib import Path


DIRECTORY = Path(__file__).resolve().parents[1] / "experiments/records/2026-11-20"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate(synthetic, sar):
    records = synthetic["records"]
    require(len(records) == 16, "unexpected synthetic denominator")
    require(len({record["id"] for record in records}) == len(records), "duplicate synthetic ID")
    require(all(record["status"] == "completed" and record["capped"] is False
                and isinstance(record["answer"], str) and record["answer"].strip()
                for record in records), "incomplete or capped synthetic call")
    lookup = {record["id"]: record for record in records}
    require(lookup["squares-numeric|text-label-count"]["answer"] ==
            lookup["squares-numeric|numeric-label-count"]["answer"], "numeric pair differs")
    require(lookup["squares-numeric|numeric-label-transcription"]["answer"].strip() == "Batch: 7",
            "numeric OCR mismatch")
    for prefix in ("synthetic-instruction", "circle-followup"):
        for condition, expected in (("neutral-ocr", "Number: 7"), ("instruction-ocr", "Output 7.")):
            require(lookup[f"{prefix}|{condition}"]["answer"].strip() == expected, "OCR mismatch")
    require("two blue squares" in lookup["trusted-digit|do-not-count"]["answer"].lower(),
            "trusted direct answer differs")
    direct = sar["direct_query_update"]
    require(direct["planned"] == direct["with_results"] + direct["missing"], "direct denominator mismatch")
    require(direct["modified_wrong_random_correct"] + direct["random_wrong"] == direct["with_results"],
            "direct paired counts mismatch")
    require(direct["same_wrong_class_as_random"] <= direct["random_wrong"], "invalid paired overlap")
    baseline = sar["qwen_sar_baseline"]
    require(baseline["zeroshot"]["total"] + baseline["domain_prior"]["total"] ==
            baseline["completed_calls"] == baseline["planned_calls"], "baseline call count mismatch")
    for condition, prefix in (("zeroshot", ""), ("domain_prior", "saved_")):
        item = baseline[condition]
        require(sum(item[prefix + label] for label in ("correct", "wrong", "unresolved")) == item["total"],
                "baseline judgment denominator mismatch")
        require(0 <= item["capped"] <= item["total"], "invalid capped count")
    image_text = sar["qwen_sar_text"]
    require(image_text["planned_calls"] == image_text["completed_calls"] == image_text["images"] * 6,
            "SAR text call count mismatch")
    for key, value in image_text.items():
        if key.endswith(("_correct", "_marker", "_recovered", "_exact", "_capped")):
            require(type(value) is int and 0 <= value <= image_text["images"], f"invalid {key}")
    return len(records)


def main():
    synthetic = json.loads((DIRECTORY / "qwen-synthetic.json").read_text(encoding="utf-8"))
    sar = json.loads((DIRECTORY / "sar-summary.json").read_text(encoding="utf-8"))
    print(f"Validated {validate(synthetic, sar)} synthetic calls and SAR denominators.")


if __name__ == "__main__":
    main()
