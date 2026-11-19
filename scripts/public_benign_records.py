"""Validate sanitized results or generate the benign blue-squares fixture."""

import argparse
from collections import Counter
import json
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_qwen(data):
    records = data["records"]
    require(len({row["id"] for row in records}) == len(records), "duplicate record ID")
    counts = Counter(row["status"] for row in records)
    require(not set(counts) - {"completed", "resource_blocked", "timeout", "preflight_blocked", "not_run"},
            "unexpected execution state")
    for row in records:
        if row["status"] != "completed":
            require(all(row.get(key) is None for key in ("answer", "semantic", "exact_pass")),
                    f'non-completed item has an answer judgment: {row["id"]}')
            continue
        answer = row["answer"]
        require(isinstance(answer, str) and bool(answer.strip()) and row["capped"] is False,
                f'ineligible answer: {row["id"]}')
        require(row["semantic"] in {"correct", "wrong", "unresolved"}, "unknown semantic label")
        expected = row["exact_expected"]
        actual = None if expected is None else answer.strip() == expected
        require(row["exact_pass"] is actual, f'exact score mismatch: {row["id"]}')
    return counts


def validate_sar(data):
    transfer = data["transfer"]
    conditions = transfer["conditions"]
    require(len({(row["model_role"], row["condition"]) for row in conditions}) == len(conditions),
            "duplicate transfer condition")
    total = 0
    for row in conditions:
        parts = [row[key] for key in ("correct", "wrong", "unresolved", "target_class_hits")]
        require(all(type(value) is int and value >= 0 for value in parts), "invalid transfer count")
        require(row["target_class_hits"] <= row["wrong"], "target hits exceed errors")
        require(sum(parts[:3]) == transfer["generation_completed_images"], "transfer row denominator mismatch")
        total += sum(parts[:3])
    require(total == transfer["new_evaluations_completed"], "total transfer denominator mismatch")
    query = data["direct_query"]
    planned, observed, missing, clean, modified, random, overlap = (query[key] for key in (
        "planned_images", "images_with_results", "missing_results", "clean_correct",
        "modified_wrong", "random_wrong", "paired_same_wrong_class",
    ))
    require(all(type(value) is int and value >= 0 for value in
                (planned, observed, missing, clean, modified, random, overlap)), "invalid direct-query count")
    require(planned == observed + missing, "direct-query denominator mismatch")
    require(max(clean, modified, random) <= observed and overlap <= min(modified, random),
            "invalid direct-query overlap")
    return total, observed, missing


def generate(output):
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(image)
    for box in ((60, 154, 199, 293), (248, 154, 387, 293)):
        draw.rectangle(box, fill=(0, 84, 185))
    with output.open("xb") as stream:
        image.save(stream, format="PNG")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    validate_parser = subcommands.add_parser("validate")
    validate_parser.add_argument("directory", type=Path)
    generate_parser = subcommands.add_parser("generate")
    generate_parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.command == "generate":
        generate(args.output)
        return
    qwen = json.loads((args.directory / "qwen-benign.json").read_text(encoding="utf-8"))
    sar = json.loads((args.directory / "sar-summary.json").read_text(encoding="utf-8"))
    print("Qwen states:", json.dumps(validate_qwen(qwen), sort_keys=True))
    total, observed, missing = validate_sar(sar)
    print(f"SAR evaluations: {total}; direct-query results: {observed}; missing: {missing}")


if __name__ == "__main__":
    main()
