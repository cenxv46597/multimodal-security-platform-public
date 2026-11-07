"""Freeze a balanced author-training source plan before acquisition/model output."""
from collections import Counter, defaultdict
import csv
import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    task_path = ROOT / "experiments/protocols/sar_single_category_v0.json"
    screen_path = ROOT / "experiments/protocols/sar_clean_development_screen_v0.json"
    task = json.loads(task_path.read_text(encoding="utf-8"))
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    source = ROOT / "data/raw/sarlang/0271408f71bc/Text.zip"
    inventory = json.loads((ROOT / "resources/manifests/sarlang_pilot_inventory_20261105.json").read_text(encoding="utf-8"))
    expected = next(e["sha256"] for e in inventory["files"] if e["path"].endswith("/Text.zip"))
    if hashlib.sha256(source.read_bytes()).hexdigest() != expected:
        raise ValueError("Text.zip does not match pinned inventory")
    format_report = json.loads((ROOT / "resources/manifests/sarlang_pilot_format_20261105.json").read_text(encoding="utf-8"))
    excluded = set(format_report["examples"]["image_headers"])
    excluded_stems = {PurePosixPath(name).stem for name in excluded}
    pools = defaultdict(list)
    seed = screen["sampling"]["seed"]
    member = "Text/VQA/train/SARVQA1_train.csv"
    with zipfile.ZipFile(source) as archive, archive.open(member) as stream:
        for index, raw in enumerate(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8-sig", newline="")), 1):
            row = {k.strip(): v.strip() for k, v in raw.items() if k is not None}
            if (row["Application"] != "object classification" or
                    row["Question"] != task["source_selection"]["Question_after_outer_whitespace_trim"] or
                    row["Answer"] not in task["labels"]):
                continue
            name = PurePosixPath(row["Image Name"].replace("\\", "/")).name
            if name in excluded or PurePosixPath(name).stem in excluded_stems:
                continue
            key = "\n".join((seed, task["revision"], row["Answer"], name))
            digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
            pools[row["Answer"]].append({
                "sample_id": "sar-clean-dev-" + digest[:20], "selection_sha256": digest,
                "image_name": name, "reference_answer": row["Answer"],
                "author_split": "train", "source_member": member,
                "source_record_index_1_based": index, "original_question": row["Question"],
                "question": task["model_question"], "formal_split": None,
                "usage": "exploratory_clean_development_screen_only",
            })
    selected = []
    for label in task["labels"]:
        pool = sorted(pools[label], key=lambda r: (r["selection_sha256"], r["image_name"]))
        if len({r["image_name"] for r in pool}) != len(pool):
            raise ValueError("Duplicate selected source image names")
        if len(pool) < screen["sampling"]["per_class"]:
            raise ValueError("Insufficient class pool")
        selected.extend(pool[:screen["sampling"]["per_class"]])
    if len(selected) != 60 or len({r["image_name"] for r in selected}) != 60:
        raise ValueError("Unexpected balanced plan size")
    plan = {
        "created_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "screen_id": screen["screen_id"], "sampling_seed": seed,
        "dataset_revision": task["revision"], "annotation_archive_sha256": expected,
        "task_protocol_sha256": hashlib.sha256(task_path.read_bytes()).hexdigest(),
        "screen_protocol_sha256": hashlib.sha256(screen_path.read_bytes()).hexdigest(),
        "eligible_pool_counts_after_demo_exclusion": {label: len(pools[label]) for label in task["labels"]},
        "selected_counts": dict(Counter(r["reference_answer"] for r in selected)),
        "selected": selected, "model_outputs_used_for_selection": False,
        "acquisition_status": "not_yet_acquired", "formal_test_created": False,
    }
    target = ROOT / "resources/manifests/sar_clean_development_plan_20261105.json"
    if target.exists():
        old = json.loads(target.read_text(encoding="utf-8"))
        if (old["selected"] != selected or old["screen_protocol_sha256"] != plan["screen_protocol_sha256"] or
                old["task_protocol_sha256"] != plan["task_protocol_sha256"]):
            raise ValueError("Existing frozen plan differs; use a new explicit version")
        print("Existing frozen plan unchanged")
    else:
        target.write_text(json.dumps(plan, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"selected": len(selected), "classes": plan["selected_counts"]}))
    print(target)


if __name__ == "__main__":
    main()
