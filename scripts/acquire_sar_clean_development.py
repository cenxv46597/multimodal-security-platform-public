"""Acquire only the frozen 60-image clean development plan from author-linked Kaggle.

Uses anonymous, version-pinned per-file Range requests. No model or upstream code
executes. CRC32 is the official archive check; SHA256 is a local acquisition hash.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import hashlib
import io
import json
from pathlib import Path, PurePosixPath
import re
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zlib

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "resources/manifests/sar_clean_development_plan_20261105.json"
DEFAULT_MAP = ROOT / ".work/next-clean-screen-20261105/images/selected_60_file_map.json"
COCO = ROOT / ".work/next-clean-screen-20261105/images/coco_train_selected60_validation.json"
RAW = ROOT / "data/raw/sardet_clean_dev/kaggle-v1"
RECORDS = ROOT / "data/processed/sar_clean_development_v0/records.jsonl"
MANIFEST = ROOT / "resources/manifests/sar_clean_development_acquisition_20261105.json"
LOCK = threading.Lock()
network_bytes = 0


def now():
    return datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds")


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def json_load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def decode(data, expected):
    import numpy as np
    from PIL import Image

    with Image.open(io.BytesIO(data)) as image:
        image.load()
        array = np.asarray(image)
        if image.mode not in ("L", "RGB") or array.dtype != np.uint8 or not np.isfinite(array).all():
            raise ValueError(f"Unspecified image representation {image.mode}/{array.dtype}")
        if list(image.size) != [expected["width"], expected["height"]]:
            raise ValueError(f"Decoded dimensions {image.size} differ from official COCO")
        return {"format": image.format, "mode": image.mode, "size": list(image.size),
                "dtype": str(array.dtype), "range": [int(array.min()), int(array.max())],
                "all_finite": True, "full_pixel_decode": "passed"}


def integrity(data, entry):
    if len(data) != entry["uncompressed_bytes"]:
        raise ValueError("File length differs from official ZIP central directory")
    crc = f"{zlib.crc32(data) & 0xffffffff:08x}"
    if crc != entry["crc32"]:
        raise ValueError(f"File CRC32 {crc} differs from official ZIP {entry['crc32']}")


def destination(entry):
    member = PurePosixPath(entry["name"])
    expected = PurePosixPath("SARDet_100K/JPEGImages") / entry["kaggle_image_directory"] / entry["image_name"]
    if member != expected or member.name != entry["image_name"]:
        raise ValueError("Unexpected member path")
    target = (RAW / entry["kaggle_image_directory"] / entry["image_name"]).resolve()
    if not target.is_relative_to(RAW.resolve()):
        raise ValueError("Download target leaves allocated raw directory")
    return target


def fetch(entry, attempts, budget):
    global network_bytes
    target = destination(entry)
    result = {"sample_id": entry["sample_id"], "image_name": entry["image_name"],
              "source_url": entry["download_url"], "dataset_version": 1,
              "storage_dataset_id": 7339941, "storage_version_id": 11694364,
              "sarvqa_author_split": entry["sarvqa_author_split"],
              "kaggle_image_directory": entry["kaggle_image_directory"],
              "official_bytes": entry["uncompressed_bytes"], "official_zip_crc32": entry["crc32"],
              "sha256_provenance": "Local SHA256 of acquired bytes; not an official upstream SHA256",
              "attempts": []}
    if target.is_file():
        data = target.read_bytes()
        integrity(data, entry)
        result.update(status="existing_integrity_verified", file=target.relative_to(ROOT).as_posix(),
                      bytes=len(data), sha256=sha256(data))
        return result, data
    size = entry["uncompressed_bytes"]
    for number in range(1, attempts + 1):
        attempt = {"number": number, "body_bytes": 0}
        try:
            request = urllib.request.Request(entry["download_url"], headers={
                "User-Agent": "sar-clean-development-fixed-plan/1.0", "Range": f"bytes=0-{size-1}"})
            with urllib.request.urlopen(request, timeout=40) as response:
                final = urllib.parse.urlsplit(response.url)
                attempt.update(status=response.status, final_host=final.netloc, final_path=final.path,
                               content_range=response.headers.get("Content-Range"), etag=response.headers.get("ETag"))
                if response.status != 206:
                    raise ValueError("Range ignored; body not read")
                match = re.fullmatch(r"bytes (\d+)-(\d+)/(\d+)", attempt["content_range"] or "")
                if not match or tuple(map(int, match.groups())) != (0, size - 1, size):
                    raise ValueError("Unexpected Content-Range; body not read")
                expected_suffix = "/datasets/7339941/11694364/" + entry["name"]
                if final.hostname != "storage.googleapis.com" or not final.path.endswith(expected_suffix):
                    raise ValueError("Unexpected pinned storage dataset/version/member; body not read")
                if response.headers.get("Content-Encoding", "identity") != "identity":
                    raise ValueError("Unexpected content encoding; body not read")
                parts = []
                count = 0
                while count <= size:
                    block = response.read(min(65536, size + 1 - count))
                    if not block:
                        break
                    count += len(block)
                    attempt["body_bytes"] += len(block)
                    with LOCK:
                        network_bytes += len(block)
                        if network_bytes > budget:
                            raise ValueError("Fixed acquisition response budget exceeded")
                    parts.append(block)
                data = b"".join(parts)
            integrity(data, entry)
            target.parent.mkdir(parents=True, exist_ok=True)
            part = target.with_name(target.name + ".part")
            part.write_bytes(data)
            part.replace(target)
            result["attempts"].append(attempt)
            result.update(status="downloaded_integrity_verified", file=target.relative_to(ROOT).as_posix(),
                          bytes=len(data), sha256=sha256(data))
            return result, data
        except Exception as error:
            attempt.update(error_type=type(error).__name__, error=str(error))
            result["attempts"].append(attempt)
            if isinstance(error, ValueError) or number == attempts:
                raise RuntimeError(json.dumps({**result, "status": "failed"})) from error
            time.sleep(min(number, 2))
    raise RuntimeError("Acquisition did not terminate")


def acquire(entry, expected, attempts, budget):
    try:
        result, data = fetch(entry, attempts, budget)
        result["image"] = decode(data, expected)
        result["coco"] = expected
        result["status"] = "passed"
        return result
    except Exception as error:
        return {"sample_id": entry["sample_id"], "image_name": entry["image_name"],
                "status": "failed", "error_type": type(error).__name__, "error": str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--file-map", type=Path, default=DEFAULT_MAP)
    parser.add_argument("--workers", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--attempts", type=int, choices=range(1, 4), default=3)
    args = parser.parse_args()
    plan_bytes = PLAN.read_bytes()
    plan = json.loads(plan_bytes)
    if args.file_map.is_file():
        file_map = json_load(args.file_map)
    elif MANIFEST.is_file():
        file_map = json_load(MANIFEST)["official_file_map"]
    else:
        raise FileNotFoundError("Need the bounded official central-directory file map")
    if file_map["source_plan_sha256"] != sha256(plan_bytes):
        raise ValueError("Frozen plan hash differs from official file map")
    if len(plan["selected"]) != 60 or len(file_map["selected"]) != 60:
        raise ValueError("Expected the unchanged 60-image plan")
    selected = {row["sample_id"]: row for row in plan["selected"]}
    entries = {row["sample_id"]: row for row in file_map["selected"]}
    if len(selected) != 60 or selected.keys() != entries.keys():
        raise ValueError("File map does not exactly cover the frozen sample IDs")
    for sample_id, row in selected.items():
        entry = entries[sample_id]
        if row["image_name"] != entry["image_name"] or row["reference_answer"] != entry["reference_answer"]:
            raise ValueError("File map changes the frozen sample or reference")
        destination(entry)
    if COCO.is_file():
        coco_bytes = COCO.read_bytes()
        coco = json.loads(coco_bytes)
    elif MANIFEST.is_file():
        prior = json_load(MANIFEST)
        coco = prior["official_coco_validation"]
        coco_bytes = json.dumps(coco, sort_keys=True).encode()
    else:
        raise FileNotFoundError("Need the bounded official COCO annotation validation")
    if coco["status"] != "passed" or coco["source_plan_sha256"] != sha256(plan_bytes):
        raise ValueError("COCO check is incomplete or differs from the frozen plan")
    expected = {row["sample_id"]: row for row in coco["selected_checks"]}
    if expected.keys() != selected.keys() or coco["problems"]:
        raise ValueError("COCO metadata differs from selected images")
    budget = sum(row["uncompressed_bytes"] for row in entries.values()) * args.attempts
    report = {"started_at": now(), "scope": "Fixed 60-image exploratory development acquisition only",
              "status": "running", "source_plan": PLAN.relative_to(ROOT).as_posix(),
              "source_plan_sha256": sha256(plan_bytes), "official_file_map": file_map,
              "official_coco_validation": coco, "coco_evidence_sha256": sha256(coco_bytes),
              "license": {"author_repository": "CC-BY-NC-4.0", "kaggle_metadata": "MIT",
                          "note": "Conflicting fields retained; MIT does not automatically override author or constituent-source terms. No redistribution performed."},
              "files": [], "network_response_body_budget": budget,
              "formal_split_created": False, "model_inference_run": False}
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    results = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {pool.submit(acquire, entries[sample_id], expected[sample_id], args.attempts, budget): sample_id for sample_id in selected}
        for job in as_completed(jobs):
            sample_id = jobs[job]
            results[sample_id] = job.result()
            print(json.dumps({"completed": len(results), "sample_id": sample_id,
                              "status": results[sample_id]["status"]}), flush=True)
    report["files"] = [results[row["sample_id"]] for row in plan["selected"]]
    passed = [row for row in report["files"] if row["status"] == "passed"]
    report.update(completed_at=now(), network_response_body_bytes=network_bytes,
                  passed_files=len(passed), selected_files=60,
                  total_verified_image_bytes=sum(row["bytes"] for row in passed),
                  status="passed" if len(passed) == 60 else "incomplete_or_needs_review")
    duplicate_hashes = defaultdict(list)
    for row in passed:
        duplicate_hashes[row["sha256"]].append(row["sample_id"])
    report["exact_file_duplicate_groups"] = [rows for rows in duplicate_hashes.values() if len(rows) > 1]
    report["scene_independence"] = "Unresolved: COCO images only provide filename/id/width/height, without original-scene lineage"
    report["derived_records_written"] = False
    if len(passed) == 60:
        records = []
        for row in plan["selected"]:
            acquired = results[row["sample_id"]]
            records.append({**row, "image_group_id": "image-bytes-sha256:" + acquired["sha256"],
                            "group_scope": "Exact acquired file bytes only; original scene independence unresolved",
                            "asset": {"file": acquired["file"], "sha256": acquired["sha256"],
                                      **acquired["image"]},
                            "source_kaggle": {"dataset": "greatbird/sardet-100k", "version": 1,
                                               "member": entries[row["sample_id"]]["name"],
                                               "directory": acquired["kaggle_image_directory"]},
                            "source_coco": acquired["coco"]})
        if Counter(row["reference_answer"] for row in records) != Counter({label: 10 for label in ("ship", "aircraft", "car", "tank", "bridge", "harbor")}):
            raise ValueError("Frozen class counts changed")
        RECORDS.parent.mkdir(parents=True, exist_ok=True)
        RECORDS.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records), encoding="utf-8")
        report["derived_records_written"] = True
        report["records"] = {"path": RECORDS.relative_to(ROOT).as_posix(), "count": 60,
                             "sha256": sha256(RECORDS.read_bytes())}
    MANIFEST.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("status", "passed_files", "selected_files", "total_verified_image_bytes", "network_response_body_bytes")}))
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
