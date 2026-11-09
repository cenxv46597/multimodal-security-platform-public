"""Acquire and verify the frozen SAR adaptation proposal."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import http.client
import io
import json
from pathlib import Path
import struct
import threading
from urllib.error import HTTPError, URLError
from urllib.parse import unquote, urlsplit
from urllib.request import HTTPRedirectHandler, ProxyHandler, Request, build_opener
import zipfile

from PIL import Image

from acquire_sar_clean_development import integrity, now


ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "resources/manifests/sar_adaptation_data_plan_20261107.json"
PLAN_SHA256 = "7ead393652361a5e9b789c62794f956a128818e814c2a5674adeda2e6759a68a"
MANIFEST = ROOT / "resources/manifests/sar_adaptation_data_acquisition_20261107.json"
RAW = ROOT / "data/raw/sardet_adaptation_v0/kaggle-v1/train"
RECORDS = ROOT / "data/processed/sar_adaptation_v0/records.jsonl"
LIMIT = 39_801_682
ATTEMPTS = 2


class NoRedirect(HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Keep redirect bodies out of the data budget and validate the destination.
        return None


class ResponseBudget:
    def __init__(self):
        self.reserved = 0
        self.received = 0
        self.lock = threading.Lock()

    def reserve(self, length):
        with self.lock:
            if self.reserved + length > LIMIT:
                raise ValueError("Fixed response body budget exhausted")
            self.reserved += length

    def count(self, length):
        with self.lock:
            self.received += length


def sha256(data):
    return hashlib.sha256(data).hexdigest()


def read_source(entry):
    data = (ROOT / entry["path"]).read_bytes()
    if sha256(data) != entry["sha256"]:
        raise ValueError(f"Source checksum changed: {entry['path']}")
    return data


def decode_image(data, expected_size=None):
    with Image.open(io.BytesIO(data)) as image:
        image.load()
        if image.mode not in ("L", "RGB"):
            raise ValueError(f"Unsupported image mode: {image.mode}")
        if expected_size is not None and image.size != expected_size:
            raise ValueError(f"Decoded dimensions {image.size} differ from COCO {expected_size}")
        with image.convert("RGB") as rgb:
            pixels = rgb.tobytes()
        pixel_hash = hashlib.sha256(struct.pack(">QQ", *image.size))
        pixel_hash.update(pixels)
        return {
            "sha256": sha256(data), "pixel_sha256": pixel_hash.hexdigest(),
            "format": image.format, "mode": image.mode, "size": list(image.size), "dtype": "uint8",
        }


def baseline_hashes(sources):
    by_file = {}
    by_pixel = {}

    def add(data, source_id):
        decoded = decode_image(data)
        by_file.setdefault(decoded["sha256"], source_id)
        by_pixel.setdefault(decoded["pixel_sha256"], source_id)

    historical = read_source(sources["historical_development_records"])
    records = [json.loads(line) for line in historical.splitlines() if line.strip()]
    if len(records) != 60:
        raise ValueError("Historical development records do not contain 60 entries")
    for row in records:
        add(read_source({"path": row["asset"]["file"], "sha256": row["asset"]["sha256"]}),
            row["sample_id"])
    examples = read_source(sources["examples_zip"])
    with zipfile.ZipFile(io.BytesIO(examples)) as archive:
        members = [member for member in archive.infolist()
                   if member.filename.startswith("Examples/images/") and not member.is_dir()]
        if len(members) != 20:
            raise ValueError("Fixed Examples.zip does not contain 20 images")
        for member in members:
            add(archive.read(member), member.filename)
    return by_file, by_pixel


def fetch_body(row, opener, budget):
    member = row["name"]
    length = row["uncompressed_bytes"]
    initial = urlsplit(row["download_url"])
    if (initial.scheme, initial.netloc, unquote(initial.path), initial.query) != (
        "https", "www.kaggle.com", "/api/v1/datasets/download/greatbird/sardet-100k/" + member,
        "datasetVersionNumber=1",
    ):
        raise ValueError("Unexpected source URL")
    budget.reserve(length)
    headers = {"Range": f"bytes=0-{length - 1}", "Accept-Encoding": "identity",
               "User-Agent": "sar-adaptation-v0-fixed-acquisition/1.0"}
    try:
        response = opener.open(Request(row["download_url"], headers=headers), timeout=25)
    except HTTPError as error:
        with error:
            if error.code not in (301, 302, 303, 307, 308):
                raise
            location = error.headers.get("Location", "")
    else:
        response.close()
        raise ValueError("Kaggle did not redirect to the pinned storage member")

    storage = urlsplit(location)
    suffix = "/datasets/7339941/11694364/" + member
    if (storage.scheme != "https" or storage.hostname != "storage.googleapis.com"
            or storage.port not in (None, 443) or storage.username is not None
            or not unquote(storage.path).endswith(suffix)):
        raise ValueError("Kaggle redirected outside the pinned storage member")
    with opener.open(Request(location, headers=headers), timeout=25) as response:
        if (response.status != 206
                or response.headers.get("Content-Range") != f"bytes 0-{length - 1}/{length}"
                or response.headers.get("Content-Encoding", "identity") != "identity"
                or response.headers.get("Content-Length") != str(length)):
            raise ValueError("Invalid pinned Range response")
        data = bytearray()
        while len(data) < length:
            try:
                block = response.read(min(65_536, length - len(data)))
            except http.client.IncompleteRead as error:
                budget.count(len(error.partial))
                raise
            budget.count(len(block))
            if not block:
                raise EOFError("Truncated Range response")
            data.extend(block)
        proof = {"status": 206, "storage_path": unquote(storage.path),
                 "content_range": response.headers["Content-Range"], "etag": response.headers.get("ETag")}
    return bytes(data), proof


def fetch(row, proxy, budget):
    handlers = [NoRedirect()]
    if proxy:
        handlers.append(ProxyHandler({"https": proxy}))
    opener = build_opener(*handlers)
    failures = []
    for attempt in range(1, ATTEMPTS + 1):
        try:
            data, proof = fetch_body(row, opener, budget)
        except HTTPError as error:
            error.close()
            failures.append(f"HTTP {error.code}")
            if error.code not in (408, 429, 500, 502, 503, 504):
                break
        except (URLError, TimeoutError, OSError, EOFError, http.client.IncompleteRead) as error:
            reason = error.reason if isinstance(error, URLError) else error
            failures.append(type(reason).__name__)
        else:
            integrity(data, row)
            decoded = decode_image(data, (row["width"], row["height"]))
            return data, decoded, {"attempt": attempt, "prior_failures": failures, **proof}
    raise RuntimeError(f"Could not fetch {row['image_name']}: {', '.join(failures)}")


def save_record(row, data, decoded, proof, by_file, by_pixel):
    for key, seen in (("sha256", by_file), ("pixel_sha256", by_pixel)):
        digest = decoded[key]
        if digest in seen:
            raise ValueError(f"{row['sample_id']} duplicates {seen[digest]} by {key}")
    for key, seen in (("sha256", by_file), ("pixel_sha256", by_pixel)):
        seen[decoded[key]] = row["sample_id"]
    path = RAW / row["image_name"]
    with path.open("xb") as stream:
        stream.write(data)
    return {
        **row, "asset": {"file": path.relative_to(ROOT).as_posix(), "bytes": len(data), **decoded},
        "acquisition": proof,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proxy", help="HTTPS proxy for this acquisition process")
    parser.add_argument("--workers", type=int, choices=(1, 2, 3, 4), default=4)
    args = parser.parse_args()

    plan = json.loads(read_source({"path": PLAN.relative_to(ROOT).as_posix(), "sha256": PLAN_SHA256}))
    rows = plan["selected"]
    if (len(rows) != 180 or sum(row["uncompressed_bytes"] for row in rows) * ATTEMPTS != LIMIT
            or plan["download_budget"]["response_body_byte_limit"] != LIMIT):
        raise ValueError("Unexpected frozen sample count or budget")
    if MANIFEST.exists() or RECORDS.exists() or (RAW.exists() and any(RAW.iterdir())):
        raise FileExistsError("Preserve existing acquisition evidence and image files")
    by_file, by_pixel = baseline_hashes(plan["sources"])
    report = {
        "status": "running", "started_at": now(), "plan": PLAN.relative_to(ROOT).as_posix(),
        "source_plan_sha256": PLAN_SHA256, "requested_images": len(rows),
        "response_body_byte_limit": LIMIT, "max_attempts_per_image": ATTEMPTS,
        "historical_development_images_checked": 60, "example_images_checked": 20,
        "pixel_hash": "SHA256 of 8-byte big-endian width, 8-byte big-endian height, and decoded RGB bytes",
        "scope": plan["validation_scope"], "records": None,
        "training_run": False, "formal_test_created": False, "files": [],
    }
    with MANIFEST.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(report, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    budget = ResponseBudget()
    records = []
    try:
        RAW.mkdir(parents=True, exist_ok=True)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for start in range(0, len(rows), args.workers):
                batch = rows[start:start + args.workers]
                futures = [pool.submit(fetch, row, args.proxy, budget) for row in batch]
                for row, future in zip(batch, futures):
                    data, decoded, proof = future.result()
                    record = save_record(row, data, decoded, proof, by_file, by_pixel)
                    records.append(record)
                    report["files"].append({"sample_id": row["sample_id"], "asset": record["asset"],
                                            "acquisition": proof})
                    if len(records) % 10 == 0:
                        print(f"verified {len(records)}/{len(rows)} images, {budget.received} response bytes",
                              flush=True)
        encoded = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in records).encode("utf-8")
        RECORDS.parent.mkdir(parents=True, exist_ok=True)
        with RECORDS.open("xb") as stream:
            stream.write(encoded)
        report.update(
            status="passed", exact_file_and_decoded_pixel_isolation="passed",
            records={"path": RECORDS.relative_to(ROOT).as_posix(), "count": len(records),
                     "sha256": sha256(encoded)},
        )
    except BaseException as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        raise
    finally:
        report.update(completed_at=now(), verified_images=len(records),
                      response_body_bytes=budget.received, response_body_bytes_reserved=budget.reserved)
        MANIFEST.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "records": report["records"],
                      "response_body_bytes": budget.received}), flush=True)


if __name__ == "__main__":
    main()
