"""Read-only verification of the pinned preparation resources (stdlib only).

Paths are resolved against this script's project root, independent of cwd.
Only an explicitly requested --output file is written. No upstream code runs.
"""

from __future__ import annotations

import argparse
import collections
import csv
import datetime
import hashlib
import io
import json
import math
from pathlib import Path
import struct
import sys
import zlib


ROOT = Path(__file__).resolve().parents[1]
RESOURCE_ROOTS = (
    "data/raw/sample/1bae4156afcb",
    "data/raw/hrrp_pc/a1ab400c91f3",
    "models/SmolVLM-256M-Instruct/7e3e67edbbed",
    "third_party/torchattacks/936e86d6387e",
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def project_file(relative_path: str) -> Path:
    path = (ROOT / relative_path).resolve()
    require(path.is_relative_to(ROOT), "resource path leaves project root")
    return path


def paeth(left: int, above: int, upper_left: int) -> int:
    prediction = left + above - upper_left
    a, b, c = (
        abs(prediction - left),
        abs(prediction - above),
        abs(prediction - upper_left),
    )
    return left if a <= b and a <= c else above if b <= c else upper_left


def inspect_png(data: bytes) -> dict:
    """Validate this selected 8-bit grayscale format and reconstruct pixels."""
    require(data[:8] == b"\x89PNG\r\n\x1a\n", "invalid PNG signature")
    position = 8
    chunks = []
    image_data = []
    ihdr = None
    while position < len(data):
        require(position + 12 <= len(data), "truncated PNG chunk")
        length = struct.unpack_from(">I", data, position)[0]
        end = position + length + 12
        require(end <= len(data), "PNG chunk exceeds file length")
        kind = data[position + 4 : position + 8]
        payload = data[position + 8 : position + 8 + length]
        crc = struct.unpack_from(">I", data, position + 8 + length)[0]
        require(zlib.crc32(kind + payload) & 0xFFFFFFFF == crc, "PNG CRC mismatch")
        chunks.append(kind)
        if kind == b"IHDR":
            require(ihdr is None and length == 13, "invalid or duplicate IHDR")
            ihdr = struct.unpack(">IIBBBBB", payload)
        elif kind == b"IDAT":
            image_data.append(payload)
        elif kind == b"IEND":
            require(length == 0, "nonempty IEND")
        position = end
        if kind == b"IEND":
            break
    require(position == len(data), "bytes after IEND")
    require(chunks and chunks[0] == b"IHDR" and chunks[-1] == b"IEND", "PNG chunk order")
    require(ihdr == (128, 128, 8, 0, 0, 0, 0), "unexpected selected PNG format")
    idat_positions = [i for i, kind in enumerate(chunks) if kind == b"IDAT"]
    require(bool(idat_positions), "missing IDAT")
    require(
        idat_positions == list(range(idat_positions[0], idat_positions[-1] + 1)),
        "nonconsecutive IDAT chunks",
    )
    width, height = ihdr[:2]
    stream = zlib.decompressobj()
    raw = stream.decompress(b"".join(image_data)) + stream.flush()
    require(stream.eof and not stream.unused_data and not stream.unconsumed_tail, "incomplete or trailing zlib stream")
    require(len(raw) == height * (width + 1), "unexpected PNG decompressed size")
    previous = bytearray(width)
    filters = collections.Counter()
    minimum, maximum = 255, 0
    for row_number in range(height):
        offset = row_number * (width + 1)
        filter_type = raw[offset]
        require(filter_type in range(5), "invalid PNG row filter")
        filters[str(filter_type)] += 1
        row = bytearray(raw[offset + 1 : offset + width + 1])
        if filter_type == 1:
            for x in range(1, width):
                row[x] = (row[x] + row[x - 1]) & 255
        elif filter_type == 2:
            for x in range(width):
                row[x] = (row[x] + previous[x]) & 255
        elif filter_type == 3:
            for x in range(width):
                left = row[x - 1] if x else 0
                row[x] = (row[x] + (left + previous[x]) // 2) & 255
        elif filter_type == 4:
            for x in range(width):
                left = row[x - 1] if x else 0
                upper_left = previous[x - 1] if x else 0
                row[x] = (row[x] + paeth(left, previous[x], upper_left)) & 255
        minimum, maximum = min(minimum, min(row)), max(maximum, max(row))
        previous = row
    return {"filters": dict(filters), "minimum": minimum, "maximum": maximum, "decoded_pixels": width * height}


def inspect_hrrp_csv(data: bytes) -> list[float]:
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig"))))
    require(len(rows) == 512 and all(len(row) == 1 for row in rows), "HRRP shape is not 512x1")
    values = [float(row[0]) for row in rows]
    require(all(map(math.isfinite, values)), "nonfinite HRRP value")
    return values


def inspect_hrrp_metadata(contents: dict[str, str]) -> dict:
    base = ROOT / RESOURCE_ROOTS[1]
    with (base / "metadata.csv").open(encoding="utf-8-sig", newline="") as handle:
        metadata = list(csv.DictReader(handle))
    index = {row["hrrp_filename"]: row for row in metadata}
    require(len(index) == len(metadata), "duplicate metadata filename")
    subset = json.loads((ROOT / "resources/manifests/hrrp_pc_format_subset.json").read_text(encoding="utf-8"))
    actual_names = set(contents)
    require({Path(name).name for name in subset["files"]} == actual_names, "HRRP subset file list mismatch")
    selected_metadata = {row["hrrp_filename"]: row for row in subset["metadata_rows"]}
    require(len(selected_metadata) == len(subset["metadata_rows"]), "duplicate subset metadata filename")
    require(set(selected_metadata) == actual_names, "HRRP subset metadata file list mismatch")
    groups = collections.defaultdict(lambda: {"splits": set(), "augmentations": set(), "files": 0})
    duplicate_contents = collections.defaultdict(lambda: {"splits": set(), "files": []})
    for filename, digest in sorted(contents.items()):
        require(filename in index, "CSV absent from metadata: " + filename)
        row = index[filename]
        require(row == selected_metadata[filename], "subset metadata differs from full metadata")
        group = groups[(row["label_str"], row["theta"])]
        group["splits"].add(row["split"])
        group["augmentations"].add(row["augmentation_type"])
        group["files"] += 1
        duplicate_contents[digest]["splits"].add(row["split"])
        duplicate_contents[digest]["files"].append(filename)
    group_cross_split = sum(len(group["splits"]) > 1 for group in groups.values())
    content_cross_split = sum(len(group["splits"]) > 1 for group in duplicate_contents.values())
    require(group_cross_split == 0, "HRRP label/theta group crosses split")
    require(content_cross_split == 0, "HRRP identical content crosses split")
    expected_splits = {"50": "train", "121": "validation", "161": "test"}
    for (label, theta), group in groups.items():
        require(group["files"] == 16 and group["augmentations"] == {str(i) for i in range(16)}, "incomplete selected augmentation group")
        require(group["splits"] == {expected_splits.get(theta)}, "unexpected selected angle/split")
    require(len(groups) == 9, "unexpected HRRP selected group count")
    return {
        "metadata_rows": len(metadata),
        "metadata_unique_filenames": len(index),
        "selected_groups": [
            {"label": label, "theta": theta, "split": next(iter(group["splits"])), "files": group["files"]}
            for (label, theta), group in sorted(groups.items())
        ],
        "groups_crossing_split": group_cross_split,
        "identical_contents_crossing_split": content_cross_split,
        "unique_selected_contents": len(duplicate_contents),
        "duplicate_selected_content_count": len(contents) - len(duplicate_contents),
        "duplicate_content_groups": [
            {"sha256": digest, "split": next(iter(group["splits"])), "files": group["files"]}
            for digest, group in sorted(duplicate_contents.items()) if len(group["files"]) > 1
        ],
        "scope": "Selected format subset only; not independent observations or a formal benchmark",
    }


def inspect_safetensors() -> dict:
    model_directory = ROOT / RESOURCE_ROOTS[2]
    path = model_directory / "model.safetensors"
    size = path.stat().st_size
    with path.open("rb") as handle:
        first = handle.read(8)
        require(len(first) == 8, "truncated safetensors length")
        header_size = struct.unpack("<Q", first)[0]
        require(0 < header_size <= min(size - 8, 100_000_000), "invalid safetensors header size")
        header = json.loads(handle.read(header_size))
    tensors = {name: value for name, value in header.items() if name != "__metadata__"}
    element_sizes = {"BOOL": 1, "U8": 1, "I8": 1, "I16": 2, "U16": 2, "F16": 2, "BF16": 2, "I32": 4, "U32": 4, "F32": 4, "I64": 8, "U64": 8, "F64": 8}
    offsets = []
    total_elements = 0
    for name, tensor in tensors.items():
        begin, end = tensor["data_offsets"]
        require(type(begin) is int and type(end) is int and 0 <= begin <= end, "invalid tensor offset: " + name)
        require(tensor["dtype"] in element_sizes, "unsupported tensor dtype: " + name)
        require(all(type(x) is int and x >= 0 for x in tensor["shape"]), "invalid tensor shape: " + name)
        elements = math.prod(tensor["shape"])
        require(end - begin == elements * element_sizes[tensor["dtype"]], "tensor shape/size mismatch: " + name)
        offsets.append((begin, end, name))
        total_elements += elements
    previous = 0
    for begin, end, name in sorted(offsets):
        require(begin == previous, "noncontiguous or overlapping tensor: " + name)
        previous = end
    require(previous == size - 8 - header_size, "safetensors payload length mismatch")
    config = json.loads((model_directory / "config.json").read_text(encoding="utf-8"))
    expected_shapes = {
        "lm_head.weight": [config["text_config"]["vocab_size"], config["text_config"]["hidden_size"]],
        "model.text_model.embed_tokens.weight": [config["text_config"]["vocab_size"], config["text_config"]["hidden_size"]],
        "model.vision_model.embeddings.patch_embedding.weight": [config["vision_config"]["hidden_size"], config["vision_config"]["num_channels"], config["vision_config"]["patch_size"], config["vision_config"]["patch_size"]],
    }
    for name, shape in expected_shapes.items():
        require(tensors[name]["shape"] == shape, "config/weight shape mismatch: " + name)
    return {
        "tensor_entries": len(tensors),
        "dtypes": dict(collections.Counter(value["dtype"] for value in tensors.values())),
        "total_elements": total_elements,
        "data_payload_bytes": previous,
        "contiguous_offsets_and_shapes_match_size": True,
        "config_checked_shapes": expected_shapes,
        "scope": "Storage structure only; execution dtype, model correctness, and domain capability untested",
    }


def verify() -> dict:
    entries = [json.loads(line) for line in (ROOT / "resources/manifests/resource_inventory.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    expected = json.loads((ROOT / "resources/manifests/verification_summary.json").read_text(encoding="utf-8"))
    errors = []
    stats = collections.defaultdict(lambda: {"files": 0, "bytes": 0})
    seen = set()
    png_count = 0
    png_pixels = 0
    png_filters = collections.Counter()
    png_classes = collections.Counter()
    png_min, png_max = 255, 0
    hrrp_count = 0
    hrrp_values = 0
    hrrp_within_unit = 0
    hrrp_min, hrrp_max = math.inf, -math.inf
    hrrp_contents = {}
    for entry in entries:
        relative = entry["path"]
        try:
            require(relative not in seen, "duplicate manifest path")
            seen.add(relative)
            path = project_file(relative)
            data = path.read_bytes()
            resource = entry["resource"]
            stats[resource]["files"] += 1
            stats[resource]["bytes"] += len(data)
            require(len(data) == entry["bytes"], "size mismatch")
            digest = hashlib.sha256(data).hexdigest()
            require(digest == entry["sha256"], "SHA256 mismatch")
            if entry.get("upstream_sha256"):
                require(digest == entry["upstream_sha256"], "upstream SHA256 mismatch")
            if entry.get("upstream_git_blob_sha1"):
                blob_digest = hashlib.sha1(b"blob " + str(len(data)).encode("ascii") + b"\0" + data).hexdigest()
                require(blob_digest == entry["upstream_git_blob_sha1"], "upstream Git blob SHA1 mismatch")
            if path.suffix.lower() == ".png":
                png = inspect_png(data)
                png_count += 1
                png_pixels += png["decoded_pixels"]
                png_filters.update(png["filters"])
                png_classes[path.parent.name] += 1
                png_min, png_max = min(png_min, png["minimum"]), max(png_max, png["maximum"])
            if resource == "HRRP_PC_format_subset":
                values = inspect_hrrp_csv(data)
                hrrp_contents[path.name] = digest
                hrrp_count += 1
                hrrp_values += len(values)
                hrrp_min, hrrp_max = min(hrrp_min, min(values)), max(hrrp_max, max(values))
                hrrp_within_unit += min(values) >= 0 and max(values) <= 1
        except (OSError, ValueError, KeyError, TypeError, struct.error, zlib.error) as exception:
            errors.append({"path": relative, "error": str(exception)})
    actual = {
        path.relative_to(ROOT).as_posix()
        for resource_root in RESOURCE_ROOTS
        for path in (ROOT / resource_root).rglob("*") if path.is_file()
    }
    unexpected = sorted(actual - seen)
    missing = sorted(seen - actual)
    partials = sorted(
        path.relative_to(ROOT).as_posix()
        for resource_root in RESOURCE_ROOTS
        for path in (ROOT / resource_root).rglob("*.part")
        if path.is_file()
    )
    if unexpected or missing or partials:
        errors.append({"resource_paths": "missing, unexpected, or partial files found"})
    total_bytes = sum(value["bytes"] for value in stats.values())
    if len(entries) != expected["total_files"] or total_bytes != expected["total_resource_bytes"]:
        errors.append({"totals": "current totals differ from original verification summary"})
    for resource, old in expected["resources"].items():
        if stats.get(resource) != {"files": old["files"], "bytes": old["bytes"]}:
            errors.append({"resource": resource, "error": "count or bytes differ from original summary"})
    extra_checks = {}
    for name, check in (("hrrp_metadata", lambda: inspect_hrrp_metadata(hrrp_contents)), ("safetensors", inspect_safetensors)):
        try:
            extra_checks[name] = check()
        except (OSError, ValueError, KeyError, TypeError, struct.error) as exception:
            errors.append({"check": name, "error": str(exception)})
    return {
        "checked_at": datetime.datetime.now(datetime.timezone(datetime.timedelta(hours=8))).isoformat(timespec="seconds"),
        "timezone": "Asia/Shanghai (UTC+08:00)",
        "scope": "Read-only stdlib recheck of the four selected pinned resource directories, including partial-file checks limited to those directories; no model execution or upstream scripts",
        "resource_directories": list(RESOURCE_ROOTS),
        "status": "passed" if not errors else "failed",
        "manifest_entries": len(entries),
        "unique_paths": len(seen),
        "resources": dict(stats),
        "total_resource_bytes": total_bytes,
        "errors": errors,
        "unexpected_in_resource_directories": unexpected,
        "missing_from_resource_directories": missing,
        "unexpected_partial_files": partials,
        "png": {
            "files": png_count,
            "format": "128x128/8bit/grayscale/noninterlaced",
            "checks": "chunk bounds, CRCs, complete zlib stream, decompressed size, all rows unfiltered",
            "classes": dict(sorted(png_classes.items())),
            "decoded_pixels": png_pixels,
            "row_filters": dict(png_filters),
            "minimum": png_min if png_count else None,
            "maximum": png_max if png_count else None,
        },
        "hrrp_csv": {
            "files": hrrp_count,
            "shape_per_file": [512, 1],
            "finite_values": hrrp_values,
            "minimum": hrrp_min if hrrp_count else None,
            "maximum": hrrp_max if hrrp_count else None,
            "files_within_unit_interval": hrrp_within_unit,
            "files_outside_unit_interval": hrrp_count - hrrp_within_unit,
        },
        **extra_checks,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="Optional JSON output path (relative to project root); default writes stdout only")
    args = parser.parse_args()
    # Validate the one writable destination before doing any work. Raw resource
    # roots and original manifests can never be selected as an output file.
    output = None
    if args.output is not None:
        output = (ROOT / args.output).resolve()
        require(output.is_relative_to(ROOT), "output must remain inside project root")
        require(all(not output.is_relative_to(ROOT / resource_root) for resource_root in RESOURCE_ROOTS), "output cannot overwrite original resources")
        require(output != ROOT / "resources/manifests/resource_inventory.jsonl", "output cannot overwrite source inventory")
        require(output != ROOT / "resources/manifests/verification_summary.json", "output cannot overwrite original verification summary")
    report = verify()
    rendered = json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered, encoding="utf-8")
    sys.stdout.write(rendered)
    return 0 if report["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
