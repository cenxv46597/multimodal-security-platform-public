"""Compare target and background occlusion on a fixed SAR development panel."""

from collections import Counter
import json
from pathlib import Path
from statistics import mean
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor

from evaluate_sar_grouped_development import load_panel
from preflight_sar_adaptation import answer_logits, answer_targets, verify_checkpoint
from run_sar_projector_pilot import PILOT, checked_local_file, load_frozen_projector_model, load_projector_weights
from run_vlm_clean_screen import MODEL, ROOT, get_inputs, load_json, sha256


def main():
    protocol_path = ROOT / "experiments/protocols/sar_occlusion_score_v0.json"
    protocol = load_json(protocol_path)
    feasibility_path = ROOT / "resources/manifests/sar_occlusion_feasibility_20261108.json"
    if sha256(feasibility_path) != protocol["feasibility_sha256"]:
        raise ValueError("Feasibility record changed")
    feasibility = load_json(feasibility_path)
    _, _, _, validation, _, _, labels, _ = load_panel(protocol["grouped_plan_sha256"])
    if [r["sample_id"] for r in validation] != [r["sample_id"] for r in feasibility["samples"]]:
        raise ValueError("Panel order changed")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    torch.set_num_threads(4)
    verify_checkpoint()
    pilot = load_json(PILOT)
    model = load_frozen_projector_model()
    load_projector_weights(model.multi_modal_projector, checked_local_file(pilot["projector_checkpoint"], ".safetensors"))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    first = get_inputs(processor, validation[0], "cpu", torch.float32)
    targets = {k: v.to("cuda") for k, v in answer_targets(processor, first, validation[0]["question"], labels).items()}
    report = {"status": "running", "protocol": str(protocol_path.relative_to(ROOT)),
              "protocol_sha256": sha256(protocol_path), "projector": pilot["projector_checkpoint"], "samples": []}
    output = ROOT / "runs/sar-occlusion-20261108/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Use a new output path")
    started = time.perf_counter()
    with torch.inference_mode():
        for record, geometry in zip(validation, feasibility["samples"], strict=True):
            if geometry["status"] != "eligible":
                report["samples"].append({"sample_id": record["sample_id"], "label": record["reference_answer"],
                                          "status": geometry["status"]})
                continue
            original = get_inputs(processor, record, "cuda", torch.float16)
            with Image.open(ROOT / record["asset"]["file"]) as image:
                rgb = np.array(image.convert("RGB"))
            fill = np.median(rgb, axis=(0, 1)).astype(np.uint8)
            messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": record["question"]}]}]
            prompt = processor.apply_chat_template(messages, add_generation_prompt=True)
            scores = {}
            for condition in ("original", "target", "background"):
                if condition == "original":
                    inputs = original
                else:
                    altered = rgb.copy()
                    x0, y0, x1, y1 = geometry[condition + "_box"]
                    altered[y0:y1, x0:x1] = fill
                    batch = processor(text=prompt, images=[Image.fromarray(altered)], return_tensors="pt")
                    inputs = {key: value.to("cuda", dtype=torch.float16 if key == "pixel_values" else value.dtype)
                              for key, value in batch.items()}
                    if not torch.equal(inputs["input_ids"], original["input_ids"]):
                        raise ValueError("Prompt changed")
                target = targets[record["reference_answer"]]
                logits = answer_logits(model, inputs, target)
                value = logits.log_softmax(dim=-1).gather(-1, target.unsqueeze(-1)).sum()
                if not bool(torch.isfinite(value)):
                    raise ValueError("Nonfinite score")
                scores[condition] = float(value)
            target_drop = scores["original"] - scores["target"]
            background_drop = scores["original"] - scores["background"]
            report["samples"].append({"sample_id": record["sample_id"], "label": record["reference_answer"],
                                      "status": "ok", "fill": fill.tolist(), "scores": scores,
                                      "target_drop": target_drop, "background_drop": background_drop,
                                      "paired_difference": target_drop - background_drop})
            print(f"Scored {len(report['samples'])}/60", flush=True)
    report["per_class"] = {}
    for label in labels:
        group = [row for row in report["samples"] if row["label"] == label]
        valid = [row for row in group if row["status"] == "ok"]
        report["per_class"][label] = {"planned": len(group), "evaluated": len(valid),
                                      "status_counts": dict(Counter(row["status"] for row in group)),
                                      "mean_paired_difference": mean(row["paired_difference"] for row in valid) if valid else None,
                                      "target_drop_larger": sum(row["paired_difference"] > 0 for row in valid)}
    report.update(status="completed", seconds=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["per_class"]), flush=True)


if __name__ == "__main__":
    main()
