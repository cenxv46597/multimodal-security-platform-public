"""Evaluate annotated-object crops using the frozen adapted LLaVA checkpoint."""

from collections import Counter
import json
import math
from pathlib import Path
import time

import torch
from PIL import Image
from transformers import AutoProcessor

from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from preflight_sar_adaptation import verify_checkpoint
from prepare_sar_grouped_development import source_rows
from run_sar_projector_pilot import PILOT, checked_local_file, load_frozen_projector_model, load_projector_weights
from run_vlm_clean_screen import MODEL, ROOT, generate_answer, load_json, sha256


def main():
    protocol_path = ROOT / "experiments/protocols/sar_target_crop_v0.json"
    protocol = load_json(protocol_path)
    _, _, _, validation, _, _, labels, screen = load_panel(protocol["grouped_plan_sha256"])
    source, _, _ = source_rows()
    annotations = {row["record"]["sample_id"]: row for row in source}
    reference_path = ROOT / "runs/sar-grouped-development-20261107/comparison.json"
    reference = load_json(reference_path)["evaluations"]["standard"]["score"]["sample_results"]
    if [row["sample_id"] for row in reference] != [row["sample_id"] for row in validation]:
        raise ValueError("Whole-image reference panel changed")
    verify_checkpoint()
    pilot = load_json(PILOT)
    model = load_frozen_projector_model()
    load_projector_weights(model.multi_modal_projector, checked_local_file(pilot["projector_checkpoint"], ".safetensors"))
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    torch.set_num_threads(4)
    output = ROOT / "runs/sar-target-crop-20261108/results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Output already exists")
    report = {"status": "running", "protocol_sha256": sha256(protocol_path),
              "reference_sha256": sha256(reference_path), "predictions": []}
    started = time.perf_counter()
    for row in validation:
        prediction = {"sample_id": row["sample_id"], "status": "ok"}
        try:
            width, height = row["asset"]["size"]
            boxes = []
            for x, y, w, h in annotations[row["sample_id"]]["boxes"]:
                clipped = (max(0, math.floor(x)), max(0, math.floor(y)),
                           min(width, math.ceil(x + w)), min(height, math.ceil(y + h)))
                if clipped[2] > clipped[0] and clipped[3] > clipped[1]:
                    boxes.append(clipped)
            if not boxes:
                raise ValueError("No valid annotated target")
            x0, y0, x1, y1 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            pad_x, pad_y = (x1 - x0) * .25, (y1 - y0) * .25
            crop_box = (max(0, math.floor(x0 - pad_x)), max(0, math.floor(y0 - pad_y)),
                        min(width, math.ceil(x1 + pad_x)), min(height, math.ceil(y1 + pad_y)))
            with Image.open(ROOT / row["asset"]["file"]) as image:
                image.load()
                crop = image.convert("RGB").crop(crop_box)
            prompt = processor.apply_chat_template([{"role": "user", "content": [
                {"type": "image"}, {"type": "text", "text": row["question"]}]}], add_generation_prompt=True)
            inputs = processor(text=prompt, images=[crop], return_tensors="pt")
            inputs = {key: value.to("cuda", dtype=torch.float16 if key == "pixel_values" else value.dtype)
                      for key, value in inputs.items()}
            if int((inputs["input_ids"] == 151646).sum()) != 729:
                raise ValueError("Unexpected image token count")
            answer, tokens = generate_answer(model, processor, inputs, 12)
            prediction.update(answer=answer, generated_tokens=tokens, target_box=[x0, y0, x1, y1],
                              crop_box=list(crop_box), crop_size=list(crop.size))
        except Exception as error:
            prediction.update(status="error", answer=None, error_type=type(error).__name__, error=str(error))
        report["predictions"].append(prediction)
        output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        print(f"Evaluated {len(report['predictions'])}/60", flush=True)
    score = score_samples(validation, report["predictions"], labels, screen)
    counts = Counter()
    for original, cropped in zip(reference, score["sample_results"], strict=True):
        counts[(original["correct"], cropped["correct"])] += 1
    report.update(status="completed", score=score,
                  paired={"both_correct": counts[(True, True)], "crop_only_correct": counts[(False, True)],
                          "whole_only_correct": counts[(True, False)], "both_wrong": counts[(False, False)]},
                  seconds=time.perf_counter() - started,
                  peak_allocated_mib=torch.cuda.max_memory_allocated() / 2**20)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"correct": score["correct"], "per_class": score["per_class"], "paired": report["paired"]}))


if __name__ == "__main__":
    main()
