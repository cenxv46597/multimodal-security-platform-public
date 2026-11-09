"""Verify SARChat resources and score the fixed grouped development panel."""

import argparse
import json
from pathlib import Path
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from run_sar_proxy_candidate import IMAGE_MEAN, IMAGE_SIZE, IMAGE_STD, prepare_prompt, predict
from run_vlm_clean_screen import ROOT, load_json, sha256, timed

PROTOCOL = ROOT / "experiments/protocols/sarchat_clean_screen_v0.json"
INVENTORY = ROOT / "resources/manifests/sarchat_snapshot_inventory_20261108.json"
SNAPSHOT = ROOT / "models/SARChat-InternVL2.5-1B/6c963963"


def main():
    import torch
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    from transformers import AutoModel, AutoTokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", required=True, choices=("probe", "screen"))
    parser.add_argument("--probe-report", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if (args.phase == "screen") != (args.probe_report is not None):
        parser.error("Screen requires a probe report")
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.exists():
        parser.error("Choose a new output path in runs")
    output.parent.mkdir(parents=True, exist_ok=True)
    protocol = load_json(PROTOCOL)
    report = {"status": "running", "phase": args.phase, "protocol_sha256": sha256(PROTOCOL),
              "runner_sha256": sha256(Path(__file__)), "inventory_sha256": sha256(INVENTORY),
              "predictions": []}
    started = time.perf_counter()
    try:
        inventory = load_json(INVENTORY)
        if inventory["model"] != protocol["model"] or inventory["revision"] != protocol["revision"]:
            raise ValueError("Inventory differs from protocol")
        for entry in inventory["files"]:
            path = (SNAPSHOT / entry["path"]).resolve()
            if not path.is_relative_to(SNAPSHOT) or path.stat().st_size != entry["size"] or sha256(path) != entry["sha256"]:
                raise ValueError(f"Snapshot verification failed for {entry['path']}")
        if sha256(SNAPSHOT / "model.safetensors") != protocol["weight_sha256"]:
            raise ValueError("Weight hash mismatch")
        _, _, _, rows, _, _, labels, screen = load_panel(protocol["grouped_plan_sha256"])
        if args.phase == "screen":
            probe = load_json(args.probe_report)
            if probe["status"] != "passed" or any(probe[k] != report[k] for k in
                                                  ("protocol_sha256", "runner_sha256", "inventory_sha256")):
                raise ValueError("Probe does not match this screen")
            report["source_probe_sha256"] = sha256(args.probe_report)
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA required")
        tokenizer = AutoTokenizer.from_pretrained(SNAPSHOT, local_files_only=True, trust_remote_code=False, use_fast=False)
        model, info = AutoModel.from_pretrained(
            SNAPSHOT, local_files_only=True, trust_remote_code=True, use_safetensors=True,
            torch_dtype=torch.float16, attn_implementation="eager", use_flash_attn=False,
            output_loading_info=True)
        if any(info.get(k) for k in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Checkpoint loading issue: {info}")
        model.eval().requires_grad_(False).to("cuda")
        if model.language_model.config._attn_implementation != "eager" or model.vision_model.config.use_flash_attn:
            raise ValueError("Expected eager attention")
        transform = Compose([Resize((IMAGE_SIZE, IMAGE_SIZE), interpolation=InterpolationMode.BICUBIC),
                             ToTensor(), Normalize(IMAGE_MEAN, IMAGE_STD)])
        inputs, separator, template = prepare_prompt(model, tokenizer, rows[0]["question"])
        inputs = {k: v.to("cuda") for k, v in inputs.items()}
        generation = {**screen["generation"], "eos_token_id": template["eos_token_id"]}
        report.update(model=protocol["model"], revision=protocol["revision"], template=template,
                      execution={"dtype": "float16", "attention": "eager", "image_size": IMAGE_SIZE,
                                 "generation": screen["generation"]},
                      gpu=torch.cuda.get_device_name())
        if args.phase == "probe":
            timed(report, "probe", lambda: predict(model, tokenizer, inputs, separator, transform,
                                                   ROOT / rows[0]["asset"]["file"], generation), "cuda")
            report.update(status="passed", probe_sample_id=rows[0]["sample_id"])
        else:
            for row in rows:
                item = {"sample_id": row["sample_id"], "status": "ok"}
                try:
                    timed(item, "generation", lambda: predict(model, tokenizer, inputs, separator, transform,
                                                              ROOT / row["asset"]["file"], generation), "cuda")
                    item.update(item.pop("generation"))
                except Exception as error:
                    item.update(status="error", answer=None, error_type=type(error).__name__, error=str(error))
                report["predictions"].append(item)
                output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
                print(f"Evaluated {len(report['predictions'])}/60", flush=True)
            report["score"] = score_samples(rows, report["predictions"], labels, screen)
            report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    if torch.cuda.is_available():
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], output, flush=True)
    return int(report["status"] == "failed")


if __name__ == "__main__":
    raise SystemExit(main())
