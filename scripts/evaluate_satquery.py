"""Evaluate the fixed SatQuery adapter on the grouped SAR development panel."""
import json
from pathlib import Path
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from evaluate_sar_grouped_development import load_panel
from run_vlm_clean_screen import ROOT, load_json, sha256


def main():
    import torch
    from PIL import Image
    from peft import PeftModel, get_peft_model_state_dict
    from safetensors.torch import load_file
    from torchvision.transforms import Compose, InterpolationMode, Normalize, Resize, ToTensor
    from transformers import AutoModel, AutoTokenizer

    protocol_path = ROOT / "experiments/protocols/sar_satquery_screen_v0.json"
    protocol = load_json(protocol_path)
    base = ROOT / ".work/satquery-20261108/base"
    adapter = ROOT / ".work/satquery-20261108/adapter"
    output = ROOT / "runs/sar-satquery-20261108/screen.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise ValueError("Output already exists")
    report = {"status": "running", "protocol_sha256": sha256(protocol_path), "predictions": []}
    start = time.perf_counter()
    try:
        if sha256(base / "model.safetensors") != protocol["base_sha256"]:
            raise ValueError("Base checkpoint hash mismatch")
        if sha256(adapter / "adapter_model.safetensors") != protocol["adapter_sha256"]:
            raise ValueError("Adapter hash mismatch")
        _, _, _, rows, _, _, labels, screen = load_panel(protocol["grouped_plan_sha256"])
        torch.set_num_threads(4)
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        tokenizer = AutoTokenizer.from_pretrained(base, local_files_only=True, trust_remote_code=True, use_fast=False)
        model, info = AutoModel.from_pretrained(base, local_files_only=True, trust_remote_code=True,
                                               torch_dtype=torch.float16, use_flash_attn=False,
                                               output_loading_info=True)
        if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
            raise ValueError(f"Base model loading errors: {info}")
        wrapped = PeftModel.from_pretrained(model, adapter, local_files_only=True)
        expected = load_file(adapter / "adapter_model.safetensors")
        actual = get_peft_model_state_dict(wrapped)
        if expected.keys() != actual.keys() or any(not torch.equal(expected[k], actual[k].cpu()) for k in expected):
            raise ValueError("Loaded adapter tensors differ from checkpoint")
        model = wrapped.get_base_model().eval().requires_grad_(False).to("cuda")
        transform = Compose([Resize((448, 448), interpolation=InterpolationMode.BICUBIC), ToTensor(),
                             Normalize((.485, .456, .406), (.229, .224, .225))])
        report["environment"] = {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                                 "dtype": "float16", "adapter_tensors_verified": len(expected)}
        for row in rows:
            prediction = {"sample_id": row["sample_id"], "status": "ok"}
            try:
                with Image.open(ROOT / row["asset"]["file"]) as image:
                    pixel_values = transform(image.convert("RGB")).unsqueeze(0).to("cuda", dtype=torch.float16)
                with torch.inference_mode():
                    answer = model.chat(tokenizer, pixel_values, "<image>\n" + row["question"],
                                        generation_config={"max_new_tokens": 12, "do_sample": False})
                prediction["answer"] = answer
            except Exception as error:
                prediction.update(status="error", answer=None, error_type=type(error).__name__, error=str(error))
            report["predictions"].append(prediction)
            output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
            print(f"Evaluated {len(report['predictions'])}/60", flush=True)
        report["score"] = score_samples(rows, report["predictions"], labels, screen)
        report["status"] = "completed"
    except Exception as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error), traceback=traceback.format_exc())
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - start
    if torch.cuda.is_available():
        report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(report["status"], output, flush=True)
    return report["status"] != "completed"


if __name__ == "__main__":
    raise SystemExit(main())
