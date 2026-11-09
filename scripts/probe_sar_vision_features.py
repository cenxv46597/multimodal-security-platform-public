"""Fit fixed linear ridge probes on frozen LLaVA image representations."""

import argparse
from copy import deepcopy
import json
from pathlib import Path
import time
import traceback

from evaluate_sar_clean_screen import score_samples
from preflight_sar_adaptation import verify_checkpoint
from run_sar_projector_pilot import (
    MANIFESTS, PILOT, PROTOCOL, RECORDS, checked_local_file, load_frozen_projector_model,
    load_projector_weights, verified_partitions,
)
from run_vlm_clean_screen import MODEL, REVISION, ROOT, get_inputs, load_json, sha256

ALPHA = 1.0
MIN_STANDARD_DEVIATION = 1e-6
SCREEN = ROOT / "experiments/protocols/sar_clean_development_screen_v0.json"
VISION_PROBE = MANIFESTS / "sar_vision_probe_results_20261107.json"


def fit_ridge(train_features, train_targets, validation_features, label_count):
    import numpy as np

    center = train_features.mean(axis=0)
    deviation = train_features.std(axis=0)
    usable = deviation > MIN_STANDARD_DEVIATION
    scale = np.where(usable, deviation, 1.0) * np.sqrt(train_features.shape[1])
    train = (train_features - center) / scale
    validation = (validation_features - center) / scale
    target = np.eye(label_count, dtype=np.float64)[train_targets]
    target_center = target.mean(axis=0)
    # Solve in the sample space, which is smaller than the feature space.
    kernel = train @ train.T + ALPHA * np.eye(len(train), dtype=np.float64)
    weights = train.T @ np.linalg.solve(kernel, target - target_center)
    train_scores = train @ weights + target_center
    validation_scores = validation @ weights + target_center
    if not np.isfinite(train_scores).all() or not np.isfinite(validation_scores).all():
        raise ValueError("Nonfinite linear probe score")
    return train_scores, validation_scores, int((~usable).sum())


def assess(records, scores, labels, screen):
    predictions = [
        {"sample_id": row["sample_id"], "status": "ok", "answer": labels[int(values.argmax())],
         "linear_scores": dict(zip(labels, values.tolist()))}
        for row, values in zip(records, scores)
    ]
    result = score_samples(records, predictions, labels, screen)
    keys = ("planned_samples", "correct", "accuracy", "per_class", "confusion", "sample_results")
    return {"predictions": predictions, **{key: result[key] for key in keys}}


def paired_counts(left, right):
    if [(row["sample_id"], row["reference"]) for row in left] != [
            (row["sample_id"], row["reference"]) for row in right]:
        raise ValueError("The predictions cover different validation rows")
    return {
        "left_only": sum(a["correct"] and not b["correct"] for a, b in zip(left, right)),
        "right_only": sum(b["correct"] and not a["correct"] for a, b in zip(left, right)),
        "both_correct": sum(a["correct"] and b["correct"] for a, b in zip(left, right)),
        "both_wrong": sum(not a["correct"] and not b["correct"] for a, b in zip(left, right)),
    }


def run(report, persist, compare_projectors):
    import numpy as np
    import torch
    from transformers import AutoProcessor

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the fixed FP16 vision encoder")
    torch.set_num_threads(4)
    train, validation, labels = verified_partitions(load_json(PROTOCOL))
    summary = load_json(PILOT)
    prior = load_json(checked_local_file(summary["source_run"], ".json"))
    records_hash = sha256(RECORDS)
    if (prior["status"] != "completed" or prior["model_revision"] != REVISION
            or prior["source_records_sha256"] != records_hash
            or prior["validation_order"] != [record["sample_id"] for record in validation]
            or prior["screen_sha256"] != sha256(SCREEN)):
        raise ValueError("The validation panel differs from the LLaVA pilot")
    weight_hash = verify_checkpoint()["model.safetensors"]
    if weight_hash != prior["base_weight_sha256"]:
        raise ValueError("The vision encoder must come from the pilot's base model")
    report.update(source_pilot=summary["source_run"], records_sha256=records_hash,
                  base_weight_sha256=weight_hash, model_revision=REVISION, labels=labels,
                  numpy_version=np.__version__, torch_version=str(torch.__version__),
                  gpu=torch.cuda.get_device_name())
    print(f"Verified {len(train)} training images and {len(validation)} development images.", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL, local_files_only=True, trust_remote_code=False, use_fast=False)
    model = load_frozen_projector_model("cpu")
    vision = model.vision_tower.to("cuda")
    layer = model.config.vision_feature_layer
    if layer != -1 or model.config.vision_feature_select_strategy != "full":
        raise ValueError("Unexpected LLaVA vision feature selection")
    config = model.config.vision_config
    patch_count = (config.image_size // config.patch_size) ** 2
    dimension = config.hidden_size
    dimensions = {"pre_projector": dimension}
    projectors = {}
    if compare_projectors:
        entry = summary["projector_checkpoint"]
        if entry["sha256"] != prior["projector_checkpoint"]["sha256"]:
            raise ValueError("The adapted projector differs from the pilot")
        checkpoint = checked_local_file(entry, ".safetensors")
        embedding_dtype = model.get_input_embeddings().weight.dtype
        if embedding_dtype != torch.float16:
            raise ValueError("Expected FP16 inputs at the language model boundary")
        original = model.multi_modal_projector
        adapted = deepcopy(original)
        load_projector_weights(adapted, checkpoint)
        projectors = {"original_projector": original.to("cuda"), "adapted_projector": adapted.to("cuda")}
        for name, projector in projectors.items():
            if projector.training or any(p.requires_grad or p.dtype != torch.float32 for p in projector.parameters()):
                raise ValueError(f"Expected a frozen FP32 projector: {name}")
            dimensions[name] = model.config.text_config.hidden_size
        report["projection"] = {
            "checkpoint": entry, "parameter_dtype": "torch.float32",
            "language_input_dtype": str(embedding_dtype),
            "pooling": "Project each patch, cast to language input dtype, then pool in FP32",
        }
    del model
    report["representation"] = {
        "encoder": "LLaVA SigLIP", "encoder_dtype": str(next(vision.parameters()).dtype),
        "selected_hidden_layer": layer, "patch_tokens": patch_count, "pooled_dimension": dimension,
        "pooling": "FP32 mean of the exact pre-projector patch features",
        "parameters": sum(p.numel() for p in vision.parameters()),
        "frozen": all(not p.requires_grad for p in vision.parameters()),
    }
    if not report["representation"]["frozen"] or vision.training:
        raise ValueError("Expected frozen SigLIP in evaluation mode")
    records = [*train, *validation]
    features = {name: [] for name in dimensions}
    started = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    with torch.inference_mode():
        for index, record in enumerate(records, 1):
            inputs = get_inputs(processor, record, "cuda", torch.float16)
            output = vision(inputs["pixel_values"], output_hidden_states=True)
            tokens = output.hidden_states[layer]
            representations = {"pre_projector": tokens}
            for name, projector in projectors.items():
                representations[name] = projector(tokens).to(dtype=embedding_dtype)
            for name, representation in representations.items():
                if tuple(representation.shape) != (1, patch_count, dimensions[name]):
                    raise ValueError(f"Unexpected feature shape for {record['sample_id']}, {name}")
                pooled = representation.float().mean(dim=1).squeeze(0).cpu().numpy()
                if not np.isfinite(pooled).all():
                    raise ValueError(f"Nonfinite feature for {record['sample_id']}, {name}")
                features[name].append(pooled)
            report["images_extracted"] = index
            if index % 30 == 0:
                print(f"Extracted {index}/{len(records)} frozen image features.", flush=True)
                persist()
    report["extraction_seconds"] = time.perf_counter() - started
    report["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
    report["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
    train_targets = [labels.index(record["reference_answer"]) for record in train]
    screen = load_json(SCREEN)
    fitted = {}
    for name, values in features.items():
        matrix = np.asarray(values, dtype=np.float64)
        if matrix.shape != (len(records), dimensions[name]):
            raise ValueError(f"Feature records are incomplete: {name}")
        # Each representation is independently fitted on the same training block.
        train_scores, validation_scores, constant = fit_ridge(
            matrix[:len(train)], train_targets, matrix[len(train):], len(labels))
        fitted[name] = {
            "dimension": dimensions[name], "near_constant_dimensions": constant,
            "training": assess(train, train_scores, labels, screen),
            "validation": assess(validation, validation_scores, labels, screen),
        }
    pre = fitted.pop("pre_projector")
    report["fit"]["near_constant_dimensions"] = pre["near_constant_dimensions"]
    report["training"] = pre["training"]
    report["validation"] = pre["validation"]
    past = prior["after"]["score"]["sample_results"]
    current = report["validation"]["sample_results"]
    paired = paired_counts(past, current)
    report["paired_against_adapted_llava"] = {
        "llava_correct": sum(row["correct"] for row in past),
        "probe_correct_llava_wrong": paired["right_only"], "probe_wrong_llava_correct": paired["left_only"],
        "both_correct": paired["both_correct"], "both_wrong": paired["both_wrong"],
    }
    if compare_projectors:
        previous_source = load_json(VISION_PROBE)["source_run"]
        previous_probe = load_json(checked_local_file(previous_source, ".json"))
        if (previous_probe["records_sha256"] != records_hash
                or previous_probe["base_weight_sha256"] != weight_hash):
            raise ValueError("Previous pre-projector probe used different inputs")
        replay = {}
        for split in ("training", "validation"):
            old_answers = previous_probe[split]["predictions"]
            new_answers = pre[split]["predictions"]
            if [row["sample_id"] for row in old_answers] != [row["sample_id"] for row in new_answers]:
                raise ValueError("The pre-projector replay used a different sample order")
            replay[split] = {"same_answers": sum(a["answer"] == b["answer"] for a, b in zip(old_answers, new_answers)),
                             "samples": len(new_answers)}
        report["pre_probe_replay"] = {"source_run": previous_source, **replay}
        for name, item in fitted.items():
            item["paired_against_pre"] = paired_counts(current, item["validation"]["sample_results"])
        report["projector_comparison"] = fitted
        report["paired_adapted_against_original_projector"] = paired_counts(
            fitted["original_projector"]["validation"]["sample_results"],
            fitted["adapted_projector"]["validation"]["sample_results"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compare-projectors", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT / "runs") or output.suffix != ".json" or output.exists():
        parser.error("Choose a new JSON report path under runs/")
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "status": "running", "scope": "Image-level linear readout diagnosis on the fixed development split",
        "fit": {
            "method": "Float64 linear ridge with centered one-hot targets and unpenalized intercept",
            "ridge_alpha": ALPHA, "near_constant_std_threshold": MIN_STANDARD_DEVIATION,
            "normalization": "Train-only mean and standard deviation, then divide by sqrt(feature dimension)",
        },
    }
    started = time.perf_counter()

    def persist():
        temporary = output.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                             encoding="utf-8")
        temporary.replace(output)

    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream)
    try:
        run(report, persist, args.compare_projectors)
        report["status"] = "completed"
    except (Exception, KeyboardInterrupt) as error:
        report.update(status="failed", error_type=type(error).__name__, error=str(error))
        traceback.print_exc()
    report["seconds"] = time.perf_counter() - started
    persist()
    print(f"{report['status']}: {output}", flush=True)
    return 0 if report["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
