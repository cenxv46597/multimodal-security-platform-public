"""Check whether matched background rectangles exist for the grouped SAR panel."""

from collections import Counter
import json
import math
from pathlib import Path

import numpy as np

from evaluate_sar_grouped_development import load_panel
from prepare_sar_grouped_development import source_rows
from run_vlm_clean_screen import ROOT, sha256


def main():
    protocol_path = ROOT / "experiments/protocols/sar_occlusion_feasibility_v0.json"
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    _, _, _, validation, _, _, labels, _ = load_panel(protocol["grouped_plan_sha256"])
    annotations, _, _ = source_rows()
    by_id = {row["record"]["sample_id"]: row for row in annotations}
    results = []
    for record in validation:
        annotation = by_id[record["sample_id"]]
        width, height = record["asset"]["size"]
        boxes = []
        for x, y, w, h in annotation["boxes"]:
            box = (max(0, math.floor(x)), max(0, math.floor(y)),
                   min(width, math.ceil(x + w)), min(height, math.ceil(y + h)))
            if box[2] > box[0] and box[3] > box[1]:
                boxes.append(box)
        if not boxes:
            results.append({"sample_id": record["sample_id"], "label": record["reference_answer"],
                            "status": "no_valid_box"})
            continue
        target = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        box_width, box_height = target[2] - target[0], target[3] - target[1]
        occupied = np.zeros((height, width), dtype=np.int32)
        for x0, y0, x1, y1 in boxes:
            occupied[y0:y1, x0:x1] = 1
        integral = np.pad(occupied, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
        sums = (integral[box_height:, box_width:] - integral[:-box_height, box_width:]
                - integral[box_height:, :-box_width] + integral[:-box_height, :-box_width])
        positions = np.argwhere(sums == 0)
        result = {"sample_id": record["sample_id"], "label": record["reference_answer"],
                  "target_box": target, "annotated_box_count": len(boxes)}
        if len(positions):
            y, x = map(int, positions[0])
            result.update(status="eligible", background_box=[x, y, x + box_width, y + box_height])
        else:
            result["status"] = "no_matching_background"
        results.append(result)
    report = {"status": "completed", "protocol": protocol_path.relative_to(ROOT).as_posix(),
              "protocol_sha256": sha256(protocol_path), "samples": results,
              "per_class": {label: dict(Counter(row["status"] for row in results if row["label"] == label))
                            for label in labels}}
    output = ROOT / "resources/manifests/sar_occlusion_feasibility_20261108.json"
    with output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2)
        stream.write("\n")
    print(json.dumps(report["per_class"]))


if __name__ == "__main__":
    main()
