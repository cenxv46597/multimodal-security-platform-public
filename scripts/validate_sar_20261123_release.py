"""Check the arithmetic and admission status in the public SAR summary."""

import json
from pathlib import Path


RECORD = (
    Path(__file__).resolve().parents[1]
    / "experiments" / "records" / "2026-11-23" / "summary.json"
)


def validate_summary(record):
    candidates = record["candidates"]
    candidate_total = candidates["total"]
    dispositions = (
        candidates["preliminary"],
        candidates["withheld_direct_development_relation"],
        candidates["pending"],
    )
    if min(dispositions) < 0 or sum(dispositions) != candidate_total:
        raise ValueError("Candidate dispositions do not reconcile")
    if candidates["blind_review_unreviewed"] != candidate_total:
        raise ValueError("Unreviewed count differs from the release scope")
    if candidates["individual_license_confirmed"] != 0 or candidates["frozen"] != 0:
        raise ValueError("This release has no individually licensed or frozen candidates")

    overlap = record["overlap"]
    candidate_pairs = overlap["candidate_pairs"]
    cross_pairs = overlap["candidate_development_pairs"]
    if candidate_pairs["total"] != candidate_total * (candidate_total - 1) // 2:
        raise ValueError("Candidate pair denominator is inconsistent")
    if cross_pairs["total"] != candidate_total * candidates["development_images"]:
        raise ValueError("Development pair denominator is inconsistent")
    for pairs in (candidate_pairs, cross_pairs):
        if not 0 <= pairs["screened"] <= pairs["total"]:
            raise ValueError("Screened pair count is out of range")
    uncovered = sum(pairs["total"] - pairs["screened"] for pairs in (candidate_pairs, cross_pairs))
    if overlap["uncovered_size_or_shape"] != uncovered:
        raise ValueError("Uncovered pair count does not reconcile")

    evidence = (
        overlap["exact_shared_gray_region"],
        overlap["near_gray_with_alignment"],
        overlap["unresolved"],
    )
    if min(evidence) < 0 or sum(evidence) != overlap["suspected"]:
        raise ValueError("Overlap evidence counts do not reconcile")
    preflight = overlap["mixed_size_preflight"]
    if preflight["status"] != "failed" or preflight["tested"] or preflight["retries"]:
        raise ValueError("The mixed-size preflight did not execute image comparisons")


if __name__ == "__main__":
    validate_summary(json.loads(RECORD.read_text(encoding="utf-8")))
    print("Public SAR summary checks passed.")
