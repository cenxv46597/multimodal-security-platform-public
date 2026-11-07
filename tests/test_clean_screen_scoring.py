"""Checks the research-critical denominator, ID join, parser and class gate."""
import copy
import json
from pathlib import Path
import unittest

from scripts.evaluate_sar_clean_screen import parse_label, score_samples

ROOT = Path(__file__).resolve().parents[1]


class CleanScreenScoringTests(unittest.TestCase):
    def setUp(self):
        self.task = json.loads((ROOT / "experiments/protocols/sar_single_category_v0.json").read_text(encoding="utf-8"))
        self.screen = json.loads((ROOT / "experiments/protocols/sar_clean_development_screen_v0.json").read_text(encoding="utf-8"))
        self.labels = self.task["labels"]
        self.samples = [{"sample_id": f"{label}-{i}", "reference_answer": label}
                        for label in self.labels for i in range(10)]

    def score(self, predictions, samples=None):
        return score_samples(samples or self.samples, predictions, self.labels, self.screen)

    def test_perfect_predictions_can_be_shuffled(self):
        predictions = [{"sample_id": s["sample_id"], "status": "ok", "answer": s["reference_answer"]}
                       for s in reversed(self.samples)]
        result = self.score(predictions)
        self.assertTrue(result["gate"]["passed"])
        self.assertEqual(result["correct"], 60)

    def test_always_ship_has_one_sixth_accuracy_and_fails_gate(self):
        result = self.score([{"sample_id": s["sample_id"], "status": "ok", "answer": "ship"} for s in self.samples])
        self.assertAlmostEqual(result["accuracy"], 1 / 6)
        self.assertEqual(result["per_class"]["tank"]["recall"], 0)
        self.assertFalse(result["gate"]["passed"])

    def test_high_total_with_one_class_all_wrong_fails(self):
        predictions = [{"sample_id": s["sample_id"], "status": "ok",
                        "answer": "ship" if s["reference_answer"] == "tank" else s["reference_answer"]}
                       for s in self.samples]
        result = self.score(predictions)
        self.assertAlmostEqual(result["accuracy"], 5 / 6)
        self.assertFalse(result["gate"]["criteria"]["per_class_recall"])

    def test_missing_invalid_and_error_stay_in_denominator(self):
        predictions = [{"sample_id": s["sample_id"], "status": "ok", "answer": s["reference_answer"]}
                       for s in self.samples[3:]]
        predictions.extend([{"sample_id": self.samples[1]["sample_id"], "status": "ok", "answer": "not ship"},
                            {"sample_id": self.samples[2]["sample_id"], "status": "error", "answer": "ship"}])
        result = self.score(predictions)
        self.assertEqual(result["planned_samples"], 60)
        self.assertEqual(result["correct"], 57)
        self.assertEqual(result["status_counts"], {"missing": 1, "invalid_answer": 1, "call_error": 1, "valid": 57})
        self.assertFalse(result["gate"]["criteria"]["complete_prediction_accounting"])

    def test_exact_parser_rejects_negation_multiple_labels_and_prose(self):
        self.assertEqual(parse_label(" SHIP\n", self.labels), "ship")
        for answer in ("not ship", "ship, bridge", "a ship", "ship.", "ships", None):
            self.assertIsNone(parse_label(answer, self.labels))

    def test_duplicate_and_unknown_ids_are_rejected(self):
        pred = {"sample_id": self.samples[0]["sample_id"], "status": "ok", "answer": "ship"}
        with self.assertRaises(ValueError):
            self.score([pred, copy.deepcopy(pred)])
        with self.assertRaises(ValueError):
            self.score([{"sample_id": "unknown", "status": "ok", "answer": "ship"}])

    def test_incomplete_class_coverage_has_no_six_class_macro_score(self):
        samples = self.samples[:10]
        result = self.score([], samples)
        self.assertIsNone(result["six_class_macro_recall"])
        self.assertFalse(result["gate"]["criteria"]["class_support"])


if __name__ == "__main__":
    unittest.main()
