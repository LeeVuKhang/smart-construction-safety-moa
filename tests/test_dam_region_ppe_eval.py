"""Tests for Describe Anything region-level PPE parsing."""

import unittest

from src.evaluation.dam_region_ppe_eval import normalize_label, parse_prediction


class DamRegionPpeEvalTests(unittest.TestCase):
    def test_normalize_label_maps_aliases(self):
        self.assertEqual(normalize_label("hard hat"), "helmet")
        self.assertEqual(normalize_label("without-helmet"), "no_helmet")
        self.assertEqual(normalize_label("worker"), "person")

    def test_parse_prediction_prefers_json_label(self):
        result = parse_prediction('{"label": "hard_hat", "confidence": 0.8, "reason": "visible"}')
        self.assertEqual(result["label"], "helmet")
        self.assertEqual(result["confidence"], 0.8)

    def test_parse_prediction_keyword_fallback(self):
        result = parse_prediction("The worker is not wearing a helmet.")
        self.assertEqual(result["label"], "no_helmet")

    def test_parse_prediction_rejects_schema_enum_label(self):
        result = parse_prediction('{"label":"person|helmet|no_helmet|other","confidence":0.0,"reason":"schema"}')
        self.assertEqual(result["label"], "unknown")


if __name__ == "__main__":
    unittest.main()
