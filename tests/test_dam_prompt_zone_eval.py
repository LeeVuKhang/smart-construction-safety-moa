"""Tests for DAM prompt-based zone parsing."""

import unittest

from src.evaluation.dam_prompt_zone_eval import normalize_zone_type, parse_zone_prediction


class DamPromptZoneEvalTests(unittest.TestCase):
    def test_normalize_zone_aliases(self):
        self.assertEqual(normalize_zone_type("work-zone"), "active_work_area")
        self.assertEqual(normalize_zone_type("restricted"), "restricted_area")
        self.assertEqual(normalize_zone_type("public area"), "general_area")

    def test_parse_json_zone_type(self):
        result = parse_zone_prediction('{"zone_type": "restricted", "confidence": 0.7, "reason": "barrier"}')
        self.assertEqual(result["zone_id"], "Z02")
        self.assertEqual(result["zone_type"], "restricted_area")

    def test_parse_ambiguous_text_as_unknown(self):
        result = parse_zone_prediction("This may be a restricted work zone.")
        self.assertEqual(result["zone_type"], "unknown")

    def test_parse_single_text_signal(self):
        result = parse_zone_prediction("The background appears to be a general access area.")
        self.assertEqual(result["zone_id"], "Z00")


if __name__ == "__main__":
    unittest.main()
