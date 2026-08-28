"""Tests for DAM prompt-based zone parsing."""

import unittest

from src.agents.dam_zone_agent import DAMPromptZoneAgent, PromptZone, normalize_zone_type


class DamPromptZoneEvalTests(unittest.TestCase):
    def agent(self):
        return DAMPromptZoneAgent(
            default_zone=PromptZone("Z00", "general_area", "ordinary public access area"),
            zones=[
                PromptZone("Z01", "active_work_area", "construction area with active work"),
                PromptZone("Z02", "restricted_area", "hazard area with barriers"),
            ],
        )

    def test_normalize_zone_aliases(self):
        self.assertEqual(normalize_zone_type("work-zone"), "active_work_area")
        self.assertEqual(normalize_zone_type("restricted"), "restricted_area")
        self.assertEqual(normalize_zone_type("public area"), "general_area")

    def test_parse_json_zone_type(self):
        result = self.agent().parse_response('{"zone_type": "restricted", "confidence": 0.7, "reason": "barrier"}')
        self.assertEqual(result["zone_id"], "Z02")
        self.assertEqual(result["zone_type"], "restricted_area")

    def test_parse_json_zone_id(self):
        result = self.agent().parse_response('{"zone_id": "Z01", "confidence": 0.7, "reason": "active work"}')
        self.assertEqual(result["zone_id"], "Z01")
        self.assertEqual(result["zone_type"], "active_work_area")

    def test_parse_ambiguous_text_as_unknown(self):
        result = self.agent().parse_response("This may be a restricted work zone.")
        self.assertEqual(result["zone_type"], "unknown")

    def test_parse_single_text_signal(self):
        result = self.agent().parse_response("The background appears to be a general access area.")
        self.assertEqual(result["zone_id"], "Z00")

    def test_parse_truncated_json_zone_type(self):
        result = self.agent().parse_response(
            '{ "zone_type": "active_work_area", "confidence": 0.8, "reason": "construction area"'
        )
        self.assertEqual(result["zone_id"], "Z01")
        self.assertEqual(result["confidence"], 0.8)

    def test_prompt_includes_configured_zone_descriptions(self):
        prompt = self.agent().build_prompt("focus on background")
        self.assertIn("Z01: active_work_area = construction area with active work", prompt)
        self.assertIn("Do not describe PPE", prompt)
        self.assertIn("Additional user prompt: focus on background", prompt)


if __name__ == "__main__":
    unittest.main()
