"""Tests for deterministic Zone Grounding Agent behavior."""

import unittest

from src.agents.zone_grounding_agent import DefaultZone, Zone, ZoneGroundingAgent
from src.detection.schemas import Detection, ZoneRecognition
from src.pipeline.baseline_pipeline import assign_background_zone_to_people


def person(object_id: str, bbox: tuple[float, float, float, float]) -> Detection:
    return Detection(object_id=object_id, class_name="person", confidence=0.9, bbox=bbox)


def agent() -> ZoneGroundingAgent:
    return ZoneGroundingAgent(
        camera_id="cam_test",
        default_zone=DefaultZone("Z00", "general_area"),
        zones=[
            Zone("Z01", "active_work_area", 30, [(0, 0), (100, 0), (100, 100), (0, 100)]),
            Zone("Z02", "restricted_area", 100, [(50, 0), (150, 0), (150, 100), (50, 100)]),
        ],
    )


class ZoneGroundingTests(unittest.TestCase):
    def test_person_inside_zone(self):
        assignment = agent().assign_person(person("P01", (10, 10, 30, 40)))
        self.assertEqual(assignment.zone_id, "Z01")
        self.assertEqual(assignment.source, "polygon")

    def test_person_outside_all_zones_gets_default_zone(self):
        assignment = agent().assign_person(person("P01", (200, 10, 240, 40)))
        self.assertEqual(assignment.zone_id, "Z00")
        self.assertEqual(assignment.source, "default")

    def test_person_on_boundary_is_inside_zone(self):
        assignment = agent().assign_person(person("P01", (90, 10, 110, 50)))
        self.assertEqual(assignment.zone_id, "Z02")

    def test_overlapping_zones_use_highest_priority(self):
        assignment = agent().assign_person(person("P01", (60, 10, 80, 50)))
        self.assertEqual(assignment.zone_id, "Z02")
        self.assertEqual(assignment.zone_type, "restricted_area")

    def test_two_persons_in_different_zones(self):
        assignments = agent().assign(
            [
                person("P01", (10, 10, 30, 40)),
                person("P02", (120, 10, 140, 40)),
            ]
        )
        self.assertEqual([assignment.zone_id for assignment in assignments], ["Z01", "Z02"])

    def test_non_person_detection_is_rejected(self):
        detection = Detection("H01", "helmet", 0.9, (10, 10, 20, 20))
        with self.assertRaises(ValueError):
            agent().assign_person(detection)

    def test_dam_background_zone_can_be_assigned_to_people(self):
        recognition = ZoneRecognition(
            zone_id="Z01",
            zone_type="active_work_area",
            confidence=0.8,
            source="dam_prompt",
            reason="active construction background",
        )
        assignments = assign_background_zone_to_people(
            [
                person("P01", (10, 10, 30, 40)),
                Detection("H01", "helmet", 0.9, (10, 10, 20, 20)),
            ],
            recognition,
        )
        self.assertEqual(len(assignments), 1)
        self.assertEqual(assignments[0].zone_id, "Z01")
        self.assertEqual(assignments[0].source, "dam_prompt")
        self.assertEqual(assignments[0].anchor_point, (20, 40))


if __name__ == "__main__":
    unittest.main()
