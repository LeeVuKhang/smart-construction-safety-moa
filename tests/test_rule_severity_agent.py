from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path

from construction_safety_moa.contracts import (
    CandidateEvent,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    Detection,
    EvidenceGateResult,
    EvidenceReasonCode,
    EvidenceRoute,
    PPEStatus,
    RegionGrounding,
)
from construction_safety_moa.routing.evidence_gate import EvidenceSufficiencyGate
from construction_safety_moa.rules.severity_agent import (
    RuleInputNotReadyError,
    RuleSeverityAgent,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_rules() -> list[dict[str, object]]:
    payload = json.loads((PROJECT_ROOT / "config" / "rules.json").read_text(encoding="utf-8"))
    return payload["rules"]


def build_candidate(
    *,
    helmet: str = "missing",
    zone_type: str = "active_work_area",
    context_evidence: list[ContextEvidence] | None = None,
) -> CandidateEvent:
    return CandidateEvent(
        event_id="EVT-RULE-001",
        worker_id="P01",
        frame_ref="F001",
        crop_ref="F001::P01::100,100,180,320",
        source_ref="frame-F001.jpg",
        detections=[
            Detection("P01", "person", [100, 100, 180, 320], 0.94),
            Detection("NH01", "no_helmet", [118, 102, 162, 148], 0.91),
            Detection("EQ01", "excavator", [220, 100, 420, 340], 0.88),
            Detection("P02", "person", [430, 100, 500, 320], 0.91),
        ],
        ppe_status=PPEStatus(
            helmet=helmet,
            confidence=0.91,
            target_id="P01",
            evidence_detection_ids=["NH01"],
        ),
        region_grounding=RegionGrounding(
            zone_id="Z-ACTIVE-01",
            zone_type=zone_type,
            spatial_relation="inside",
            confidence=0.98,
            target_id="P01",
        ),
        context_evidence=list(context_evidence or []),
    )


def relation_evidence(label: str = "NEAR", **overrides: object) -> ContextEvidence:
    values: dict[str, object] = {
        "evidence_id": f"CTXE-{label}",
        "kind": ContextEvidenceKind.LOCAL_RELATION,
        "label": label,
        "subject_detection_id": "P01",
        "object_detection_id": "EQ01",
        "frame_ref": "F001",
        "crop_ref": "F001::P01::100,100,180,320",
        "zone_ref": "Z-ACTIVE-01",
        "confidence": 0.9,
        "status": ContextEvidenceStatus.CONFIRMED,
        "reason_code": f"{label}_CONFIRMED",
    }
    values.update(overrides)
    return ContextEvidence(**values)  # type: ignore[arg-type]


def ready_gate() -> EvidenceGateResult:
    return EvidenceGateResult(
        route=EvidenceRoute.READY_FOR_RULE,
        reason_codes=[EvidenceReasonCode.EVIDENCE_SUFFICIENT],
    )


class TestRuleSeverityAgent(unittest.TestCase):
    def setUp(self) -> None:
        self.rules = load_rules()
        self.agent = RuleSeverityAgent(self.rules)

    def test_clear_missing_helmet_in_active_zone_maps_to_medium_rule(self) -> None:
        candidate = build_candidate()
        gate_result = EvidenceSufficiencyGate().evaluate(candidate)

        match = self.agent.apply(candidate, gate_result)

        self.assertEqual(gate_result.route, EvidenceRoute.READY_FOR_RULE)
        self.assertEqual(match.rule_id, "PPE_ACTIVE_ZONE_001")
        self.assertTrue(match.violation)
        self.assertEqual(match.severity, "medium")
        self.assertIn("NEAR_EQUIPMENT_RELATION_NOT_PROVIDED", match.missing_evidence)
        self.assertIn("NH01", match.evidence_refs)

    def test_visible_helmet_maps_to_safe_catalog_rule(self) -> None:
        candidate = build_candidate(helmet="visible")

        match = self.agent.apply(candidate, ready_gate())

        self.assertEqual(match.rule_id, "SAFE_HELMET_VISIBLE_001")
        self.assertFalse(match.violation)
        self.assertEqual(match.severity, "none")

    def test_valid_confirmed_near_relation_can_upgrade_candidate_severity(self) -> None:
        candidate = build_candidate(context_evidence=[relation_evidence()])

        match = self.agent.apply(candidate, ready_gate())

        self.assertEqual(match.severity, "critical")
        self.assertIn("NEAR_HEAVY_EQUIPMENT_CONFIRMED", match.reason_codes)
        self.assertIn("CTXE-NEAR", match.evidence_refs)

    def test_untrusted_relations_do_not_upgrade_severity(self) -> None:
        invalid_relations = [
            relation_evidence(status=ContextEvidenceStatus.PROVISIONAL),
            relation_evidence(subject_detection_id="P02"),
            relation_evidence(object_detection_id="MISSING-EQUIPMENT"),
            relation_evidence(object_detection_id="P02"),
            relation_evidence(frame_ref="INVENTED-FRAME"),
            relation_evidence(crop_ref="INVENTED-CROP"),
            relation_evidence(zone_ref="INVENTED-ZONE"),
        ]

        for evidence in invalid_relations:
            with self.subTest(evidence=evidence):
                match = self.agent.apply(
                    build_candidate(context_evidence=[evidence]),
                    ready_gate(),
                )
                self.assertEqual(match.severity, "medium")

    def test_non_near_relation_keeps_equipment_relation_unknown(self) -> None:
        match = self.agent.apply(
            build_candidate(context_evidence=[relation_evidence("ADJACENT")]),
            ready_gate(),
        )

        self.assertEqual(match.severity, "medium")
        self.assertIn("NEAR_EQUIPMENT_RELATION_NOT_PROVIDED", match.missing_evidence)

    def test_non_ready_route_fails_closed(self) -> None:
        not_ready = replace(
            ready_gate(),
            route=EvidenceRoute.NEEDS_CONTEXT,
        )

        with self.assertRaisesRegex(
            RuleInputNotReadyError,
            "RULE_INPUT_NOT_READY:NEEDS_CONTEXT",
        ):
            self.agent.apply(build_candidate(), not_ready)

    def test_unsupported_zone_fails_closed(self) -> None:
        candidate = build_candidate(zone_type="invented_zone")
        gate_result = EvidenceSufficiencyGate().evaluate(candidate)

        self.assertEqual(gate_result.route, EvidenceRoute.UNRESOLVABLE)
        self.assertIn(EvidenceReasonCode.UNSUPPORTED_ZONE_TYPE, gate_result.reason_codes)
        with self.assertRaisesRegex(RuleInputNotReadyError, "UNSUPPORTED_ZONE_TYPE"):
            self.agent.apply(candidate, ready_gate())

    def test_missing_catalog_rule_fails_closed(self) -> None:
        incomplete = [rule for rule in self.rules if rule["rule_id"] != "PPE_ACTIVE_ZONE_001"]

        with self.assertRaisesRegex(RuleInputNotReadyError, "RULE_NOT_CONFIGURED"):
            RuleSeverityAgent(incomplete).apply(build_candidate(), ready_gate())

    def test_supported_zones_map_only_to_catalog_rules(self) -> None:
        configured_ids = {rule["rule_id"] for rule in self.rules}
        expected = {
            "active_work_area": "PPE_ACTIVE_ZONE_001",
            "restricted_zone": "PPE_RESTRICTED_ZONE_001",
            "work_at_height": "PPE_HEIGHT_ZONE_001",
            "site_office": "PPE_OFFICE_EXCEPTION_001",
            "rest_area": "PPE_OFFICE_EXCEPTION_001",
        }

        for zone_type, expected_rule_id in expected.items():
            with self.subTest(zone_type=zone_type):
                match = self.agent.apply(
                    build_candidate(zone_type=zone_type),
                    ready_gate(),
                )
                self.assertEqual(match.rule_id, expected_rule_id)
                self.assertIn(match.rule_id, configured_ids)


if __name__ == "__main__":
    unittest.main()
