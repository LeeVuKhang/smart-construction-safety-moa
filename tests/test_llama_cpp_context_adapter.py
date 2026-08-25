from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from construction_safety_moa.context.agent import ContextAgent
from construction_safety_moa.context.llama_cpp_adapter import (
    CONTEXT_VISION_SYSTEM_PROMPT,
    LlamaCppAdapterConfig,
    LlamaCppContextModelAdapter,
    build_context_proposal_schema,
)
from construction_safety_moa.context.media import MediaManifestEntry, MediaResolver
from construction_safety_moa.contracts import (
    ContextAction,
    ContextRequest,
    Detection,
    EvidenceIssue,
    PPEStatus,
    RegionGrounding,
)

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


class FakeTransport:
    def __init__(self, response: bytes | Exception) -> None:
        self.response = response
        self.call_count = 0
        self.payload: dict[str, object] | None = None

    def post_json(self, endpoint: str, payload: dict[str, object], timeout_seconds: float) -> bytes:
        self.call_count += 1
        self.payload = payload
        if isinstance(self.response, Exception):
            raise self.response
        return self.response


def build_request() -> ContextRequest:
    return ContextRequest(
        request_id="CTXREQ-ADAPTER-001",
        event_id="EVT-ADAPTER-001",
        worker_id="W01",
        frame_ref="FRAME-001",
        crop_ref="FRAME-001::W01::20,20,90,150",
        detections=[
            Detection("W01", "person", [20, 20, 90, 150], 0.96),
            Detection("NH01", "no_helmet", [35, 22, 70, 60], 0.93),
            Detection("EQ01", "excavator", [105, 15, 220, 165], 0.9),
        ],
        ppe_finding=PPEStatus("missing", 0.93, "W01", ["NH01"]),
        zone_grounding=RegionGrounding(
            "ZONE-01",
            "active_work_area",
            "inside",
            0.98,
            "W01",
        ),
        evidence_issues=[EvidenceIssue.RELATION_UNCLEAR],
        allowed_context_actions=[
            ContextAction.EMIT_CONTEXT_EVIDENCE,
            ContextAction.ABSTAIN,
        ],
    )


def valid_model_content(**overrides: object) -> dict[str, object]:
    content: dict[str, object] = {
        "evidence": [
            {
                "evidence_id": "CTXE-001",
                "kind": "LOCAL_RELATION",
                "label": "NEAR",
                "subject_detection_id": "W01",
                "object_detection_id": "EQ01",
                "frame_ref": "FRAME-001",
                "crop_ref": "FRAME-001::W01::20,20,90,150",
                "zone_ref": "ZONE-01",
                "confidence": 0.91,
                "status": "CONFIRMED",
                "reason_code": "LOCAL_RELATION_VISIBLE",
            }
        ],
        "selected_action": "EMIT_CONTEXT_EVIDENCE",
        "action_parameters": {"evidence_ids": ["CTXE-001"]},
        "context_confidence": 0.91,
        "model_metadata": {},
    }
    content.update(overrides)
    return content


def completion_bytes(content: object) -> bytes:
    return json.dumps(
        {
            "choices": [
                {
                    "message": {
                        "role": "assistant",
                        "content": content if isinstance(content, str) else json.dumps(content),
                    }
                }
            ]
        }
    ).encode("utf-8")


@unittest.skipIf(Image is None, "Pillow is an optional Context pilot dependency")
class TestLlamaCppContextModelAdapter(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.media_root = Path(self.temp_dir.name)
        image_path = self.media_root / "frame.png"
        Image.new("RGB", (240, 180), color=(220, 225, 230)).save(image_path, format="PNG")
        checksum = hashlib.sha256(image_path.read_bytes()).hexdigest()
        self.resolver = MediaResolver(
            self.media_root,
            {
                "FRAME-001": MediaManifestEntry(
                    logical_ref="FRAME-001",
                    relative_path=image_path.name,
                    sha256=checksum,
                    mime_type="image/png",
                    zone_polygons={
                        "ZONE-01": [[5, 5], [235, 5], [235, 175], [5, 175]],
                    },
                )
            },
        )
        self.config = LlamaCppAdapterConfig(
            endpoint="http://127.0.0.1:8080/v1/chat/completions",
            model_repository="Qwen/Qwen3-VL-2B-Instruct-GGUF",
            model_revision="test-revision-pinned",
            gguf_filename="Qwen3VL-2B-Instruct-Q4_K_M.gguf",
            gguf_sha256="a" * 64,
            quantization="Q4_K_M",
            mmproj_filename="mmproj-Qwen3VL-2B-Instruct-Q8_0.gguf",
            mmproj_sha256="b" * 64,
            llama_cpp_commit="test-commit",
            llama_cpp_build="test-build",
            timeout_seconds=0.2,
            max_tokens=256,
        )

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def _adapter(
        self,
        response: bytes | Exception,
    ) -> tuple[LlamaCppContextModelAdapter, FakeTransport]:
        transport = FakeTransport(response)
        return (
            LlamaCppContextModelAdapter(self.config, self.resolver, transport=transport),
            transport,
        )

    def test_valid_response_is_parsed_then_accepted_by_context_agent(self) -> None:
        adapter, transport = self._adapter(completion_bytes(valid_model_content()))

        result = ContextAgent(adapter).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.EMIT_CONTEXT_EVIDENCE)
        self.assertEqual(transport.call_count, 1)
        self.assertEqual(result.validation_errors, [])
        metadata = result.model_metadata
        self.assertEqual(metadata["provider"], "local")
        self.assertEqual(metadata["runtime"], "llama.cpp/llama-server")
        self.assertEqual(metadata["raw_output_validation_status"], "schema_valid")
        self.assertEqual(metadata["model_revision"], "test-revision-pinned")
        self.assertEqual(metadata["inference_parameters"]["temperature"], 0.0)
        self.assertRegex(metadata["prompt_sha256"], r"^[0-9a-f]{64}$")
        self.assertRegex(metadata["schema_sha256"], r"^[0-9a-f]{64}$")
        self.assertEqual(len(metadata["images"]), 2)
        self.assertTrue(all("base64" not in str(item).lower() for item in metadata["images"]))

        payload = transport.payload
        self.assertIsNotNone(payload)
        self.assertEqual(payload["temperature"], 0.0)
        user_content = payload["messages"][1]["content"]
        image_items = [item for item in user_content if item["type"] == "image_url"]
        self.assertEqual(len(image_items), 2)
        schema = payload["response_format"]["json_schema"]["schema"]
        self.assertEqual(
            schema["properties"]["selected_action"]["enum"],
            ["EMIT_CONTEXT_EVIDENCE", "ABSTAIN"],
        )

    def test_schema_is_limited_to_existing_contract_fields_and_p0_actions(self) -> None:
        schema = build_context_proposal_schema(build_request())

        self.assertEqual(
            set(schema["properties"]),
            {
                "evidence",
                "selected_action",
                "action_parameters",
                "context_confidence",
                "model_metadata",
            },
        )
        evidence_properties = schema["properties"]["evidence"]["items"]["properties"]
        self.assertEqual(
            set(evidence_properties),
            {
                "evidence_id",
                "kind",
                "label",
                "subject_detection_id",
                "object_detection_id",
                "frame_ref",
                "crop_ref",
                "zone_ref",
                "confidence",
                "status",
                "reason_code",
            },
        )
        self.assertNotIn("REQUEST_MORE_FRAMES", json.dumps(schema))
        self.assertNotIn("REQUEST_HIGHER_RESOLUTION_CROP", json.dumps(schema))

    def test_prompt_artifact_matches_adapter_invariant_prompt(self) -> None:
        prompt_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "construction_safety_moa"
            / "context"
            / "prompt.txt"
        )

        self.assertEqual(
            prompt_path.read_text(encoding="utf-8").strip(),
            CONTEXT_VISION_SYSTEM_PROMPT,
        )

    def test_connection_timeout_and_oom_fail_closed_without_retry(self) -> None:
        cases = {
            "connection": (ConnectionError("server unavailable"), "CONNECTION_ERROR"),
            "timeout": (TimeoutError("deadline exceeded"), "MODEL_TIMEOUT"),
            "oom": (RuntimeError("CUDA out of memory"), "MODEL_OOM"),
        }
        for name, (failure, expected_reason) in cases.items():
            with self.subTest(name=name):
                adapter, transport = self._adapter(failure)

                result = ContextAgent(adapter).analyze(build_request())

                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertEqual(result.action_parameters.reason_code, expected_reason)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(
                    result.model_metadata["raw_output_validation_status"],
                    "transport_error",
                )

    def test_invalid_outer_json_content_json_and_missing_fields_fail_closed(self) -> None:
        cases = {
            "outer_json": (b"not-json", "INVALID_RESPONSE_JSON"),
            "content_json": (completion_bytes("{not-json"), "INVALID_MODEL_JSON"),
            "schema": (
                completion_bytes({"selected_action": "ABSTAIN"}),
                "INVALID_MODEL_SCHEMA",
            ),
        }
        for name, (response, expected_reason) in cases.items():
            with self.subTest(name=name):
                adapter, transport = self._adapter(response)

                result = ContextAgent(adapter).analyze(build_request())

                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertEqual(result.action_parameters.reason_code, expected_reason)
                self.assertEqual(transport.call_count, 1)

    def test_invented_refs_and_disallowed_action_are_left_for_context_validation(self) -> None:
        invented = valid_model_content()
        invented["evidence"] = [
            {
                **invented["evidence"][0],
                "object_detection_id": "INVENTED-OBJECT",
            }
        ]
        invented_subject = valid_model_content()
        invented_subject["evidence"] = [
            {
                **invented_subject["evidence"][0],
                "subject_detection_id": "INVENTED-WORKER",
            }
        ]
        invented_frame = valid_model_content()
        invented_frame["evidence"] = [
            {**invented_frame["evidence"][0], "frame_ref": "INVENTED-FRAME"}
        ]
        invented_crop = valid_model_content()
        invented_crop["evidence"] = [
            {**invented_crop["evidence"][0], "crop_ref": "INVENTED-CROP"}
        ]
        invented_zone = valid_model_content()
        invented_zone["evidence"] = [
            {**invented_zone["evidence"][0], "zone_ref": "INVENTED-ZONE"}
        ]
        cases = {
            "object": (
                invented,
                "UNKNOWN_OBJECT_DETECTION_REF:INVENTED-OBJECT",
            ),
            "subject": (
                invented_subject,
                "SUBJECT_MUST_MATCH_WORKER:INVENTED-WORKER",
            ),
            "frame": (invented_frame, "FRAME_REF_MISMATCH:INVENTED-FRAME"),
            "crop": (invented_crop, "CROP_REF_MISMATCH:INVENTED-CROP"),
            "zone": (invented_zone, "ZONE_REF_MISMATCH:INVENTED-ZONE"),
            "disallowed": (
                valid_model_content(selected_action="SEND_CRITICAL_ALERT"),
                "UNSUPPORTED_ACTION:SEND_CRITICAL_ALERT",
            ),
        }
        for name, (content, expected_error) in cases.items():
            with self.subTest(name=name):
                adapter, transport = self._adapter(completion_bytes(content))

                result = ContextAgent(adapter).analyze(build_request())

                self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
                self.assertIn(expected_error, result.validation_errors)
                self.assertEqual(transport.call_count, 1)
                self.assertEqual(
                    result.model_metadata["raw_output_validation_status"],
                    "schema_invalid",
                )

    def test_media_failure_abstains_without_calling_server(self) -> None:
        resolver = MediaResolver(self.media_root, {})
        transport = FakeTransport(completion_bytes(valid_model_content()))
        adapter = LlamaCppContextModelAdapter(self.config, resolver, transport=transport)

        result = ContextAgent(adapter).analyze(build_request())

        self.assertEqual(result.selected_action, ContextAction.ABSTAIN)
        self.assertEqual(result.action_parameters.reason_code, "MEDIA_RESOLUTION_FAILED")
        self.assertEqual(transport.call_count, 0)
        self.assertIn(
            "MEDIA_REF_NOT_IN_MANIFEST:FRAME-001",
            result.model_metadata["media_validation_errors"],
        )

    def test_non_local_endpoint_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "endpoint must be local"):
            LlamaCppAdapterConfig(
                endpoint="https://api.example.com/v1/chat/completions",
                model_repository="example/model",
                model_revision="revision",
                gguf_filename="model.gguf",
                gguf_sha256="a" * 64,
                quantization="Q4_K_M",
                llama_cpp_commit="commit",
                llama_cpp_build="build",
            )


if __name__ == "__main__":
    unittest.main()
