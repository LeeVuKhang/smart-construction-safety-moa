# ruff: noqa: E501

from __future__ import annotations

import base64
import hashlib
import ipaddress
import json
import socket
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Protocol
from urllib.parse import urlsplit

from construction_safety_moa.context.media import MediaResolver, ResolvedContextMedia
from construction_safety_moa.contracts import (
    ContextAction,
    ContextEvidence,
    ContextEvidenceKind,
    ContextEvidenceStatus,
    ContextProposal,
    ContextRequest,
)

P0_CONTEXT_ACTIONS = (
    ContextAction.EMIT_CONTEXT_EVIDENCE,
    ContextAction.ABSTAIN,
)
P0_RELATION_LABELS = ("NEAR", "ADJACENT", "OVERLAPPING")

CONTEXT_VISION_SYSTEM_PROMPT = """You are the bounded visual Context model in a fail-closed construction-safety pipeline.

The worker to assess is exactly request.worker_id. Use only the images presented in this call and only detection IDs already present in the request. The annotated frame labels the existing worker, existing object detections, and the existing zone boundary. The crop is derived deterministically from the worker bbox.

You may only propose an action listed in allowed_context_actions. In this P0 pilot that means either EMIT_CONTEXT_EVIDENCE or ABSTAIN. For EMIT_CONTEXT_EVIDENCE, emit at least one CONFIRMED LOCAL_RELATION with label NEAR, ADJACENT, or OVERLAPPING and select its evidence_id in action_parameters.evidence_ids. Evidence must exactly reuse the request worker, object detection, frame, crop, and zone refs.

Use these operational visual meanings consistently. OVERLAPPING means the visible 2D subject and object silhouettes intersect. ADJACENT means they are immediately beside each other with only a very small visible gap and no overlap. NEAR means they are clearly close in the same local scene but remain visibly separated. If they are far apart, the target is ambiguous, or perspective, darkness, blur, truncation, or occlusion prevents a confident distinction, return ABSTAIN.

Do not decide or change PPE status, detector confidence, zone grounding, a safety rule, a violation, severity, alerting, or reporting. Do not create a worker, object, bbox, frame ref, crop ref, zone ref, or source ref. Do not infer a relation from metadata alone. If the presented pixels do not clearly establish a permitted relation for the specified worker and an existing object, return ABSTAIN with no evidence.

Return only one JSON object matching the supplied schema. Do not include prose, markdown, secrets, image data, or local paths in the response."""


class JsonHttpTransport(Protocol):
    def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> bytes:
        """POST JSON once and return the response body."""


class ContextTransportError(RuntimeError):
    def __init__(self, category: str, *, http_status: int | None = None) -> None:
        super().__init__(category)
        self.category = category
        self.http_status = http_status


class UrllibJsonHttpTransport:
    """Small dependency-free transport for the local OpenAI-compatible endpoint."""

    MAX_ERROR_BODY_BYTES = 4096

    def post_json(
        self,
        endpoint: str,
        payload: dict[str, object],
        timeout_seconds: float,
    ) -> bytes:
        request = urllib.request.Request(
            endpoint,
            data=json.dumps(payload, separators=(",", ":")).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
                return response.read()
        except urllib.error.HTTPError as error:
            try:
                body = error.read(self.MAX_ERROR_BODY_BYTES)
            except OSError:
                body = b""
            category = (
                "MODEL_OOM"
                if _looks_like_oom(body.decode("utf-8", errors="ignore"))
                else "HTTP_ERROR"
            )
            raise ContextTransportError(category, http_status=error.code) from error
        except urllib.error.URLError as error:
            if isinstance(error.reason, (TimeoutError, socket.timeout)):
                raise TimeoutError("local model request timed out") from error
            raise ConnectionError("local model endpoint unavailable") from error
        except TimeoutError as error:
            raise TimeoutError("local model request timed out") from error


@dataclass(frozen=True)
class LlamaCppAdapterConfig:
    endpoint: str
    model_repository: str
    model_revision: str
    gguf_filename: str
    gguf_sha256: str
    quantization: str
    llama_cpp_commit: str
    llama_cpp_build: str
    mmproj_filename: str | None = None
    mmproj_sha256: str | None = None
    timeout_seconds: float = 30.0
    temperature: float = 0.0
    max_tokens: int = 768
    seed: int = 0
    supports_json_schema: bool = True

    def __post_init__(self) -> None:
        parsed = urlsplit(self.endpoint)
        if (
            parsed.scheme != "http"
            or parsed.hostname is None
            or not _is_loopback_host(parsed.hostname)
            or parsed.username is not None
            or parsed.password is not None
            or bool(parsed.query)
            or bool(parsed.fragment)
        ):
            raise ValueError("llama.cpp endpoint must be local HTTP without credentials")
        required_text = {
            "model_repository": self.model_repository,
            "model_revision": self.model_revision,
            "gguf_filename": self.gguf_filename,
            "quantization": self.quantization,
            "llama_cpp_commit": self.llama_cpp_commit,
            "llama_cpp_build": self.llama_cpp_build,
        }
        if any(not value.strip() for value in required_text.values()):
            raise ValueError("model and llama.cpp revision metadata must be pinned")
        if not _valid_sha256(self.gguf_sha256):
            raise ValueError("gguf_sha256 must be a 64-character hexadecimal checksum")
        if (self.mmproj_filename is None) != (self.mmproj_sha256 is None):
            raise ValueError("mmproj filename and checksum must be supplied together")
        if self.mmproj_sha256 is not None and not _valid_sha256(self.mmproj_sha256):
            raise ValueError("mmproj_sha256 must be a 64-character hexadecimal checksum")
        if self.timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        if self.temperature != 0.0:
            raise ValueError("P0 Context inference must use temperature 0")
        if self.max_tokens < 1:
            raise ValueError("max_tokens must be positive")


def build_context_proposal_schema(request: ContextRequest) -> dict[str, Any]:
    """Build the request-narrowed P0 JSON schema sent to llama-server."""

    allowed_actions = [
        action.value for action in request.allowed_context_actions if action in P0_CONTEXT_ACTIONS
    ]
    detection_ids = [detection.object_id for detection in request.detections]
    object_ids = [object_id for object_id in detection_ids if object_id != request.worker_id]
    object_ref_schema: dict[str, Any] = {"type": "null"}
    if object_ids:
        object_ref_schema = {
            "oneOf": [
                {"type": "null"},
                {"type": "string", "enum": object_ids},
            ]
        }

    evidence_properties = {
        "evidence_id": {"type": "string", "minLength": 1, "maxLength": 128},
        "kind": {"type": "string", "enum": ["LOCAL_RELATION"]},
        "label": {"type": "string", "enum": list(P0_RELATION_LABELS)},
        "subject_detection_id": {"type": "string", "enum": [request.worker_id]},
        "object_detection_id": object_ref_schema,
        "frame_ref": {"type": "string", "enum": [request.frame_ref]},
        "crop_ref": {"type": "string", "enum": [request.crop_ref]},
        "zone_ref": {"type": "string", "enum": [request.zone_grounding.zone_id]},
        "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "status": {"type": "string", "enum": ["CONFIRMED"]},
        "reason_code": {"type": "string", "minLength": 1, "maxLength": 128},
    }
    evidence_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": list(evidence_properties),
        "properties": evidence_properties,
    }
    emit_parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": ["evidence_ids"],
        "properties": {
            "evidence_ids": {
                "type": "array",
                "minItems": 1,
                "uniqueItems": True,
                "items": {"type": "string", "minLength": 1, "maxLength": 128},
            }
        },
    }
    abstain_parameters = {
        "type": "object",
        "additionalProperties": False,
        "required": ["reason_code"],
        "properties": {"reason_code": {"type": "string", "minLength": 1, "maxLength": 128}},
    }
    properties: dict[str, Any] = {
        "evidence": {"type": "array", "maxItems": 8, "items": evidence_schema},
        "selected_action": {"type": "string", "enum": allowed_actions},
        "action_parameters": {
            "oneOf": [emit_parameters, abstain_parameters],
        },
        "context_confidence": {
            "type": "number",
            "minimum": 0.0,
            "maximum": 1.0,
        },
        "model_metadata": {
            "type": "object",
            "additionalProperties": False,
            "properties": {},
        },
    }
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
        "allOf": [
            {
                "if": {"properties": {"selected_action": {"const": "EMIT_CONTEXT_EVIDENCE"}}},
                "then": {
                    "properties": {
                        "evidence": {"minItems": 1},
                        "action_parameters": emit_parameters,
                    }
                },
            },
            {
                "if": {"properties": {"selected_action": {"const": "ABSTAIN"}}},
                "then": {
                    "properties": {
                        "evidence": {"maxItems": 0},
                        "action_parameters": abstain_parameters,
                    }
                },
            },
        ],
    }


class LlamaCppContextModelAdapter:
    """One-call local llama.cpp multimodal adapter behind ContextModelAdapter."""

    def __init__(
        self,
        config: LlamaCppAdapterConfig,
        media_resolver: MediaResolver,
        *,
        transport: JsonHttpTransport | None = None,
    ) -> None:
        self.config = config
        self.media_resolver = media_resolver
        self.transport = transport or UrllibJsonHttpTransport()

    def analyze(self, request: ContextRequest) -> ContextProposal:
        started = time.perf_counter()
        resolved = self.media_resolver.resolve(request)
        if not resolved.ready:
            return self._abstain(
                "MEDIA_RESOLUTION_FAILED",
                started,
                resolved=resolved,
                validation_status="media_validation_failed",
                extra_metadata={
                    "media_validation_errors": list(resolved.validation_errors),
                },
            )

        allowed_p0_actions = [
            action for action in request.allowed_context_actions if action in P0_CONTEXT_ACTIONS
        ]
        if not allowed_p0_actions:
            return self._abstain(
                "NO_P0_ACTION_AUTHORIZED",
                started,
                resolved=resolved,
                validation_status="not_called",
            )

        schema = build_context_proposal_schema(request)
        schema_text = _canonical_json(schema)
        grounding_text = self._grounding_text(request, resolved)
        prompt_text = f"{CONTEXT_VISION_SYSTEM_PROMPT}\n\n{grounding_text}"
        prompt_sha256 = hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()
        schema_sha256 = hashlib.sha256(schema_text.encode("utf-8")).hexdigest()
        payload = self._payload(grounding_text, schema, resolved)

        try:
            response_bytes = self.transport.post_json(
                self.config.endpoint,
                payload,
                self.config.timeout_seconds,
            )
        except Exception as error:  # local runtime is still an untrusted boundary
            reason_code, status = self._transport_failure(error)
            extra: dict[str, object] = {"transport_error_type": type(error).__name__}
            if isinstance(error, ContextTransportError) and error.http_status is not None:
                extra["http_status"] = error.http_status
            return self._abstain(
                reason_code,
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status=status,
                extra_metadata=extra,
            )

        try:
            response = json.loads(response_bytes.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            return self._abstain(
                "INVALID_RESPONSE_JSON",
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status="invalid_response_json",
            )
        content = self._completion_content(response)
        if content is None:
            return self._abstain(
                "INVALID_RESPONSE_SCHEMA",
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status="invalid_response_schema",
            )
        try:
            raw_proposal = json.loads(content)
        except json.JSONDecodeError:
            return self._abstain(
                "INVALID_MODEL_JSON",
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status="invalid_json",
            )

        proposal, structural_errors = self._parse_proposal(raw_proposal)
        if proposal is None:
            return self._abstain(
                "INVALID_MODEL_SCHEMA",
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status="schema_invalid",
                extra_metadata={"raw_output_validation_errors": structural_errors},
            )

        schema_errors = self._schema_errors(request, raw_proposal, proposal)
        metadata = self._metadata(
            started,
            resolved=resolved,
            prompt_sha256=prompt_sha256,
            schema_sha256=schema_sha256,
            validation_status="schema_valid" if not schema_errors else "schema_invalid",
            extra_metadata={"raw_output_validation_errors": schema_errors},
        )
        if schema_errors and not self._context_can_reject(schema_errors):
            return ContextProposal(
                evidence=[],
                selected_action=ContextAction.ABSTAIN,
                action_parameters={"reason_code": "INVALID_MODEL_SCHEMA"},
                context_confidence=0.0,
                model_metadata=metadata,
            )
        proposal.model_metadata = metadata
        return proposal

    def _payload(
        self,
        grounding_text: str,
        schema: dict[str, Any],
        resolved: ResolvedContextMedia,
    ) -> dict[str, object]:
        content: list[dict[str, object]] = [{"type": "text", "text": grounding_text}]
        for artifact in resolved.artifacts:
            encoded = base64.b64encode(artifact.data).decode("ascii")
            content.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:{artifact.mime_type};base64,{encoded}",
                    },
                }
            )
        payload: dict[str, object] = {
            "model": self.config.gguf_filename,
            "messages": [
                {"role": "system", "content": CONTEXT_VISION_SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            "temperature": self.config.temperature,
            "max_tokens": self.config.max_tokens,
            "seed": self.config.seed,
            "stream": False,
        }
        if self.config.supports_json_schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "context_proposal",
                    "strict": True,
                    "schema": schema,
                },
            }
        return payload

    def _grounding_text(
        self,
        request: ContextRequest,
        resolved: ResolvedContextMedia,
    ) -> str:
        model_request = {
            "request_id": request.request_id,
            "worker_id": request.worker_id,
            "frame_ref": request.frame_ref,
            "crop_ref": request.crop_ref,
            "zone_ref": request.zone_grounding.zone_id,
            "zone_type": request.zone_grounding.zone_type,
            "detections": [
                {
                    "object_id": detection.object_id,
                    "class_label": detection.class_label,
                    "bbox": detection.bbox,
                }
                for detection in request.detections
            ],
            "allowed_context_actions": [
                action.value
                for action in request.allowed_context_actions
                if action in P0_CONTEXT_ACTIONS
            ],
            "presented_media": {
                "frame_refs": resolved.presented_frame_refs,
                "crop_refs": resolved.presented_crop_refs,
                "available_but_not_acquired_refs": (resolved.available_but_not_acquired_refs),
                "images": [artifact.metadata() for artifact in resolved.artifacts],
            },
        }
        return "Grounded request JSON:\n" + _canonical_json(model_request)

    def _completion_content(self, response: object) -> str | None:
        if not isinstance(response, dict):
            return None
        choices = response.get("choices")
        if not isinstance(choices, list) or len(choices) != 1:
            return None
        choice = choices[0]
        if not isinstance(choice, dict):
            return None
        message = choice.get("message")
        if not isinstance(message, dict):
            return None
        content = message.get("content")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = [
                item.get("text")
                for item in content
                if isinstance(item, dict)
                and item.get("type") in {"text", "output_text"}
                and isinstance(item.get("text"), str)
            ]
            if text_parts:
                return "".join(text_parts)
        return None

    def _parse_proposal(
        self,
        raw: object,
    ) -> tuple[ContextProposal | None, list[str]]:
        required = {
            "evidence",
            "selected_action",
            "action_parameters",
            "context_confidence",
            "model_metadata",
        }
        if not isinstance(raw, dict):
            return None, ["TOP_LEVEL_NOT_OBJECT"]
        errors: list[str] = []
        if set(raw) != required:
            errors.append("TOP_LEVEL_FIELDS_MISMATCH")
        evidence_items = raw.get("evidence")
        if not isinstance(evidence_items, list):
            errors.append("EVIDENCE_NOT_ARRAY")
            evidence_items = []
        action = raw.get("selected_action")
        if not isinstance(action, str):
            errors.append("ACTION_NOT_STRING")
        parameters = raw.get("action_parameters")
        if not isinstance(parameters, dict):
            errors.append("ACTION_PARAMETERS_NOT_OBJECT")
        confidence = raw.get("context_confidence")
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            errors.append("CONTEXT_CONFIDENCE_NOT_NUMBER")
        raw_metadata = raw.get("model_metadata")
        if not isinstance(raw_metadata, dict):
            errors.append("MODEL_METADATA_NOT_OBJECT")
        elif raw_metadata:
            errors.append("MODEL_METADATA_MUST_BE_EMPTY")

        evidence: list[ContextEvidence] = []
        evidence_fields = {
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
        }
        for index, item in enumerate(evidence_items):
            prefix = f"EVIDENCE_{index}"
            if not isinstance(item, dict):
                errors.append(f"{prefix}_NOT_OBJECT")
                continue
            if set(item) != evidence_fields:
                errors.append(f"{prefix}_FIELDS_MISMATCH")
                continue
            string_fields = evidence_fields - {"object_detection_id", "confidence"}
            if any(not isinstance(item[field], str) for field in string_fields):
                errors.append(f"{prefix}_STRING_FIELD_INVALID")
                continue
            object_ref = item["object_detection_id"]
            if object_ref is not None and not isinstance(object_ref, str):
                errors.append(f"{prefix}_OBJECT_REF_INVALID_TYPE")
                continue
            item_confidence = item["confidence"]
            if isinstance(item_confidence, bool) or not isinstance(item_confidence, (int, float)):
                errors.append(f"{prefix}_CONFIDENCE_INVALID_TYPE")
                continue
            evidence.append(
                ContextEvidence(
                    evidence_id=item["evidence_id"],
                    kind=item["kind"],
                    label=item["label"],
                    subject_detection_id=item["subject_detection_id"],
                    object_detection_id=object_ref,
                    frame_ref=item["frame_ref"],
                    crop_ref=item["crop_ref"],
                    zone_ref=item["zone_ref"],
                    confidence=float(item_confidence),
                    status=item["status"],
                    reason_code=item["reason_code"],
                )
            )
        if errors:
            return None, errors
        return (
            ContextProposal(
                evidence=evidence,
                selected_action=action,
                action_parameters=dict(parameters),
                context_confidence=float(confidence),
                model_metadata={},
            ),
            [],
        )

    def _schema_errors(
        self,
        request: ContextRequest,
        raw: dict[str, Any],
        proposal: ContextProposal,
    ) -> list[str]:
        errors: list[str] = []
        allowed_actions = {
            action.value
            for action in request.allowed_context_actions
            if action in P0_CONTEXT_ACTIONS
        }
        raw_action = str(proposal.selected_action)
        if raw_action not in allowed_actions:
            errors.append(f"ACTION_ENUM:{raw_action}")

        detection_ids = {detection.object_id for detection in request.detections}
        for index, evidence in enumerate(proposal.evidence):
            prefix = f"EVIDENCE_{index}"
            if evidence.kind != ContextEvidenceKind.LOCAL_RELATION.value:
                errors.append(f"{prefix}_KIND_ENUM")
            if evidence.label not in P0_RELATION_LABELS:
                errors.append(f"{prefix}_LABEL_ENUM")
            if evidence.status != ContextEvidenceStatus.CONFIRMED.value:
                errors.append(f"{prefix}_STATUS_ENUM")
            if evidence.subject_detection_id != request.worker_id:
                errors.append(f"{prefix}_SUBJECT_REF")
            if evidence.object_detection_id not in detection_ids:
                errors.append(f"{prefix}_OBJECT_REF")
            if evidence.frame_ref != request.frame_ref:
                errors.append(f"{prefix}_FRAME_REF")
            if evidence.crop_ref != request.crop_ref:
                errors.append(f"{prefix}_CROP_REF")
            if evidence.zone_ref != request.zone_grounding.zone_id:
                errors.append(f"{prefix}_ZONE_REF")
            if not 0.0 <= evidence.confidence <= 1.0:
                errors.append(f"{prefix}_CONFIDENCE_RANGE")

        parameters = proposal.action_parameters
        if raw_action == ContextAction.EMIT_CONTEXT_EVIDENCE.value:
            if not proposal.evidence:
                errors.append("EMIT_EVIDENCE_EMPTY")
            if set(parameters) != {"evidence_ids"}:
                errors.append("EMIT_PARAMETERS_FIELDS")
            evidence_ids = parameters.get("evidence_ids")
            if (
                not isinstance(evidence_ids, list)
                or not evidence_ids
                or any(not isinstance(item, str) for item in evidence_ids)
            ):
                errors.append("EMIT_EVIDENCE_IDS_INVALID")
        elif raw_action == ContextAction.ABSTAIN.value:
            if proposal.evidence:
                errors.append("ABSTAIN_EVIDENCE_NOT_EMPTY")
            if set(parameters) != {"reason_code"}:
                errors.append("ABSTAIN_PARAMETERS_FIELDS")
            if not isinstance(parameters.get("reason_code"), str) or not parameters.get(
                "reason_code"
            ):
                errors.append("ABSTAIN_REASON_INVALID")

        if not 0.0 <= proposal.context_confidence <= 1.0:
            errors.append("CONTEXT_CONFIDENCE_RANGE")
        if raw.get("model_metadata") != {}:
            errors.append("MODEL_METADATA_NOT_EMPTY")
        return list(dict.fromkeys(errors))

    def _context_can_reject(self, errors: list[str]) -> bool:
        forwardable_markers = (
            "ACTION_ENUM:",
            "_KIND_ENUM",
            "_LABEL_ENUM",
            "_STATUS_ENUM",
            "_SUBJECT_REF",
            "_OBJECT_REF",
            "_FRAME_REF",
            "_CROP_REF",
            "_ZONE_REF",
            "_CONFIDENCE_RANGE",
        )
        return bool(errors) and all(
            any(marker in error for marker in forwardable_markers) for error in errors
        )

    def _transport_failure(self, error: Exception) -> tuple[str, str]:
        if isinstance(error, ContextTransportError):
            return error.category, "transport_error"
        if isinstance(error, (TimeoutError, socket.timeout)):
            return "MODEL_TIMEOUT", "transport_error"
        if isinstance(error, ConnectionError):
            return "CONNECTION_ERROR", "transport_error"
        if _looks_like_oom(str(error)):
            return "MODEL_OOM", "transport_error"
        return "MODEL_TRANSPORT_ERROR", "transport_error"

    def _abstain(
        self,
        reason_code: str,
        started: float,
        *,
        resolved: ResolvedContextMedia | None = None,
        prompt_sha256: str | None = None,
        schema_sha256: str | None = None,
        validation_status: str,
        extra_metadata: dict[str, object] | None = None,
    ) -> ContextProposal:
        return ContextProposal(
            evidence=[],
            selected_action=ContextAction.ABSTAIN,
            action_parameters={"reason_code": reason_code},
            context_confidence=0.0,
            model_metadata=self._metadata(
                started,
                resolved=resolved,
                prompt_sha256=prompt_sha256,
                schema_sha256=schema_sha256,
                validation_status=validation_status,
                extra_metadata=extra_metadata,
            ),
        )

    def _metadata(
        self,
        started: float,
        *,
        resolved: ResolvedContextMedia | None,
        prompt_sha256: str | None,
        schema_sha256: str | None,
        validation_status: str,
        extra_metadata: dict[str, object] | None = None,
    ) -> dict[str, object]:
        metadata: dict[str, object] = {
            "provider": "local",
            "runtime": "llama.cpp/llama-server",
            "endpoint": self.config.endpoint,
            "model_repository": self.config.model_repository,
            "model_revision": self.config.model_revision,
            "gguf_filename": self.config.gguf_filename,
            "gguf_sha256": self.config.gguf_sha256,
            "quantization": self.config.quantization,
            "mmproj_filename": self.config.mmproj_filename,
            "mmproj_sha256": self.config.mmproj_sha256,
            "llama_cpp_commit": self.config.llama_cpp_commit,
            "llama_cpp_build": self.config.llama_cpp_build,
            "prompt_sha256": prompt_sha256,
            "schema_sha256": schema_sha256,
            "images": (
                [artifact.metadata() for artifact in resolved.artifacts]
                if resolved is not None
                else []
            ),
            "presented_frame_refs": (resolved.presented_frame_refs if resolved is not None else []),
            "presented_crop_refs": (resolved.presented_crop_refs if resolved is not None else []),
            "media_resolution": (
                {
                    "resolved_at": resolved.resolved_at,
                    "available_but_not_acquired_refs": (resolved.available_but_not_acquired_refs),
                    "validation_errors": resolved.validation_errors,
                }
                if resolved is not None
                else None
            ),
            "inference_parameters": {
                "temperature": self.config.temperature,
                "max_tokens": self.config.max_tokens,
                "seed": self.config.seed,
                "timeout_seconds": self.config.timeout_seconds,
                "max_model_calls": 1,
                "json_schema_requested": self.config.supports_json_schema,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000.0, 3),
            "raw_output_validation_status": validation_status,
        }
        if extra_metadata:
            metadata.update(extra_metadata)
        return metadata


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _valid_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdefABCDEF" for character in value)
    )


def _is_loopback_host(hostname: str) -> bool:
    if hostname.lower() == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def _looks_like_oom(message: str) -> bool:
    lowered = message.lower()
    return "out of memory" in lowered or "cuda oom" in lowered
