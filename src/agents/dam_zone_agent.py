"""DAM-based prompt zone recognition for construction-site frames."""

from __future__ import annotations

import base64
import json
import re
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import requests
import yaml

from src.detection.schemas import ZoneRecognition


@dataclass(frozen=True)
class PromptZone:
    """Configured semantic zone option for prompt-based recognition."""

    zone_id: str
    zone_type: str
    description: str


@dataclass(frozen=True)
class DAMZoneClientConfig:
    """DAM OpenAI-compatible endpoint settings."""

    server_url: str = "http://localhost:8000"
    model: str = "describe_anything_model"
    timeout: int = 180
    max_tokens: int = 96
    temperature: float = 0.0
    top_p: float = 0.5


class DAMPromptZoneAgent:
    """Recognize a frame-level background safety zone with DAM."""

    def __init__(
        self,
        zones: list[PromptZone],
        default_zone: PromptZone,
        client_config: DAMZoneClientConfig | None = None,
    ):
        self.zones = zones
        self.default_zone = default_zone
        self.client_config = client_config or DAMZoneClientConfig()

    @classmethod
    def from_yaml(
        cls,
        config_path: str | Path,
        client_config: DAMZoneClientConfig | None = None,
    ) -> "DAMPromptZoneAgent":
        """Load prompt-zone options from a camera zone config."""
        with Path(config_path).open("r", encoding="utf-8") as file:
            config = yaml.safe_load(file)

        default_config = config["default_zone"]
        default_zone = PromptZone(
            zone_id=default_config["zone_id"],
            zone_type=default_config["zone_type"],
            description=default_config.get(
                "semantic_description",
                "ordinary non-restricted construction-site access area",
            ),
        )
        zones = [
            PromptZone(
                zone_id=zone["zone_id"],
                zone_type=zone["zone_type"],
                description=zone.get("semantic_description", zone["zone_type"]),
            )
            for zone in config.get("zones", [])
        ]
        return cls(zones=zones, default_zone=default_zone, client_config=client_config)

    def recognize(self, image_path: str | Path, extra_prompt: str = "") -> ZoneRecognition:
        """Return the background zone recognized from one image."""
        prompt = self.build_prompt(extra_prompt)
        raw_text = self._query_dam(Path(image_path), prompt)
        parsed = self.parse_response(raw_text)
        return ZoneRecognition(
            zone_id=parsed["zone_id"],
            zone_type=parsed["zone_type"],
            confidence=parsed["confidence"],
            source="dam_prompt",
            reason=parsed["reason"],
            raw_response=raw_text,
        )

    def build_prompt(self, extra_prompt: str = "") -> str:
        """Build a constrained zone-recognition prompt from configured zones."""
        options = [self.default_zone, *self.zones]
        option_lines = "\n".join(
            f"- {zone.zone_id}: {zone.zone_type} = {zone.description}" for zone in options
        )
        zone_types = ", ".join(zone.zone_type for zone in options)
        prompt = f"""
You are identifying the background safety zone in one construction-site image.
Use only scene/background evidence such as barriers, warning signs, excavation,
heavy machinery proximity, materials, active work, and general access context.

Configured zone options:
{option_lines}

Return exactly one compact JSON object with keys zone_id, zone_type, confidence, reason.
Allowed zone_id values: {", ".join(zone.zone_id for zone in options)}, unknown.
Allowed zone_type values: {zone_types}, unknown.
Choose unknown only when there is not enough visual evidence.
Do not describe PPE and do not classify individual workers.
""".strip()
        if extra_prompt:
            prompt = f"{prompt}\n\nAdditional user prompt: {extra_prompt.strip()}"
        return prompt

    def parse_response(self, text: str) -> dict:
        """Parse DAM text into configured zone fields."""
        parsed = {
            "zone_id": "unknown",
            "zone_type": "unknown",
            "confidence": None,
            "reason": text,
        }
        json_match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        fallback_text = text
        if json_match:
            try:
                candidate = json.loads(json_match.group(0))
                parsed_candidate = self._parse_candidate(candidate)
                if parsed_candidate:
                    return parsed_candidate
                fallback_text = str(candidate.get("reason", ""))
            except json.JSONDecodeError:
                pass

        regex_candidate = self._parse_truncated_json(text)
        if regex_candidate:
            return regex_candidate

        inferred = self._infer_from_text(fallback_text)
        if inferred:
            return inferred
        return parsed

    def _query_dam(self, image_path: Path, prompt: str) -> str:
        """Call the DAM OpenAI-compatible API."""
        config = self.client_config
        payload = {
            "model": config.model,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": encode_full_frame_region(image_path)}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
            "max_tokens": config.max_tokens,
            "temperature": config.temperature,
            "top_p": config.top_p,
            "use_cache": True,
            "num_beams": 1,
        }
        response = requests.post(
            f"{config.server_url.rstrip('/')}/chat/completions",
            json=payload,
            timeout=config.timeout,
        )
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]

    def _parse_candidate(self, candidate: dict) -> dict | None:
        zone_id = normalize_zone_id(str(candidate.get("zone_id", "")))
        zone_type = normalize_zone_type(str(candidate.get("zone_type", "")))
        resolved = self._resolve_zone(zone_id=zone_id, zone_type=zone_type)
        if not resolved:
            return None
        return {
            "zone_id": resolved.zone_id,
            "zone_type": resolved.zone_type,
            "confidence": _to_float_or_none(candidate.get("confidence")),
            "reason": str(candidate.get("reason", "")),
        }

    def _parse_truncated_json(self, text: str) -> dict | None:
        zone_id = ""
        zone_type = ""
        zone_id_match = re.search(r'"?zone_id"?\s*:\s*"?([a-zA-Z0-9_\-]+)"?', text)
        zone_type_match = re.search(r'"?zone_type"?\s*:\s*"?([a-zA-Z_\- ]+)"?', text)
        if zone_id_match:
            zone_id = normalize_zone_id(zone_id_match.group(1))
        if zone_type_match:
            zone_type = normalize_zone_type(zone_type_match.group(1))
        resolved = self._resolve_zone(zone_id=zone_id, zone_type=zone_type)
        if not resolved:
            return None
        confidence = None
        confidence_match = re.search(r'"?confidence"?\s*:\s*([0-9.]+)', text)
        if confidence_match:
            confidence = _to_float_or_none(confidence_match.group(1))
        return {
            "zone_id": resolved.zone_id,
            "zone_type": resolved.zone_type,
            "confidence": confidence,
            "reason": text,
        }

    def _infer_from_text(self, text: str) -> dict | None:
        lowered = text.lower()
        matches = []
        for zone in [self.default_zone, *self.zones]:
            if zone.zone_type in lowered or zone.zone_id.lower() in lowered:
                matches.append(zone)
        for zone in self.zones:
            keywords = _keywords(zone.zone_type, zone.description)
            if any(keyword in lowered for keyword in keywords):
                matches.append(zone)
        if any(keyword in lowered for keyword in _keywords(self.default_zone.zone_type, self.default_zone.description)):
            matches.append(self.default_zone)

        unique = {(zone.zone_id, zone.zone_type): zone for zone in matches}
        if len(unique) != 1:
            return None
        zone = next(iter(unique.values()))
        return {
            "zone_id": zone.zone_id,
            "zone_type": zone.zone_type,
            "confidence": None,
            "reason": text,
        }

    def _resolve_zone(self, zone_id: str, zone_type: str) -> PromptZone | None:
        options = [self.default_zone, *self.zones]
        if zone_id == "unknown" or zone_type == "unknown":
            return PromptZone("unknown", "unknown", "unknown")
        for zone in options:
            if zone_id and zone.zone_id == zone_id:
                if not zone_type or zone_type == zone.zone_type:
                    return zone
            if zone_type and zone.zone_type == zone_type:
                if not zone_id or zone_id == zone.zone_id:
                    return zone
        return None


def encode_full_frame_region(image_path: Path) -> str:
    """Return an RGBA data URL with the full frame selected by alpha."""
    bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"Could not read image: {image_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    alpha = np.full(rgb.shape[:2], 255, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGRA))
    if not ok:
        raise ValueError(f"Could not encode image: {image_path}")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def normalize_zone_type(value: str) -> str:
    """Normalize a DAM zone label into the configured zone vocabulary."""
    normalized = value.strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "active": "active_work_area",
        "work_area": "active_work_area",
        "work_zone": "active_work_area",
        "construction_zone": "active_work_area",
        "construction_area": "active_work_area",
        "restricted": "restricted_area",
        "hazard_zone": "restricted_area",
        "hazardous_area": "restricted_area",
        "danger_zone": "restricted_area",
        "general": "general_area",
        "safe_zone": "general_area",
        "public_area": "general_area",
    }
    return aliases.get(normalized, normalized)


def normalize_zone_id(value: str) -> str:
    """Normalize a DAM zone ID."""
    normalized = value.strip().upper().replace("-", "_").replace(" ", "_")
    if normalized in {"Z00", "Z01", "Z02", "UNKNOWN"}:
        return normalized.lower() if normalized == "UNKNOWN" else normalized
    return normalized


def _to_float_or_none(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _keywords(zone_type: str, description: str) -> list[str]:
    """Return parser keywords for a configured zone type."""
    del description
    keyword_map = {
        "active_work_area": [
            "active work",
            "work zone",
            "construction work",
            "construction area",
            "active workers",
        ],
        "restricted_area": [
            "restricted",
            "hazard",
            "danger",
            "barrier",
            "excavation",
            "warning",
        ],
        "general_area": [
            "general",
            "ordinary",
            "public",
            "access area",
        ],
    }
    return keyword_map.get(zone_type, [])
