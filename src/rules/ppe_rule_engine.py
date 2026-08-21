"""Minimal PPE rule engine placeholder for future evidence joining."""

from __future__ import annotations

from pathlib import Path

import yaml


def load_rules(config_path: str | Path) -> dict:
    """Load PPE rule configuration."""
    with Path(config_path).open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def required_ppe_for_zone(rules: dict, zone_type: str) -> list[str]:
    """Return configured PPE requirements for a zone type."""
    zone_rules = rules.get("zones", {}).get(zone_type, {})
    return list(zone_rules.get("required_ppe", []))
