from __future__ import annotations

import hashlib
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

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
except ImportError:  # pragma: no cover - exercised by the optional-dependency test
    Image = None


def build_request() -> ContextRequest:
    return ContextRequest(
        request_id="CTXREQ-MEDIA-001",
        event_id="EVT-MEDIA-001",
        worker_id="W01",
        frame_ref="FRAME-001",
        crop_ref="FRAME-001::W01::40,40,120,180",
        higher_resolution_source_ref="FRAME-001-HIRES",
        neighbor_frame_refs=["FRAME-000", "FRAME-002"],
        detections=[
            Detection("W01", "person", [40, 40, 120, 180], 0.95),
            Detection("NH01", "no_helmet", [55, 42, 95, 82], 0.92),
            Detection("EQ01", "excavator", [145, 35, 285, 205], 0.91),
        ],
        ppe_finding=PPEStatus("missing", 0.92, "W01", ["NH01"]),
        zone_grounding=RegionGrounding(
            "ZONE-01",
            "active_work_area",
            "inside",
            0.99,
            "W01",
        ),
        evidence_issues=[EvidenceIssue.RELATION_UNCLEAR],
        allowed_context_actions=[
            ContextAction.EMIT_CONTEXT_EVIDENCE,
            ContextAction.ABSTAIN,
        ],
    )


@unittest.skipIf(Image is None, "Pillow is an optional Context pilot dependency")
class TestMediaResolver(unittest.TestCase):
    def _write_image(self, path: Path, size: tuple[int, int] = (320, 240)) -> str:
        image = Image.new("RGB", size, color=(228, 232, 235))
        image.save(path, format="PNG", compress_level=9)
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def _entry(self, path: Path, checksum: str) -> MediaManifestEntry:
        return MediaManifestEntry(
            logical_ref="FRAME-001",
            relative_path=path.name,
            sha256=checksum,
            mime_type="image/png",
            zone_polygons={
                "ZONE-01": [[20, 20], [300, 20], [300, 220], [20, 220]],
            },
        )

    def test_path_traversal_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root) / "media"
            media_root.mkdir()
            outside = Path(raw_root) / "outside.png"
            checksum = self._write_image(outside)
            resolver = MediaResolver(
                media_root,
                {
                    "FRAME-001": MediaManifestEntry(
                        logical_ref="FRAME-001",
                        relative_path="../outside.png",
                        sha256=checksum,
                        mime_type="image/png",
                        zone_polygons={"ZONE-01": [[0, 0], [1, 0], [1, 1]]},
                    )
                },
            )

            resolved = resolver.resolve(build_request())

            self.assertFalse(resolved.ready)
            self.assertIn("MEDIA_PATH_OUTSIDE_ROOT:FRAME-001", resolved.validation_errors)
            self.assertEqual(resolved.artifacts, [])

    def test_missing_file_and_checksum_mismatch_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            missing_resolver = MediaResolver(
                media_root,
                {
                    "FRAME-001": MediaManifestEntry(
                        logical_ref="FRAME-001",
                        relative_path="missing.png",
                        sha256="0" * 64,
                        mime_type="image/png",
                        zone_polygons={"ZONE-01": [[0, 0], [1, 0], [1, 1]]},
                    )
                },
            )
            self.assertIn(
                "MEDIA_FILE_MISSING:FRAME-001",
                missing_resolver.resolve(build_request()).validation_errors,
            )

            image_path = media_root / "frame.png"
            self._write_image(image_path)
            mismatch = self._entry(image_path, "f" * 64)
            mismatch_resolver = MediaResolver(media_root, {"FRAME-001": mismatch})

            resolved = mismatch_resolver.resolve(build_request())

            self.assertFalse(resolved.ready)
            self.assertIn("SOURCE_CHECKSUM_MISMATCH:FRAME-001", resolved.validation_errors)

    def test_mime_mismatch_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            wrong_mime_entry = replace(
                self._entry(image_path, checksum),
                mime_type="image/jpeg",
            )

            resolved = MediaResolver(
                media_root,
                {"FRAME-001": wrong_mime_entry},
            ).resolve(build_request())

            self.assertFalse(resolved.ready)
            self.assertIn("SOURCE_MIME_MISMATCH:FRAME-001", resolved.validation_errors)

    def test_crop_and_overlay_are_deterministic_and_traceable(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            resolver = MediaResolver(
                media_root,
                {"FRAME-001": self._entry(image_path, checksum)},
                crop_padding_ratio=0.1,
                max_image_side=640,
            )

            first = resolver.resolve(build_request())
            second = resolver.resolve(build_request())

            self.assertTrue(first.ready, first.validation_errors)
            self.assertEqual(first.presented_frame_refs, ["FRAME-001"])
            self.assertEqual(
                first.presented_crop_refs,
                ["FRAME-001::W01::40,40,120,180"],
            )
            self.assertEqual(
                [artifact.preprocessed_sha256 for artifact in first.artifacts],
                [artifact.preprocessed_sha256 for artifact in second.artifacts],
            )
            self.assertTrue(all(artifact.source_sha256 == checksum for artifact in first.artifacts))
            self.assertEqual(first.artifacts[0].width, 320)
            self.assertEqual(first.artifacts[0].height, 240)
            self.assertEqual(first.artifacts[1].crop_coordinates, [32, 26, 128, 194])
            self.assertEqual(first.artifacts[1].padding_parameters["ratio"], 0.1)
            self.assertEqual(
                first.artifacts[0].overlay_parameters["detection_ids"],
                ["W01", "NH01", "EQ01"],
            )
            self.assertEqual(first.artifacts[0].overlay_parameters["zone_id"], "ZONE-01")

    def test_unmanifested_media_is_never_presented(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            self._write_image(media_root / "unmanifested-neighbor.png")
            resolver = MediaResolver(
                media_root,
                {"FRAME-001": self._entry(image_path, checksum)},
            )

            resolved = resolver.resolve(build_request())

            self.assertTrue(resolved.ready, resolved.validation_errors)
            presented = resolved.presented_frame_refs + resolved.presented_crop_refs
            self.assertNotIn("FRAME-000", presented)
            self.assertNotIn("FRAME-002", presented)
            self.assertNotIn("FRAME-001-HIRES", presented)
            self.assertEqual(resolved.available_but_not_acquired_refs, [])

            missing_manifest = MediaResolver(media_root, {}).resolve(build_request())
            self.assertFalse(missing_manifest.ready)
            self.assertIn("MEDIA_REF_NOT_IN_MANIFEST:FRAME-001", missing_manifest.validation_errors)

    def test_manifested_neighbor_is_available_but_not_acquired(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            neighbor_path = media_root / "neighbor.png"
            neighbor_checksum = self._write_image(neighbor_path)
            resolver = MediaResolver(
                media_root,
                {
                    "FRAME-001": self._entry(image_path, checksum),
                    "FRAME-000": MediaManifestEntry(
                        logical_ref="FRAME-000",
                        relative_path=neighbor_path.name,
                        sha256=neighbor_checksum,
                        mime_type="image/png",
                    ),
                },
            )

            resolved = resolver.resolve(build_request())

            self.assertEqual(resolved.available_but_not_acquired_refs, ["FRAME-000"])
            self.assertEqual(resolved.presented_frame_refs, ["FRAME-001"])

    def test_zone_boundary_is_required_and_request_is_not_mutated(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            request = build_request()
            original = request.to_dict()
            entry = replace(self._entry(image_path, checksum), zone_polygons={})

            resolved = MediaResolver(media_root, {"FRAME-001": entry}).resolve(request)

            self.assertFalse(resolved.ready)
            self.assertIn("ZONE_BOUNDARY_NOT_IN_MANIFEST:ZONE-01", resolved.validation_errors)
            self.assertEqual(request.to_dict(), original)

    def test_crop_ref_must_match_existing_worker_bbox(self) -> None:
        with tempfile.TemporaryDirectory() as raw_root:
            media_root = Path(raw_root)
            image_path = media_root / "frame.png"
            checksum = self._write_image(image_path)
            resolver = MediaResolver(
                media_root,
                {"FRAME-001": self._entry(image_path, checksum)},
            )

            resolved = resolver.resolve(
                replace(build_request(), crop_ref="FRAME-001::W01::0,0,1,1")
            )

            self.assertFalse(resolved.ready)
            self.assertIn(
                "CROP_REF_BBOX_MISMATCH:FRAME-001::W01::0,0,1,1",
                resolved.validation_errors,
            )


if __name__ == "__main__":
    unittest.main()
