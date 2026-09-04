from __future__ import annotations

import hashlib
import io
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from construction_safety_moa.contracts import ContextRequest, Detection

try:
    from PIL import Image, ImageDraw, ImageOps, UnidentifiedImageError
except ImportError:  # Pillow is deliberately optional for the core prototype.
    Image = None
    ImageDraw = None
    ImageOps = None

    class UnidentifiedImageError(Exception):
        pass


SUPPORTED_IMAGE_MIME_TYPES = frozenset(
    {
        "image/jpeg",
        "image/png",
        "image/webp",
    }
)


@dataclass(frozen=True)
class MediaManifestEntry:
    """Trusted mapping from one logical ref to a file under the configured media root."""

    logical_ref: str
    relative_path: str
    sha256: str
    mime_type: str
    zone_polygons: dict[str, list[list[float]]] = field(default_factory=dict)


@dataclass
class ResolvedMediaArtifact:
    logical_ref: str
    role: str
    data: bytes = field(repr=False)
    mime_type: str = "image/png"
    width: int = 0
    height: int = 0
    source_width: int = 0
    source_height: int = 0
    source_sha256: str = ""
    preprocessed_sha256: str = ""
    crop_coordinates: list[int] | None = None
    resize_parameters: dict[str, Any] = field(default_factory=dict)
    padding_parameters: dict[str, Any] = field(default_factory=dict)
    overlay_parameters: dict[str, Any] = field(default_factory=dict)

    def metadata(self) -> dict[str, Any]:
        """Return trace metadata without embedding image bytes."""

        return {
            "logical_ref": self.logical_ref,
            "role": self.role,
            "mime_type": self.mime_type,
            "width": self.width,
            "height": self.height,
            "source_width": self.source_width,
            "source_height": self.source_height,
            "source_sha256": self.source_sha256,
            "preprocessed_sha256": self.preprocessed_sha256,
            "crop_coordinates": self.crop_coordinates,
            "resize_parameters": self.resize_parameters,
            "padding_parameters": self.padding_parameters,
            "overlay_parameters": self.overlay_parameters,
        }


@dataclass
class ResolvedContextMedia:
    presented_frame_refs: list[str] = field(default_factory=list)
    presented_crop_refs: list[str] = field(default_factory=list)
    available_but_not_acquired_refs: list[str] = field(default_factory=list)
    artifacts: list[ResolvedMediaArtifact] = field(default_factory=list, repr=False)
    resolved_at: str = ""
    validation_errors: list[str] = field(default_factory=list)

    @property
    def ready(self) -> bool:
        return not self.validation_errors and len(self.artifacts) == 2

    def metadata(self) -> dict[str, Any]:
        return {
            "presented_frame_refs": list(self.presented_frame_refs),
            "presented_crop_refs": list(self.presented_crop_refs),
            "available_but_not_acquired_refs": list(self.available_but_not_acquired_refs),
            "resolved_at": self.resolved_at,
            "validation_errors": list(self.validation_errors),
            "artifacts": [artifact.metadata() for artifact in self.artifacts],
        }


class MediaResolver:
    """Resolve manifest-authorized image refs and build deterministic VLM inputs."""

    OVERLAY_STYLE = {
        "worker_color": "#dc2626",
        "object_color": "#d97706",
        "zone_color": "#0f766e",
        "line_width": 3,
        "label_background": "#111827",
        "label_foreground": "#ffffff",
    }

    def __init__(
        self,
        media_root: str | Path,
        manifest: dict[str, MediaManifestEntry],
        *,
        crop_padding_ratio: float = 0.15,
        max_image_side: int = 1600,
        max_source_bytes: int = 25 * 1024 * 1024,
        max_source_pixels: int = 40_000_000,
    ) -> None:
        self.media_root = Path(media_root).resolve()
        self.manifest = dict(manifest)
        if not 0.0 <= crop_padding_ratio <= 1.0:
            raise ValueError("crop_padding_ratio must be between 0 and 1")
        if max_image_side < 64:
            raise ValueError("max_image_side must be at least 64")
        self.crop_padding_ratio = float(crop_padding_ratio)
        self.max_image_side = int(max_image_side)
        self.max_source_bytes = int(max_source_bytes)
        self.max_source_pixels = int(max_source_pixels)

    def resolve(self, request: ContextRequest) -> ResolvedContextMedia:
        resolved = ResolvedContextMedia(
            resolved_at=datetime.now(timezone.utc).isoformat(),
            available_but_not_acquired_refs=self._available_refs(request),
        )
        if Image is None or ImageDraw is None or ImageOps is None:
            resolved.validation_errors.append("OPTIONAL_DEPENDENCY_MISSING:Pillow")
            return resolved

        entry = self.manifest.get(request.frame_ref)
        if entry is None:
            resolved.validation_errors.append(f"MEDIA_REF_NOT_IN_MANIFEST:{request.frame_ref}")
            return resolved
        if entry.logical_ref != request.frame_ref:
            resolved.validation_errors.append(f"MANIFEST_LOGICAL_REF_MISMATCH:{entry.logical_ref}")
            return resolved

        source_path, path_error = self._authorized_path(entry, request.frame_ref)
        if path_error:
            resolved.validation_errors.append(path_error)
            return resolved
        if source_path is None or not source_path.is_file():
            resolved.validation_errors.append(f"MEDIA_FILE_MISSING:{request.frame_ref}")
            return resolved

        try:
            source_size = source_path.stat().st_size
        except OSError:
            resolved.validation_errors.append(f"MEDIA_FILE_UNREADABLE:{request.frame_ref}")
            return resolved
        if source_size > self.max_source_bytes:
            resolved.validation_errors.append(f"MEDIA_FILE_TOO_LARGE:{request.frame_ref}")
            return resolved

        try:
            source_bytes = source_path.read_bytes()
        except OSError:
            resolved.validation_errors.append(f"MEDIA_FILE_UNREADABLE:{request.frame_ref}")
            return resolved

        source_sha256 = hashlib.sha256(source_bytes).hexdigest()
        if not self._valid_sha256(entry.sha256):
            resolved.validation_errors.append(f"INVALID_MANIFEST_CHECKSUM:{request.frame_ref}")
            return resolved
        if source_sha256 != entry.sha256.lower():
            resolved.validation_errors.append(f"SOURCE_CHECKSUM_MISMATCH:{request.frame_ref}")
            return resolved

        detected_mime = self._detect_mime(source_bytes)
        if entry.mime_type not in SUPPORTED_IMAGE_MIME_TYPES:
            resolved.validation_errors.append(f"UNSUPPORTED_MANIFEST_MIME:{entry.mime_type}")
            return resolved
        if detected_mime != entry.mime_type:
            resolved.validation_errors.append(f"SOURCE_MIME_MISMATCH:{request.frame_ref}")
            return resolved

        try:
            with Image.open(io.BytesIO(source_bytes)) as verification_image:
                verification_image.verify()
            with Image.open(io.BytesIO(source_bytes)) as opened:
                encoded_width, encoded_height = opened.size
                if encoded_width < 1 or encoded_height < 1:
                    resolved.validation_errors.append(
                        f"INVALID_IMAGE_DIMENSIONS:{request.frame_ref}"
                    )
                    return resolved
                if encoded_width * encoded_height > self.max_source_pixels:
                    resolved.validation_errors.append(
                        f"IMAGE_PIXEL_LIMIT_EXCEEDED:{request.frame_ref}"
                    )
                    return resolved
                image = ImageOps.exif_transpose(opened).convert("RGB")
        except (OSError, UnidentifiedImageError, ValueError):
            resolved.validation_errors.append(f"INVALID_IMAGE_CONTENT:{request.frame_ref}")
            return resolved

        source_width, source_height = image.size
        if source_width * source_height > self.max_source_pixels:
            resolved.validation_errors.append(f"IMAGE_PIXEL_LIMIT_EXCEEDED:{request.frame_ref}")
            return resolved

        zone_polygon = entry.zone_polygons.get(request.zone_grounding.zone_id)
        if zone_polygon is None:
            resolved.validation_errors.append(
                f"ZONE_BOUNDARY_NOT_IN_MANIFEST:{request.zone_grounding.zone_id}"
            )
            return resolved
        normalized_zone = self._validated_polygon(
            zone_polygon,
            source_width,
            source_height,
        )
        if normalized_zone is None:
            resolved.validation_errors.append(
                f"INVALID_ZONE_BOUNDARY:{request.zone_grounding.zone_id}"
            )
            return resolved

        normalized_detections: list[tuple[Detection, list[float]]] = []
        for detection in request.detections:
            bbox = self._validated_bbox(detection, source_width, source_height)
            if bbox is None:
                resolved.validation_errors.append(f"INVALID_DETECTION_BBOX:{detection.object_id}")
            else:
                normalized_detections.append((detection, bbox))
        if resolved.validation_errors:
            return resolved

        target = next(
            (item for item in normalized_detections if item[0].object_id == request.worker_id),
            None,
        )
        if target is None:
            resolved.validation_errors.append(f"WORKER_BBOX_NOT_FOUND:{request.worker_id}")
            return resolved
        expected_crop_ref = self._expected_crop_ref(
            request.frame_ref,
            request.worker_id,
            target[1],
        )
        if request.crop_ref != expected_crop_ref:
            resolved.validation_errors.append(f"CROP_REF_BBOX_MISMATCH:{request.crop_ref}")
            return resolved

        overlay_image, overlay_resize, overlay_parameters = self._overlay(
            image,
            normalized_detections,
            normalized_zone,
            request,
        )
        overlay_bytes = self._png_bytes(overlay_image)
        crop_coordinates = self._crop_coordinates(
            target[1],
            source_width,
            source_height,
        )
        crop_image = image.crop(tuple(crop_coordinates))
        crop_image, crop_resize = self._resize(crop_image)
        crop_bytes = self._png_bytes(crop_image)

        resolved.presented_frame_refs = [request.frame_ref]
        resolved.presented_crop_refs = [request.crop_ref]
        resolved.artifacts = [
            ResolvedMediaArtifact(
                logical_ref=request.frame_ref,
                role="annotated_frame",
                data=overlay_bytes,
                mime_type="image/png",
                width=overlay_image.width,
                height=overlay_image.height,
                source_width=source_width,
                source_height=source_height,
                source_sha256=source_sha256,
                preprocessed_sha256=hashlib.sha256(overlay_bytes).hexdigest(),
                resize_parameters=overlay_resize,
                overlay_parameters=overlay_parameters,
            ),
            ResolvedMediaArtifact(
                logical_ref=request.crop_ref,
                role="worker_crop",
                data=crop_bytes,
                mime_type="image/png",
                width=crop_image.width,
                height=crop_image.height,
                source_width=source_width,
                source_height=source_height,
                source_sha256=source_sha256,
                preprocessed_sha256=hashlib.sha256(crop_bytes).hexdigest(),
                crop_coordinates=crop_coordinates,
                resize_parameters=crop_resize,
                padding_parameters={
                    "ratio": self.crop_padding_ratio,
                    "coordinate_space": "source_pixels",
                },
                overlay_parameters={"applied": False},
            ),
        ]
        return resolved

    def _available_refs(self, request: ContextRequest) -> list[str]:
        possible = [
            request.higher_resolution_source_ref,
            *request.neighbor_frame_refs,
        ]
        return list(
            dict.fromkeys(
                ref
                for ref in possible
                if isinstance(ref, str) and ref != request.frame_ref and ref in self.manifest
            )
        )

    def _authorized_path(
        self,
        entry: MediaManifestEntry,
        logical_ref: str,
    ) -> tuple[Path | None, str | None]:
        raw_path = Path(entry.relative_path)
        if raw_path.is_absolute():
            return None, f"MEDIA_PATH_OUTSIDE_ROOT:{logical_ref}"
        try:
            resolved_path = (self.media_root / raw_path).resolve()
            resolved_path.relative_to(self.media_root)
        except (OSError, ValueError):
            return None, f"MEDIA_PATH_OUTSIDE_ROOT:{logical_ref}"
        return resolved_path, None

    def _valid_sha256(self, value: object) -> bool:
        if not isinstance(value, str) or len(value) != 64:
            return False
        return all(character in "0123456789abcdefABCDEF" for character in value)

    def _detect_mime(self, data: bytes) -> str | None:
        if data.startswith(b"\x89PNG\r\n\x1a\n"):
            return "image/png"
        if data.startswith(b"\xff\xd8\xff"):
            return "image/jpeg"
        if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
            return "image/webp"
        return None

    def _validated_bbox(
        self,
        detection: Detection,
        width: int,
        height: int,
    ) -> list[float] | None:
        if len(detection.bbox) != 4:
            return None
        try:
            x1, y1, x2, y2 = (float(value) for value in detection.bbox)
        except (TypeError, ValueError):
            return None
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            return None
        return [x1, y1, x2, y2]

    def _validated_polygon(
        self,
        polygon: object,
        width: int,
        height: int,
    ) -> list[list[float]] | None:
        if not isinstance(polygon, list) or len(polygon) < 3:
            return None
        points: list[list[float]] = []
        for point in polygon:
            if not isinstance(point, list) or len(point) != 2:
                return None
            try:
                x, y = float(point[0]), float(point[1])
            except (TypeError, ValueError):
                return None
            if not (0 <= x <= width and 0 <= y <= height):
                return None
            points.append([x, y])
        return points

    def _resize(self, image: Any) -> tuple[Any, dict[str, Any]]:
        original_size = [image.width, image.height]
        scale = min(1.0, self.max_image_side / max(image.width, image.height))
        if scale < 1.0:
            output_size = [
                max(1, round(image.width * scale)),
                max(1, round(image.height * scale)),
            ]
            image = image.resize(tuple(output_size), resample=Image.Resampling.LANCZOS)
        else:
            output_size = original_size
        return image, {
            "max_image_side": self.max_image_side,
            "original_size": original_size,
            "output_size": output_size,
            "scale": round(scale, 8),
            "resample": "LANCZOS" if scale < 1.0 else "NONE",
        }

    def _overlay(
        self,
        source_image: Any,
        detections: list[tuple[Detection, list[float]]],
        zone_polygon: list[list[float]],
        request: ContextRequest,
    ) -> tuple[Any, dict[str, Any], dict[str, Any]]:
        overlay_image, resize_parameters = self._resize(source_image.copy())
        scale = float(resize_parameters["scale"])
        draw = ImageDraw.Draw(overlay_image)
        style = self.OVERLAY_STYLE
        line_width = max(1, round(int(style["line_width"]) * scale))

        scaled_zone = [(point[0] * scale, point[1] * scale) for point in zone_polygon]
        draw.line(
            [*scaled_zone, scaled_zone[0]],
            fill=str(style["zone_color"]),
            width=line_width,
        )
        self._draw_label(
            draw,
            scaled_zone[0],
            f"ZONE {request.zone_grounding.zone_id}",
        )

        for detection, bbox in detections:
            scaled_bbox = [coordinate * scale for coordinate in bbox]
            is_worker = detection.object_id == request.worker_id
            color = str(style["worker_color"] if is_worker else style["object_color"])
            draw.rectangle(scaled_bbox, outline=color, width=line_width)
            prefix = "WORKER" if is_worker else "OBJECT"
            self._draw_label(
                draw,
                (scaled_bbox[0], scaled_bbox[1]),
                f"{prefix} {detection.object_id} {detection.class_label}",
            )

        parameters = {
            "applied": True,
            "worker_id": request.worker_id,
            "detection_ids": [item.object_id for item, _ in detections],
            "detection_bboxes_source_pixels": {item.object_id: bbox for item, bbox in detections},
            "zone_id": request.zone_grounding.zone_id,
            "zone_polygon_source_pixels": zone_polygon,
            "style": dict(style),
        }
        return overlay_image, resize_parameters, parameters

    def _draw_label(self, draw: Any, origin: tuple[float, float], text: str) -> None:
        x, y = round(origin[0]), round(origin[1])
        text_box = draw.textbbox((x, y), text)
        background = (
            text_box[0] - 2,
            text_box[1] - 1,
            text_box[2] + 2,
            text_box[3] + 1,
        )
        draw.rectangle(background, fill=str(self.OVERLAY_STYLE["label_background"]))
        draw.text((x, y), text, fill=str(self.OVERLAY_STYLE["label_foreground"]))

    def _crop_coordinates(
        self,
        bbox: list[float],
        width: int,
        height: int,
    ) -> list[int]:
        x1, y1, x2, y2 = bbox
        pad_x = (x2 - x1) * self.crop_padding_ratio
        pad_y = (y2 - y1) * self.crop_padding_ratio
        return [
            max(0, int(x1 - pad_x)),
            max(0, int(y1 - pad_y)),
            min(width, int(x2 + pad_x + 0.999999)),
            min(height, int(y2 + pad_y + 0.999999)),
        ]

    def _expected_crop_ref(
        self,
        frame_ref: str,
        worker_id: str,
        bbox: list[float],
    ) -> str:
        coordinates = ",".join(
            str(int(value)) if float(value).is_integer() else str(float(value)) for value in bbox
        )
        return f"{frame_ref}::{worker_id}::{coordinates}"

    def _png_bytes(self, image: Any) -> bytes:
        buffer = io.BytesIO()
        image.save(
            buffer,
            format="PNG",
            optimize=False,
            compress_level=9,
        )
        return buffer.getvalue()
