from __future__ import annotations

import hashlib
import io
from datetime import datetime, timezone
from types import MappingProxyType

import numpy as np
from PIL import Image, ImageOps, ImageStat, UnidentifiedImageError

from construction_safety_moa.contracts import EvidenceIssue
from construction_safety_moa.preprocessing.models import (
    ImageTransform,
    PreparedFrame,
    PreprocessingConfig,
    RawCameraFrame,
    RGBArray,
)

SUPPORTED_IMAGE_FORMATS = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "WEBP": "image/webp",
}


class FramePreprocessingError(ValueError):
    """Fail-closed error with a stable machine-readable reason code."""

    def __init__(self, reason_code: str, detail: str = "") -> None:
        self.reason_code = reason_code
        self.detail = detail
        message = reason_code if not detail else f"{reason_code}:{detail}"
        super().__init__(message)


class FramePreprocessor:
    """Validate and normalize one camera frame without transport or model inference."""

    def __init__(self, config: PreprocessingConfig | None = None) -> None:
        self.config = config or PreprocessingConfig()

    def prepare(self, raw_frame: RawCameraFrame) -> PreparedFrame:
        timestamp = self._validated_metadata(raw_frame)
        source_bytes = raw_frame.image_bytes
        if not isinstance(source_bytes, bytes) or not source_bytes:
            raise FramePreprocessingError("EMPTY_IMAGE_BYTES")
        if len(source_bytes) > self.config.max_source_bytes:
            raise FramePreprocessingError("IMAGE_TOO_LARGE")

        image, mime_type = self._decode(source_bytes)
        original_width, original_height = image.size
        mean_luminance = float(ImageStat.Stat(image.convert("L")).mean[0])
        quality_flags = self._quality_flags(
            original_width,
            original_height,
            mean_luminance,
        )
        model_image, transform = self._letterbox(image)

        original_rgb = self._read_only_array(image)
        model_rgb = self._read_only_array(model_image)
        metadata = MappingProxyType(
            {
                "source_mime_type": mime_type,
                "pixel_format": "RGB",
                "original_size": (original_width, original_height),
                "model_size": self.config.target_size,
                "resize_method": "letterbox_bilinear",
                "padding_value": self.config.padding_value,
                "mean_luminance": mean_luminance,
            }
        )

        return PreparedFrame(
            frame_id=raw_frame.frame_id.strip(),
            camera_id=raw_frame.camera_id.strip(),
            timestamp=timestamp,
            source_ref=raw_frame.source_ref.strip(),
            source_sha256=hashlib.sha256(source_bytes).hexdigest(),
            original_rgb=original_rgb,
            model_rgb=model_rgb,
            transform=transform,
            quality_flags=quality_flags,
            metadata=metadata,
        )

    def _validated_metadata(self, raw_frame: RawCameraFrame) -> str:
        if not isinstance(raw_frame, RawCameraFrame):
            raise FramePreprocessingError("INVALID_FRAME_METADATA")
        values = (raw_frame.frame_id, raw_frame.camera_id, raw_frame.source_ref)
        if any(not isinstance(value, str) or not value.strip() for value in values):
            raise FramePreprocessingError("INVALID_FRAME_METADATA")
        if not isinstance(raw_frame.timestamp, str) or not raw_frame.timestamp.strip():
            raise FramePreprocessingError("INVALID_FRAME_METADATA")
        timestamp_text = raw_frame.timestamp.strip()
        if timestamp_text.endswith("Z"):
            timestamp_text = f"{timestamp_text[:-1]}+00:00"
        try:
            parsed = datetime.fromisoformat(timestamp_text)
            offset = parsed.utcoffset()
        except (TypeError, ValueError):
            raise FramePreprocessingError("INVALID_FRAME_METADATA") from None
        if parsed.tzinfo is None or offset is None:
            raise FramePreprocessingError("INVALID_FRAME_METADATA")
        return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")

    def _decode(self, source_bytes: bytes) -> tuple[Image.Image, str]:
        try:
            with Image.open(io.BytesIO(source_bytes)) as verification_image:
                image_format = str(verification_image.format or "").upper()
                if image_format not in SUPPORTED_IMAGE_FORMATS:
                    raise FramePreprocessingError("UNSUPPORTED_IMAGE_MIME")
                width, height = verification_image.size
                self._validate_dimensions(width, height)
                verification_image.verify()

            with Image.open(io.BytesIO(source_bytes)) as opened_image:
                image = ImageOps.exif_transpose(opened_image)
                image.load()
                image = image.convert("RGB")
        except FramePreprocessingError:
            raise
        except (Image.DecompressionBombError, OSError, UnidentifiedImageError, ValueError):
            raise FramePreprocessingError("INVALID_IMAGE_CONTENT") from None

        self._validate_dimensions(*image.size)
        return image, SUPPORTED_IMAGE_FORMATS[image_format]

    def _validate_dimensions(self, width: int, height: int) -> None:
        if width <= 0 or height <= 0:
            raise FramePreprocessingError("INVALID_IMAGE_DIMENSIONS")
        if width * height > self.config.max_source_pixels:
            raise FramePreprocessingError("IMAGE_PIXEL_LIMIT_EXCEEDED")

    def _quality_flags(
        self,
        width: int,
        height: int,
        mean_luminance: float,
    ) -> tuple[EvidenceIssue, ...]:
        flags: list[EvidenceIssue] = []
        if min(width, height) < self.config.min_short_side:
            flags.append(EvidenceIssue.LOW_RESOLUTION)
        if mean_luminance < self.config.low_light_threshold:
            flags.append(EvidenceIssue.LOW_LIGHT)
        return tuple(flags)

    def _letterbox(self, image: Image.Image) -> tuple[Image.Image, ImageTransform]:
        original_width, original_height = image.size
        target_width, target_height = self.config.target_size
        scale = min(target_width / original_width, target_height / original_height)
        resized_width = min(target_width, max(1, round(original_width * scale)))
        resized_height = min(target_height, max(1, round(original_height * scale)))
        resized = image.resize((resized_width, resized_height), resample=Image.Resampling.BILINEAR)

        remaining_width = target_width - resized_width
        remaining_height = target_height - resized_height
        left = remaining_width // 2
        top = remaining_height // 2
        right = remaining_width - left
        bottom = remaining_height - top
        fill = (self.config.padding_value,) * 3
        model_image = Image.new("RGB", self.config.target_size, color=fill)
        model_image.paste(resized, (left, top))

        transform = ImageTransform(
            original_size=(original_width, original_height),
            model_size=self.config.target_size,
            scale=scale,
            padding=(left, top, right, bottom),
        )
        return model_image, transform

    def _read_only_array(self, image: Image.Image) -> RGBArray:
        array = np.array(image, dtype=np.uint8, copy=True, order="C")
        array.setflags(write=False)
        return array
