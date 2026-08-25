from __future__ import annotations

import hashlib
import io
import unittest
from dataclasses import replace

import numpy as np
from PIL import Image

from construction_safety_moa.contracts import EvidenceIssue
from construction_safety_moa.preprocessing import (
    FramePreprocessingError,
    FramePreprocessor,
    ImageTransform,
    PreparedFrame,
    PreprocessingConfig,
    RawCameraFrame,
)


def image_bytes(
    mode: str = "RGB",
    size: tuple[int, int] = (800, 400),
    color: object = (80, 120, 160),
    *,
    image_format: str = "PNG",
    exif_orientation: int | None = None,
) -> bytes:
    image = Image.new(mode, size, color=color)
    buffer = io.BytesIO()
    save_options: dict[str, object] = {}
    if exif_orientation is not None:
        exif = Image.Exif()
        exif[274] = exif_orientation
        save_options["exif"] = exif
    image.save(buffer, format=image_format, **save_options)
    return buffer.getvalue()


def raw_frame(data: bytes | None = None, **overrides: object) -> RawCameraFrame:
    values: dict[str, object] = {
        "frame_id": "FRAME-001",
        "camera_id": "CAM-01",
        "timestamp": "2026-08-25T08:15:30+07:00",
        "source_ref": "camera/CAM-01/FRAME-001",
        "image_bytes": data if data is not None else image_bytes(),
    }
    values.update(overrides)
    return RawCameraFrame(**values)  # type: ignore[arg-type]


class TestFramePreprocessor(unittest.TestCase):
    def test_prepares_original_and_letterboxed_rgb_images(self) -> None:
        source = image_bytes(size=(800, 400), color=(10, 20, 30))

        prepared = FramePreprocessor().prepare(raw_frame(source))

        self.assertIsInstance(prepared, PreparedFrame)
        self.assertEqual(prepared.frame_id, "FRAME-001")
        self.assertEqual(prepared.camera_id, "CAM-01")
        self.assertEqual(prepared.timestamp, "2026-08-25T01:15:30Z")
        self.assertEqual(prepared.source_ref, "camera/CAM-01/FRAME-001")
        self.assertEqual(prepared.source_sha256, hashlib.sha256(source).hexdigest())
        self.assertEqual(prepared.original_rgb.shape, (400, 800, 3))
        self.assertEqual(prepared.model_rgb.shape, (640, 640, 3))
        self.assertEqual(prepared.original_rgb.dtype, np.uint8)
        self.assertEqual(prepared.model_rgb.dtype, np.uint8)
        self.assertTrue(prepared.original_rgb.flags.c_contiguous)
        self.assertTrue(prepared.model_rgb.flags.c_contiguous)
        self.assertFalse(prepared.original_rgb.flags.writeable)
        self.assertFalse(prepared.model_rgb.flags.writeable)
        np.testing.assert_array_equal(prepared.original_rgb[0, 0], [10, 20, 30])
        np.testing.assert_array_equal(prepared.model_rgb[0, 0], [114, 114, 114])
        np.testing.assert_array_equal(prepared.model_rgb[320, 320], [10, 20, 30])
        self.assertEqual(prepared.transform.original_size, (800, 400))
        self.assertEqual(prepared.transform.model_size, (640, 640))
        self.assertAlmostEqual(prepared.transform.scale, 0.8)
        self.assertEqual(prepared.transform.padding, (0, 160, 0, 160))
        self.assertEqual(prepared.quality_flags, ())
        self.assertEqual(prepared.metadata["source_mime_type"], "image/png")
        self.assertEqual(prepared.metadata["pixel_format"], "RGB")

    def test_converts_grayscale_and_rgba_to_rgb(self) -> None:
        cases = [
            (image_bytes("L", (400, 400), 25), [25, 25, 25]),
            (image_bytes("RGBA", (400, 400), (7, 8, 9, 10)), [7, 8, 9]),
        ]

        for source, expected_pixel in cases:
            with self.subTest(expected_pixel=expected_pixel):
                prepared = FramePreprocessor().prepare(raw_frame(source))
                self.assertEqual(prepared.original_rgb.shape, (400, 400, 3))
                np.testing.assert_array_equal(prepared.original_rgb[0, 0], expected_pixel)

    def test_supports_jpeg_png_and_webp_by_decoded_content(self) -> None:
        expected_mime_types = {
            "JPEG": "image/jpeg",
            "PNG": "image/png",
            "WEBP": "image/webp",
        }

        for image_format, mime_type in expected_mime_types.items():
            with self.subTest(image_format=image_format):
                source = image_bytes(size=(400, 400), image_format=image_format)
                prepared = FramePreprocessor().prepare(raw_frame(source))
                self.assertEqual(prepared.metadata["source_mime_type"], mime_type)

    def test_applies_exif_orientation_before_recording_dimensions(self) -> None:
        source = image_bytes(
            size=(80, 40),
            image_format="JPEG",
            exif_orientation=6,
        )
        config = PreprocessingConfig(min_short_side=1)

        prepared = FramePreprocessor(config).prepare(raw_frame(source))

        self.assertEqual(prepared.original_rgb.shape, (80, 40, 3))
        self.assertEqual(prepared.transform.original_size, (40, 80))

    def test_letterbox_transform_round_trips_bboxes_and_polygons(self) -> None:
        transform = ImageTransform(
            original_size=(800, 400),
            model_size=(640, 640),
            scale=0.8,
            padding=(0, 160, 0, 160),
        )
        source_bbox = [100.0, 50.0, 700.0, 350.0]
        source_polygon = [[0.0, 0.0], [800.0, 0.0], [400.0, 400.0]]

        model_bbox = transform.bbox_to_model(source_bbox)
        model_polygon = transform.polygon_to_model(source_polygon)

        self.assertEqual(model_bbox, [80.0, 200.0, 560.0, 440.0])
        self.assertEqual(
            model_polygon,
            [[0.0, 160.0], [640.0, 160.0], [320.0, 480.0]],
        )
        self.assertEqual(transform.bbox_to_original(model_bbox), source_bbox)
        self.assertEqual(transform.polygon_to_original(model_polygon), source_polygon)
        self.assertEqual(
            transform.bbox_to_original([-10.0, -20.0, 900.0, 900.0]),
            [0.0, 0.0, 800.0, 400.0],
        )

    def test_portrait_letterbox_uses_horizontal_padding(self) -> None:
        config = PreprocessingConfig(target_size=(640, 640), min_short_side=1)
        source = image_bytes(size=(200, 400), color=(20, 30, 40))

        prepared = FramePreprocessor(config).prepare(raw_frame(source))

        self.assertAlmostEqual(prepared.transform.scale, 1.6)
        self.assertEqual(prepared.transform.padding, (160, 0, 160, 0))
        np.testing.assert_array_equal(prepared.model_rgb[320, 0], [114, 114, 114])
        np.testing.assert_array_equal(prepared.model_rgb[320, 320], [20, 30, 40])

    def test_low_quality_frames_are_flagged_without_enhancement_or_rejection(self) -> None:
        config = PreprocessingConfig(min_short_side=360, low_light_threshold=40.0)
        source = image_bytes(size=(320, 200), color=(10, 10, 10))

        prepared = FramePreprocessor(config).prepare(raw_frame(source))

        self.assertEqual(
            prepared.quality_flags,
            (EvidenceIssue.LOW_RESOLUTION, EvidenceIssue.LOW_LIGHT),
        )
        np.testing.assert_array_equal(prepared.original_rgb[0, 0], [10, 10, 10])

    def test_same_input_and_config_produce_identical_outputs(self) -> None:
        source = image_bytes(size=(641, 359), color=(60, 70, 80))
        preprocessor = FramePreprocessor()
        frame = raw_frame(source)

        first = preprocessor.prepare(frame)
        second = preprocessor.prepare(frame)

        self.assertEqual(first.source_sha256, second.source_sha256)
        self.assertEqual(first.transform, second.transform)
        self.assertEqual(first.quality_flags, second.quality_flags)
        np.testing.assert_array_equal(first.original_rgb, second.original_rgb)
        np.testing.assert_array_equal(first.model_rgb, second.model_rgb)

    def test_rejects_invalid_metadata(self) -> None:
        invalid_frames = [
            raw_frame(frame_id=""),
            raw_frame(camera_id="  "),
            raw_frame(timestamp="2026-08-25T08:15:30"),
            raw_frame(timestamp="not-a-timestamp"),
            raw_frame(source_ref=""),
        ]

        for frame in invalid_frames:
            with self.subTest(frame=frame):
                self.assert_error(frame, "INVALID_FRAME_METADATA")

    def test_rejects_empty_oversized_unsupported_and_corrupt_media(self) -> None:
        self.assert_error(raw_frame(b""), "EMPTY_IMAGE_BYTES")

        oversized_config = PreprocessingConfig(max_source_bytes=10)
        self.assert_error(
            raw_frame(image_bytes(size=(20, 20))),
            "IMAGE_TOO_LARGE",
            oversized_config,
        )

        gif_source = image_bytes(size=(20, 20), image_format="GIF")
        self.assert_error(raw_frame(gif_source), "UNSUPPORTED_IMAGE_MIME")
        self.assert_error(raw_frame(b"not an image"), "INVALID_IMAGE_CONTENT")
        self.assert_error(raw_frame(b"\xff\xd8\xfftruncated jpeg"), "INVALID_IMAGE_CONTENT")

    def test_rejects_images_over_the_pixel_limit(self) -> None:
        config = PreprocessingConfig(max_source_pixels=399)

        self.assert_error(
            raw_frame(image_bytes(size=(20, 20))),
            "IMAGE_PIXEL_LIMIT_EXCEEDED",
            config,
        )

    def test_validates_config_and_transform_inputs(self) -> None:
        invalid_configs = [
            {"target_size": (0, 640)},
            {"padding_value": 256},
            {"max_source_bytes": 0},
            {"max_source_pixels": 0},
            {"min_short_side": 0},
            {"low_light_threshold": -1.0},
            {"low_light_threshold": 256.0},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                PreprocessingConfig(**kwargs)  # type: ignore[arg-type]

        transform = ImageTransform((10, 10), (20, 20), 2.0, (0, 0, 0, 0))
        with self.assertRaises(ValueError):
            transform.bbox_to_model([0.0, 1.0, 2.0])
        with self.assertRaises(ValueError):
            transform.polygon_to_model([[0.0], [1.0, 2.0]])

    def test_prepared_frame_is_shared_by_ppe_and_zone_without_coordinate_drift(self) -> None:
        prepared = FramePreprocessor().prepare(raw_frame())
        person_bbox_model = [80.0, 200.0, 160.0, 400.0]
        configured_zone_original = [[0.0, 0.0], [800.0, 0.0], [800.0, 400.0]]

        def ppe_stub(frame: PreparedFrame) -> list[float]:
            self.assertIs(frame.model_rgb, prepared.model_rgb)
            return frame.transform.bbox_to_original(person_bbox_model)

        def zone_stub(frame: PreparedFrame) -> list[list[float]]:
            self.assertIs(frame.original_rgb, prepared.original_rgb)
            return configured_zone_original

        self.assertEqual(ppe_stub(prepared), [100.0, 50.0, 200.0, 300.0])
        self.assertEqual(zone_stub(prepared), configured_zone_original)

    def assert_error(
        self,
        frame: RawCameraFrame,
        reason_code: str,
        config: PreprocessingConfig | None = None,
    ) -> None:
        with self.assertRaises(FramePreprocessingError) as caught:
            FramePreprocessor(config).prepare(frame)
        self.assertEqual(caught.exception.reason_code, reason_code)


class TestRawCameraFrameIsolation(unittest.TestCase):
    def test_preprocessing_does_not_mutate_the_raw_frame(self) -> None:
        frame = raw_frame()
        snapshot = replace(frame)

        FramePreprocessor().prepare(frame)

        self.assertEqual(frame, snapshot)


if __name__ == "__main__":
    unittest.main()
