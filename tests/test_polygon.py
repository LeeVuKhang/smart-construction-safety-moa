"""Tests for polygon validation and point membership."""

import unittest

from src.geometry.polygon import point_in_polygon, validate_polygon


SQUARE = [(0, 0), (10, 0), (10, 10), (0, 10)]


class PolygonTests(unittest.TestCase):
    def test_point_inside_polygon(self):
        self.assertTrue(point_in_polygon((5, 5), SQUARE))

    def test_point_on_boundary_is_inside(self):
        self.assertTrue(point_in_polygon((10, 5), SQUARE))

    def test_point_outside_polygon(self):
        self.assertFalse(point_in_polygon((15, 5), SQUARE))

    def test_malformed_polygon_raises_error(self):
        with self.assertRaises(ValueError):
            validate_polygon([(0, 0), (10, 10)])


if __name__ == "__main__":
    unittest.main()
