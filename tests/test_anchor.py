"""Tests for person anchor-point calculation."""

import unittest

from src.geometry.anchor import bottom_center


class AnchorTests(unittest.TestCase):
    def test_bottom_center_anchor_calculation(self):
        self.assertEqual(bottom_center((10, 20, 30, 80)), (20, 80))


if __name__ == "__main__":
    unittest.main()
