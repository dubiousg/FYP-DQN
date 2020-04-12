import unittest
import math

class Tester(unittest.TestCase):

    def test_equal(self, input, expected):
        self.assertEqual(input, expected)

    #cannot be used for nans
    def test_not_equal(self, input, expected):
        self.assertFalse(input == expected, "failed")

    def test_not_nan(self, input):
        self.assertFalse(math.isnan(input))

    def test_greater(self, threshold, input):
        self.assertTrue(input < threshold)
        self.assertFalse(input > threshold)

    def test_type(self, input, expected):
        self.assertEqual(type(input), expected)