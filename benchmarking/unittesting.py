import unittest
import math

#Tester class
#   *a unit testing class that uses assertions
#   *all functions use assertions to test inputs
#   *function behaviour should be apparent from name
class Tester(unittest.TestCase):

    def test_equal(self, input, expected):
        self.assertEqual(input, expected)

    def test_not_equal(self, input, expected):
        self.assertFalse(input == expected, "failed")

    def test_not_nan(self, input):
        self.assertFalse(math.isnan(input))

    def test_greater(self, threshold, input):
        self.assertTrue(input < threshold)
        self.assertFalse(input > threshold)

    def test_type(self, input, expected):
        self.assertEqual(type(input), expected)