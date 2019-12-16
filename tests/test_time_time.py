import unittest
from mu.time import time


class TimeTest(unittest.TestCase):
    def test_conversion(self):
        self.assertEqual(time.Time.seconds2miliseconds(1), 1000)
        self.assertEqual(time.Time.minutes2miliseconds(1), 1000 * 60)
        self.assertEqual(time.Time.hours2miliseconds(1), 1000 * 60 * 60)

    def test_constructor(self):
        self.assertEqual(time.Time.from_seconds(1), time.Time(1000))
        self.assertEqual(time.Time.from_minutes(1), time.Time(1000 * 60))
        self.assertEqual(time.Time.from_hours(1), time.Time(1000 * 60 * 60))
