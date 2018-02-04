# @Author: Levin Eric Zimmermann <Levin_Eric_Zimmermann>
# @Date:   2018-02-04T00:42:34+01:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   Levin_Eric_Zimmermann
# @Last modified time: 2018-02-04T00:42:48+01:00


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
