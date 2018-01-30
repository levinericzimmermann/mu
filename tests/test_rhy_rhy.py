import unittest
from mu.time import time
from mu.rhy import rhy


class RhyTest(unittest.TestCase):
    def test_delay(self):
        rh0 = rhy.RhyUnit(4)
        rhy_comp0 = rhy.RhyCompound([rhy.RhyUnit(2),
                                     rhy.RhyUnit(4),
                                     rhy.RhyUnit(1)])
        rhy_comp1 = rhy.RhyCompound([rhy.RhyUnit(4),
                                     rhy.RhyUnit(8),
                                     rhy.RhyUnit(2)])
        self.assertEqual(rh0.delay, time.Time(4))
        self.assertEqual(rhy_comp0.delay, time.Time(7))
        self.assertEqual(rhy_comp1.delay, time.Time(14))

    def test_stretch(self):
        rh0 = rhy.RhyUnit(2)
        rh1 = rhy.RhyUnit(4)
        rh2 = rhy.RhyUnit(1)
        self.assertEqual(rh0.stretch(2), rh1)
        self.assertEqual(rh0.stretch(0.5), rh2)
        rhy_comp0 = rhy.RhyCompound([rhy.RhyUnit(2),
                                     rhy.RhyUnit(4),
                                     rhy.RhyUnit(1)])
        rhy_comp1 = rhy.RhyCompound([rhy.RhyUnit(4),
                                     rhy.RhyUnit(8),
                                     rhy.RhyUnit(2)])
        rhy_comp2 = rhy.RhyCompound([rhy.RhyUnit(1),
                                     rhy.RhyUnit(2),
                                     rhy.RhyUnit(0.5)])
        self.assertEqual(rhy_comp0, rhy_comp1.stretch(0.5))
        self.assertEqual(rhy_comp0.stretch(2), rhy_comp1)
        self.assertEqual(rhy_comp0.stretch(0.5), rhy_comp2)

    def test_convert2music21(self):
        pass


class PulseChromaTest(unittest.TestCase):
    def test_count_subpulse(self):
        chrome0 = rhy.PulseChroma(90)
        chrome1 = rhy.PulseChroma(30)
        chrome2 = rhy.PulseChroma(15)
        chrome3 = rhy.PulseChroma(10)
        chrome4 = rhy.PulseChroma(6)
        chrome5 = rhy.PulseChroma(2)
        self.assertEqual(chrome0.count_subpulse(), (3,))
        self.assertEqual(set(chrome1.count_subpulse()), set((2, 3, 5)))
        self.assertEqual(set(chrome2.count_subpulse()), set((3, 5)))
        self.assertEqual(set(chrome3.count_subpulse()), set((2, 5)))
        self.assertEqual(set(chrome4.count_subpulse()), set((2, 3)))
        self.assertEqual(chrome5.count_subpulse(), (0,))

    def test_subpulse(self):
        chrome0 = rhy.PulseChroma(12)
        chrome1 = rhy.PulseChroma(6)
        chrome2 = rhy.PulseChroma(3)
        chrome3 = rhy.PulseChroma(2)
        self.assertEqual(chrome0.subpulse, (chrome1,))
        self.assertEqual(set(chrome1.subpulse), set((chrome2, chrome3)))
        chrome0 = rhy.PulseChroma(90)
        chrome1 = rhy.PulseChroma(30)
        chrome2 = rhy.PulseChroma(15)
        chrome3 = rhy.PulseChroma(10)
        chrome4 = rhy.PulseChroma(6)
        chrome5 = rhy.PulseChroma(2)
        chrome6 = rhy.PulseChroma(3)
        chrome7 = rhy.PulseChroma(5)
        self.assertEqual(chrome0.subpulse, (chrome1,))
        self.assertEqual(set(chrome1.subpulse), set(
                (chrome2, chrome3, chrome4)))
        self.assertEqual(set(chrome2.subpulse), set((chrome6, chrome7)))
        self.assertEqual(set(chrome3.subpulse), set((chrome5, chrome7)))
        self.assertEqual(set(chrome4.subpulse), set((chrome5, chrome6)))
