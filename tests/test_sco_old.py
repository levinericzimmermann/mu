import unittest
from mu.sco import old
from mu.mel import mel
from mu.mel import ji
from mu.rhy import rhy


class ToneTest(unittest.TestCase):
    def test_abstract_error(self):
        pass


class MelodyTest(unittest.TestCase):
    p0 = ji.r(14, 9)
    d0 = rhy.RhyUnit(400)
    t0 = old.Tone(p0, d0)
    mel0 = mel.Mel([p0] * 3)
    rhy0 = rhy.RhyCompound([d0] * 3)
    melody0 = old.Melody([t0] * 3)

    def test_constructor(self):
        old.Melody([self.t0, self.t0, self.t0])

    def test_alternative_constructor(self):
        melody1 = old.Melody.from_parameter(self.mel0, self.rhy0)
        self.assertEqual(self.melody0, melody1)

    def test_duration(self):
        self.assertEqual(self.melody0.duration, sum(self.rhy0))

    def test_get_attributes(self):
        self.assertEqual(self.melody0.mel, self.mel0)
        self.assertEqual(self.melody0.rhy, self.rhy0)

    def test_set_attributes(self):
        melody0 = old.Melody([])
        melody0.mel = self.mel0
        melody0.rhy = self.rhy0
        self.assertEqual(melody0.mel, self.mel0)
        self.assertEqual(melody0.rhy, self.rhy0)

    def test_freq(self):
        self.assertEqual(self.melody0.freq, self.mel0.freq)


"""
class CadenceTest(unittest.TestCase):
    p0 = ji.r(5, 4)
    p1 = ji.r(3, 2)
    p2 = ji.r(1, 1)
    p3 = ji.r(6, 5)
    p4 = ji.r(7, 4)
    p5 = ji.r(9, 8)
    h0 = mel.Harmony([p0, p1, p2])
    h1 = mel.Harmony([p3, p4, p5])
    d0 = rhy.RhyUnit(400)
    rhy0 = rhy.RhyCompound([d0] * 2)
    chord0 = old.Chord(h0, d0)
    chord1 = old.Chord(h1, d0)
    cadence0 = old.Cadence([chord0, chord1])

    def test_constructor(self):
        old.Cadence([self.chord0, self.chord1])

    def test_alternative_constructor(self):
        cadence1 = old.Cadence.from_parameter((self.h0, self.h1), self.rhy0)
        self.assertEqual(self.cadence0, cadence1)
"""


if __name__ == "__main__":
    unittest.main()
