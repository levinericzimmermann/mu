import unittest
from mu.sco import old
from mu.mel import abstract as mel_abstract
from mu.mel import ji
from mu.rhy import rhy


class ToneTest(unittest.TestCase):
    def test_abstract_error(self):
        pass


class MelodyTest(unittest.TestCase):
    p0 = ji.r(14, 9)
    d0 = rhy.RhyUnit(400)
    t0 = old.Tone(p0, d0)
    mel0 = mel_abstract.Mel([p0] * 3)
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


if __name__ == "__main__":
    unittest.main()
