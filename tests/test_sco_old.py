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
    p1 = ji.r(7, 4)
    d0 = rhy.RhyUnit(400)
    d1 = rhy.RhyUnit(800)
    t0 = old.Tone(p0, d0)
    mel0 = mel.Mel([p0] * 3)
    mel1 = mel.Mel([p1] * 3)
    rhy0 = rhy.RhyCompound([d0] * 3)
    rhy1 = rhy.RhyCompound([d1] * 3)
    melody0 = old.Melody([t0] * 3)

    def test_constructor(self):
        old.Melody([self.t0, self.t0, self.t0])

    def test_alternative_constructor(self):
        melody1 = old.Melody.from_parameter(self.mel0, self.rhy0)
        self.assertEqual(self.melody0, melody1)

    def test_duration(self):
        self.assertEqual(self.melody0.duration, sum(self.rhy0))

    def test_get_attributes(self):
        self.assertEqual(self.melody0.get_mel(), self.mel0)
        self.assertEqual(self.melody0.get_rhy(), self.rhy0)
        self.assertEqual(self.melody0.get_dur(), self.rhy0)

    def test_set_attributes(self):
        melody0 = old.Melody([])
        melody0.set_mel(self.mel0)
        melody0.set_rhy(self.rhy0)
        self.assertEqual(melody0.get_mel(), self.mel0)
        self.assertEqual(melody0.get_rhy(), self.rhy0)
        self.assertEqual(melody0.sequences[0], self.mel0)
        self.assertEqual(melody0.sequences[1], self.rhy0)
        melody0.set_mel(self.mel1)
        melody0.set_rhy(self.rhy1)
        melody0.set_dur(self.rhy0)
        self.assertEqual(melody0.get_mel(), self.mel1)
        self.assertEqual(melody0.get_rhy(), self.rhy1)
        self.assertEqual(melody0.get_dur(), self.rhy0)
        self.assertEqual(melody0.sequences[0], self.mel1)
        self.assertEqual(melody0.sequences[1], self.rhy1)
        self.assertEqual(melody0.sequences[2], self.rhy0)

    def test_get_attributes_syntactic_sugar(self):
        self.assertEqual(self.melody0.mel, self.mel0)
        self.assertEqual(self.melody0.rhy, self.rhy0)
        self.assertEqual(self.melody0.dur, self.rhy0)

    def test_set_attributes_syntactic_sugar(self):
        melody0 = old.Melody([])
        melody0.mel = self.mel0
        melody0.rhy = self.rhy0
        self.assertEqual(melody0.mel, self.mel0)
        self.assertEqual(melody0.rhy, self.rhy0)
        self.assertEqual(melody0.sequences[0], self.mel0)
        self.assertEqual(melody0.sequences[1], self.rhy0)
        melody0.mel = self.mel1
        melody0.rhy = self.rhy1
        melody0.dur = self.rhy0
        self.assertEqual(melody0.mel, self.mel1)
        self.assertEqual(melody0.rhy, self.rhy1)
        self.assertEqual(melody0.sequences[0], self.mel1)
        self.assertEqual(melody0.sequences[1], self.rhy1)
        self.assertEqual(melody0.sequences[2], self.rhy0)

    def test_freq(self):
        self.assertEqual(self.melody0.freq, self.mel0.freq)


class ToneSetTest(unittest.TestCase):
    p0 = ji.r(5, 4)
    p1 = ji.r(3, 2)
    p2 = ji.r(1, 1)
    p3 = ji.r(6, 5)
    p4 = ji.r(7, 4)
    p5 = ji.r(9, 8)
    t0 = old.Tone(p0, rhy.RhyUnit(1))
    t1 = old.Tone(p1, rhy.RhyUnit(1))
    t2 = old.Tone(p2, rhy.RhyUnit(1))
    t3 = old.Tone(p3, rhy.RhyUnit(1))
    t3 = old.Tone(p3, rhy.RhyUnit(1))
    t4 = old.Tone(p4, rhy.RhyUnit(1))
    t5 = old.Tone(p5, rhy.RhyUnit(1))
    t0_set = old.Tone(p0, rhy.RhyUnit(0), rhy.RhyUnit(1))
    t1_set = old.Tone(p1, rhy.RhyUnit(1), rhy.RhyUnit(1))
    t2_set = old.Tone(p2, rhy.RhyUnit(2), rhy.RhyUnit(1))
    t3_set = old.Tone(p3, rhy.RhyUnit(3), rhy.RhyUnit(1))
    t4_set = old.Tone(p4, rhy.RhyUnit(4), rhy.RhyUnit(1))
    t5_set = old.Tone(p5, rhy.RhyUnit(5), rhy.RhyUnit(1))
    mel0 = old.Melody([t0, t1, t2, t3, t4, t5])
    mel1 = old.Melody([old.Rest(rhy.RhyUnit(1)), t1, t2, t3, t4, t5])
    mel2 = old.Melody([t0, t1])
    set0 = old.ToneSet([t0_set, t1_set, t2_set, t3_set, t4_set, t5_set])
    set1 = old.ToneSet([t1_set, t2_set, t3_set, t4_set, t5_set])

    def test_constructor(self):
        self.assertEqual(old.ToneSet.from_melody(ToneSetTest.mel0),
                         ToneSetTest.set0)

    def test_converter(self):
        self.assertEqual(ToneSetTest.mel0,
                         ToneSetTest.set0.convert2melody())
        self.assertEqual(ToneSetTest.mel1,
                         ToneSetTest.set1.convert2melody())

    def test_pop_by(self):
        popped = ToneSetTest.set0.copy().pop_by_pitch(
            ToneSetTest.p0, ToneSetTest.p1)
        self.assertEqual(ToneSetTest.mel2, popped.convert2melody())
        popped = ToneSetTest.set0.copy().pop_by_start(
            rhy.RhyUnit(0), rhy.RhyUnit(1))
        self.assertEqual(ToneSetTest.mel2, popped.convert2melody())


if __name__ == "__main__":
    unittest.main()
