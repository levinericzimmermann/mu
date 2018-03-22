# @Author: Levin Eric Zimmermann
# @Date:   2018-02-07T18:28:10+01:00
# @Email:  levin-eric.zimmermann@folkwang-uni.de
# @Project: mu
# @Last modified by:   uummoo
# @Last modified time: 2018-03-13T10:39:43+01:00


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
    t1 = old.Tone(p1, d1)
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
        melody1 = old.Melody([old.Rest(3)])
        self.assertEqual(melody1.duration, 3)

    def test_get_attributes(self):
        self.assertEqual(self.melody0.get_pitch(), self.mel0)
        self.assertEqual(self.melody0.get_delay(), self.rhy0)
        self.assertEqual(self.melody0.get_dur(), self.rhy0)

    def test_set_attributes(self):
        melody0 = old.Melody([])
        melody0.set_pitch(self.mel0)
        melody0.set_delay(self.rhy0)
        self.assertEqual(melody0.get_pitch(), self.mel0)
        self.assertEqual(melody0.get_delay(), self.rhy0)
        self.assertEqual(melody0.sequences[0], self.mel0)
        self.assertEqual(melody0.sequences[1], self.rhy0)
        melody0.set_pitch(self.mel1)
        melody0.set_delay(self.rhy1)
        melody0.set_dur(self.rhy0)
        self.assertEqual(melody0.get_pitch(), self.mel1)
        self.assertEqual(melody0.get_delay(), self.rhy1)
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

    def test_add(self):
        compound = old.Melody([self.t0, self.t1, self.t1])
        melody0 = old.Melody([self.t0])
        melody1 = old.Melody([self.t1] * 2)
        self.assertEqual(melody0 + melody1, compound)

    def test_tie(self):
        melodyTest0 = old.Melody([old.Tone(self.t0.pitch, self.t0.delay * 3)])
        self.assertEqual(self.melody0.tie(), melodyTest0)
        melodyTest1 = old.Melody([old.Tone(self.t0.pitch, self.t0.delay * 2),
                                  self.t1])
        melody1 = old.Melody([self.t0, self.t0, self.t1])
        self.assertEqual(melody1.tie(), melodyTest1)
        melody2 = old.Melody([self.t0, self.t1, self.t0])
        self.assertEqual(melody2.tie(), melody2)

    def test_split(self):
        tone0 = old.Tone(ji.r(1, 1, 2), rhy.RhyUnit(2), rhy.RhyUnit(1))
        tone0B = old.Tone(ji.r(1, 1, 2), rhy.RhyUnit(1), rhy.RhyUnit(1))
        tone1 = old.Tone(ji.r(1, 1, 2), rhy.RhyUnit(3), rhy.RhyUnit(1))
        tone1B = old.Tone(ji.r(1, 1, 2), rhy.RhyUnit(1), rhy.RhyUnit(1))
        pause0 = old.Rest(rhy.RhyUnit(1))
        pause1 = old.Rest(rhy.RhyUnit(2))
        melody0 = old.Melody([tone0, tone1])
        melody1 = old.Melody([tone0B, pause0, tone1B, pause1])
        self.assertEqual(melody0.split(), melody1)


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
    t6_set = old.Tone(p5, rhy.RhyUnit(1), rhy.RhyUnit(5))
    mel0 = old.Melody([t0, t1, t2, t3, t4, t5])
    mel1 = old.Melody([old.Rest(rhy.RhyUnit(1)), t1, t2, t3, t4, t5])
    mel2 = old.Melody([t0, t1])
    set0 = old.ToneSet([t0_set, t1_set, t2_set, t3_set, t4_set, t5_set])
    set1 = old.ToneSet([t1_set, t2_set, t3_set, t4_set, t5_set])
    set2 = old.ToneSet([t1_set, t6_set, t2_set])

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

    def test_pop_by_time(self):
        for t in self.set0.pop_by_time(1):
            self.assertEqual(t, self.t1_set)
        for t in self.set0.pop_by_time(1.5):
            self.assertEqual(t, self.t1_set)
        test_set0 = self.set2.pop_by_time(1.5)
        test_set_compare0 = old.ToneSet([self.t1_set,
                                        self.t6_set])
        test_set1 = self.set2.pop_by_time(2.7)
        test_set_compare1 = old.ToneSet([self.t2_set,
                                        self.t6_set])
        self.assertEqual(test_set0, test_set_compare0)
        self.assertEqual(test_set1, test_set_compare1)

    def test_pop_by_correct_dur_and_delay(self):
        poped_by = self.set0.pop_by_pitch(self.p0, self.p5)
        melody = poped_by.convert2melody()
        self.assertEqual(melody[0].delay, rhy.RhyUnit(5))
        self.assertEqual(melody[0].duration, rhy.RhyUnit(1))


class PolyTest(unittest.TestCase):
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
    melody0 = old.JIMelody((t0, t1))
    melody1 = old.JIMelody((t2, t3))
    poly0 = old.Polyphon([melody0, melody1])

    def test_chordify(self):
        chord0 = old.Chord(ji.JIHarmony([self.t0, self.t2]), rhy.RhyUnit(1))
        chord1 = old.Chord(ji.JIHarmony([self.t1, self.t3]), rhy.RhyUnit(1))
        cadence0 = old.Cadence([chord0, chord1])
        self.assertEqual(cadence0, self.poly0.chordify())


if __name__ == "__main__":
    unittest.main()
