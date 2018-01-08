import unittest
from mu.utils import music21
from mu.rhy import rhy
from mu.sco import old
from mu.mel import ji


class TestRhythmConverter(unittest.TestCase):
    def test_converter(self):
        # TODO: test 1 / 3 or 1 / 5
        r_mu_0 = rhy.RhyUnit(2)
        r_m21_0 = music21.m21.duration.Duration(2)
        r_mu_1 = rhy.RhyUnit(3.5)
        r_m21_1 = music21.m21.duration.Duration(3.5)
        r_mu_2 = rhy.RhyUnit(1.5)
        r_m21_2 = music21.m21.duration.Duration(1.5)
        r_mu_3 = rhy.RhyUnit(0.25)
        r_m21_3 = music21.m21.duration.Duration(0.25)
        r_mu_4 = rhy.RhyUnit(0.125)
        r_m21_4 = music21.m21.duration.Duration(0.125)
        r_mu_5 = rhy.RhyUnit(0.0625)
        r_m21_5 = music21.m21.duration.Duration(0.0625)
        r_mu_6 = rhy.RhyUnit(0.375)
        r_m21_6 = music21.m21.duration.Duration(0.375)
        r_mu_7 = rhy.RhyUnit(2.375)
        r_m21_7 = music21.m21.duration.Duration(2.375)
        self.assertEqual(r_mu_0.convert2music21(), r_m21_0)
        self.assertEqual(r_mu_1.convert2music21(), r_m21_1)
        self.assertEqual(r_mu_2.convert2music21(), r_m21_2)
        self.assertEqual(r_mu_3.convert2music21(), r_m21_3)
        self.assertEqual(r_mu_4.convert2music21(), r_m21_4)
        self.assertEqual(r_mu_5.convert2music21(), r_m21_5)
        self.assertEqual(r_mu_6.convert2music21(), r_m21_6)
        self.assertEqual(r_mu_7.convert2music21(), r_m21_7)


class TestPitchConverter(unittest.TestCase):
    def test_converter(self):
        p_mu_0 = ji.r(1, 1)
        p_mu_0.multiply = 440
        p_m21_0 = music21.m21.pitch.Pitch('a4')
        p_mu_1 = ji.r(1, 1)
        p_mu_1.multiply = 445
        p_mu_1_converted = p_mu_1.convert2music21()
        ct_deviation = p_mu_1_converted.microtone.cents
        p_m21_1 = music21.m21.pitch.Pitch('a4')
        p_m21_1.microtone = music21.m21.pitch.Microtone(ct_deviation)
        self.assertEqual(p_mu_0.convert2music21(), p_m21_0)
        self.assertEqual(p_mu_1_converted, p_m21_1)


class TestToneConverter(unittest.TestCase):
    def test_converter(self):
        p_mu_0 = ji.r(1, 1)
        p_mu_0.multiply = 440
        p_m21_0 = music21.m21.pitch.Pitch('a4')
        r_mu_0 = rhy.RhyUnit(2)
        r_m21_0 = music21.m21.duration.Duration(2)
        t_mu_0 = old.Tone(p_mu_0, r_mu_0)
        t_m21_0 = music21.m21.note.Note(p_m21_0, duration=r_m21_0)
        stream0 = music21.m21.stream.Stream([t_m21_0])
        self.assertEqual(t_mu_0.convert2music21()[0], stream0[0])
        r_mu_1 = rhy.RhyUnit(3)
        t_mu_1 = old.Tone(p_mu_0, r_mu_1, r_mu_0)
        r_m21_1 = music21.m21.duration.Duration(1)
        rest_m21_0 = music21.m21.note.Rest(duration=r_m21_1)
        stream1 = music21.m21.stream.Stream([t_m21_0, rest_m21_0])
        self.assertEqual(t_mu_1.convert2music21()[0], stream1[0])
        self.assertEqual(t_mu_1.convert2music21()[1], stream1[1])


class TestMelodyConverter(unittest.TestCase):
    def test_converter(self):
        p_mu_0 = ji.r(1, 1)
        p_mu_0.multiply = 440
        p_m21_0 = music21.m21.pitch.Pitch('a4')
        r_mu_0 = rhy.RhyUnit(2)
        r_m21_0 = music21.m21.duration.Duration(2)
        t_mu_0 = old.Tone(p_mu_0, r_mu_0)
        t_m21_0 = music21.m21.note.Note(p_m21_0, duration=r_m21_0)
        melody_mu_0 = old.Melody([t_mu_0, t_mu_0])
        melody_mu_0_converted = melody_mu_0.convert2music21()
        melody_m21_0 = music21.m21.stream.Stream([t_m21_0, t_m21_0])
        for t0, t1 in zip(melody_mu_0_converted, melody_m21_0):
            self.assertEqual(t0, t1)


class TestPolyConverter(unittest.TestCase):
    def test_converter0(self):
        t_mu_0 = old.Tone(ji.r(3, 2), rhy.RhyUnit(2))
        melody_mu_0 = old.Melody([t_mu_0, t_mu_0])
        polyphon_mu_0 = old.Polyphon([melody_mu_0, melody_mu_0])
        m21_obj = polyphon_mu_0.convert2music21()
        self.assertEqual(len(m21_obj), 2)
