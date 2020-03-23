import unittest

from mu.mel import ji
from mu.mel import mel
from mu.rhy import rhy
from mu.sco import old


class InterpolationEventTest(unittest.TestCase):
    def test_construction(self):
        self.assertRaises(TypeError, old.InterpolationEvent)

    def test_pitch_interpolation(self):
        p0 = old.PitchInterpolation(3, mel.SimplePitch(0, 200))
        p1 = old.PitchInterpolation(0, mel.SimplePitch(0, 0))
        p2 = old.PitchInterpolation(0, mel.SimplePitch(0, 0))
        self.assertNotEqual(hash(p0), hash(p1))
        self.assertEqual(hash(p1), hash(p2))
        interp0 = (200, 100, 0)
        self.assertEqual(p0.interpolate(p1, 3), interp0)

    def test_rhythmic_interpolation(self):
        p0 = old.RhythmicInterpolation(3, rhy.Unit(3))
        p1 = old.RhythmicInterpolation(0, rhy.Unit(2))
        p2 = old.RhythmicInterpolation(0, rhy.Unit(2))
        self.assertNotEqual(hash(p0), hash(p1))
        self.assertEqual(hash(p1), hash(p2))
        interp0 = (3, 2.5, 2)
        self.assertEqual(p0.interpolate(p1, 3), interp0)


class InterpolationLineTest(unittest.TestCase):
    def test_interpolation(self):
        p0 = old.PitchInterpolation(3, mel.SimplePitch(0, 300))
        p1 = old.PitchInterpolation(4, mel.SimplePitch(0, 0))
        p2 = old.PitchInterpolation(0, mel.SimplePitch(0, -400))
        line = old.InterpolationLine((p0, p1, p2))
        interpol = line(1)
        self.assertEqual(interpol, (300, 200, 100, 0, -100, -200, -300))
        self.assertRaises(ValueError, lambda: old.InterpolationLine((p0, p0, p1)))

    def test_glissando_line(self):
        p0 = old.PitchInterpolation(3, mel.SimplePitch(0, 300))
        p1 = old.PitchInterpolation(4, mel.SimplePitch(0, 0))
        p2 = old.PitchInterpolation(0, mel.SimplePitch(0, -400))
        line = old.InterpolationLine((p0, p1, p2))
        gliss = old.GlissandoLine(line)
        interpol0 = gliss.interpolate(1)
        self.assertEqual(interpol0, (300, 200, 100, 0, -100, -200, -300))

    def test_vibrato_line(self):
        p0 = old.PitchInterpolation(2, mel.SimplePitch(0, 300))
        p1 = old.PitchInterpolation(4, mel.SimplePitch(0, 200))
        p2 = old.PitchInterpolation(0, mel.SimplePitch(0, 600))
        p3 = old.PitchInterpolation(4, mel.SimplePitch(0, -200))
        p4 = old.PitchInterpolation(2, mel.SimplePitch(0, -600))
        p5 = old.PitchInterpolation(0, mel.SimplePitch(0, -900))
        r0 = old.RhythmicInterpolation(6, rhy.Unit(2))
        r1 = old.RhythmicInterpolation(0, rhy.Unit(2))
        line0 = old.InterpolationLine((p0, p1, p2))
        line1 = old.InterpolationLine((p3, p4, p5))
        line2 = old.InterpolationLine((r0, r1))
        vib0 = old.VibratoLine(line0, line1, line2, "up")
        vib1 = old.VibratoLine(line0, line1, line2, "down")
        interpol0 = vib0.interpolate(0.5)
        interpol1 = vib1.interpolate(0.5)
        self.assertNotEqual(interpol0, interpol1)
        self.assertEqual(
            tuple(round(n, 2) for n in interpol0),
            (0, 275, 0, -371.43, 0, 250, 0, -600, 0, 450, 0, -840),
        )
        self.assertEqual(
            tuple(round(n, 2) for n in interpol1),
            (0, -257.14, 0, 225, 0, -485.71, 0, 350, 0, -720, 0, 550),
        )


class MelodyTest(unittest.TestCase):
    p0 = ji.r(14, 9)
    p1 = ji.r(7, 4)
    d0 = rhy.Unit(400)
    d1 = rhy.Unit(800)
    t0 = old.Tone(p0, d0)
    t1 = old.Tone(p1, d1)
    mel0 = mel.Mel([p0] * 3)
    mel1 = mel.Mel([p1] * 3)
    rhy0 = rhy.Compound([d0] * 3)
    rhy1 = rhy.Compound([d1] * 3)
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

    def test_set_item(self):
        t0 = old.Tone(ji.r(1, 1), rhy.Unit(2))
        t1 = old.Tone(ji.r(2, 1), rhy.Unit(2))
        melody0 = old.Melody([t0, t1])
        melody1 = old.Melody([t1, t0])
        melody0[0], melody0[1] = melody1[0], melody1[1]
        self.assertEqual(melody0, melody1)

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
        melodyTest1 = old.Melody([old.Tone(self.t0.pitch, self.t0.delay * 2), self.t1])
        melody1 = old.Melody([self.t0, self.t0, self.t1])
        self.assertEqual(melody1.tie(), melodyTest1)
        melody2 = old.Melody([self.t0, self.t1, self.t0])
        self.assertEqual(melody2.tie(), melody2)

    def test_split(self):
        tone0 = old.Tone(ji.r(1, 1, 2), rhy.Unit(2), rhy.Unit(1))
        tone0B = old.Tone(ji.r(1, 1, 2), rhy.Unit(1), rhy.Unit(1))
        tone1 = old.Tone(ji.r(1, 1, 2), rhy.Unit(3), rhy.Unit(1))
        tone1B = old.Tone(ji.r(1, 1, 2), rhy.Unit(1), rhy.Unit(1))
        pause0 = old.Rest(rhy.Unit(1))
        pause1 = old.Rest(rhy.Unit(2))
        melody0 = old.Melody([tone0, tone1])
        melody1 = old.Melody([tone0B, pause0, tone1B, pause1])
        self.assertEqual(melody0.split(), melody1)

    def test_cut_up_by_time(self):
        t0 = old.Tone(ji.r(1, 1), rhy.Unit(2))
        t1 = old.Tone(ji.r(2, 1), rhy.Unit(2))
        t2 = old.Tone(ji.r(1, 1), rhy.Unit(1))
        r0 = old.Rest(1)

        melody0 = old.Melody([t0, t1, t1, t0, t1])
        melody1 = old.Melody([t1, t1, t0])
        melody2 = old.Melody([r0, t1, t1, t0])
        melody3 = old.Melody([t2, t1, t1, t0])
        melody4 = old.Melody([t1, t1, t2])

        self.assertEqual(melody0.cut_up_by_time(2, 8), melody1)
        self.assertEqual(melody0.cut_up_by_time(1, 8), melody2)
        self.assertEqual(melody0.cut_up_by_time(1, 8, add_earlier=True), melody3)
        self.assertEqual(melody0.cut_up_by_time(2, 7, hard_cut=True), melody4)
        self.assertEqual(melody0.cut_up_by_time(2, 7, hard_cut=False), melody1)

    def test_convert2absolute(self):
        melody_converted = old.Melody(
            (
                old.Tone(self.p0, self.d0 * 0, self.d0 * 1),
                old.Tone(self.p0, self.d0 * 1, self.d0 * 2),
                old.Tone(self.p0, self.d0 * 2, self.d0 * 3),
            )
        )
        self.assertEqual(self.melody0.convert2absolute(), melody_converted)

        melody_converted = old.Melody(
            (
                old.Tone(self.p0, self.d0 * 0, self.d0 * 1),
                old.Tone(self.p0, self.d0 * 1, self.d0 * 2),
                old.Tone(self.p0, self.d0 * 2, self.d0 * 3),
            ),
            time_measure="relative",
        )
        self.assertEqual(self.melody0.convert2absolute(), melody_converted)

    def test_convert2relative(self):
        melody_converted = old.Melody(
            (
                old.Tone(self.p0, self.d0 * 0, self.d0 * 1),
                old.Tone(self.p0, self.d0 * 1, self.d0 * 2),
                old.Tone(self.p0, self.d0 * 2, self.d0 * 3),
            ),
            time_measure="absolute",
        )
        self.assertEqual(melody_converted.convert2relative(), self.melody0)

    def test_copy(self):
        melody0 = old.Melody([old.Tone(self.p0, self.d0), old.Tone(self.p0, self.d0)])
        self.assertEqual(melody0, melody0.copy())
        self.assertEqual(type(melody0.mel), (type(melody0.copy().mel)))


class ToneSetTest(unittest.TestCase):
    p0 = ji.r(5, 4)
    p1 = ji.r(3, 2)
    p2 = ji.r(1, 1)
    p3 = ji.r(6, 5)
    p4 = ji.r(7, 4)
    p5 = ji.r(9, 8)
    t0 = old.Tone(p0, rhy.Unit(1))
    t1 = old.Tone(p1, rhy.Unit(1))
    t2 = old.Tone(p2, rhy.Unit(1))
    t3 = old.Tone(p3, rhy.Unit(1))
    t3 = old.Tone(p3, rhy.Unit(1))
    t4 = old.Tone(p4, rhy.Unit(1))
    t5 = old.Tone(p5, rhy.Unit(1))
    t0_set = old.Tone(p0, rhy.Unit(0), rhy.Unit(1))
    t1_set = old.Tone(p1, rhy.Unit(1), rhy.Unit(1))
    t2_set = old.Tone(p2, rhy.Unit(2), rhy.Unit(1))
    t3_set = old.Tone(p3, rhy.Unit(3), rhy.Unit(1))
    t4_set = old.Tone(p4, rhy.Unit(4), rhy.Unit(1))
    t5_set = old.Tone(p5, rhy.Unit(5), rhy.Unit(1))
    t6_set = old.Tone(p5, rhy.Unit(1), rhy.Unit(5))
    mel0 = old.Melody([t0, t1, t2, t3, t4, t5])
    mel1 = old.Melody([old.Rest(rhy.Unit(1)), t1, t2, t3, t4, t5])
    mel2 = old.Melody([t0, t1])
    set0 = old.ToneSet([t0_set, t1_set, t2_set, t3_set, t4_set, t5_set])
    set1 = old.ToneSet([t1_set, t2_set, t3_set, t4_set, t5_set])
    set2 = old.ToneSet([t1_set, t6_set, t2_set])

    def test_constructor(self):
        self.assertEqual(old.ToneSet.from_melody(ToneSetTest.mel0), ToneSetTest.set0)

    def test_converter(self):
        self.assertEqual(ToneSetTest.mel0, ToneSetTest.set0.convert2melody())
        self.assertEqual(ToneSetTest.mel1, ToneSetTest.set1.convert2melody())

    def test_pop_by(self):
        popped = ToneSetTest.set0.copy().pop_by_pitch(ToneSetTest.p0, ToneSetTest.p1)
        self.assertEqual(ToneSetTest.mel2, popped.convert2melody())
        popped = ToneSetTest.set0.copy().pop_by_start(rhy.Unit(0), rhy.Unit(1))
        self.assertEqual(ToneSetTest.mel2, popped.convert2melody())

    def test_pop_by_time(self):
        for t in self.set0.pop_by_time(1):
            self.assertEqual(t, self.t1_set)
        for t in self.set0.pop_by_time(1.5):
            self.assertEqual(t, self.t1_set)
        test_set0 = self.set2.pop_by_time(1.5)
        test_set_compare0 = old.ToneSet([self.t1_set, self.t6_set])
        test_set1 = self.set2.pop_by_time(2.7)
        test_set_compare1 = old.ToneSet([self.t2_set, self.t6_set])
        self.assertEqual(test_set0, test_set_compare0)
        self.assertEqual(test_set1, test_set_compare1)

    def test_pop_by_correct_dur_and_delay(self):
        poped_by = self.set0.pop_by_pitch(self.p0, self.p5)
        melody = poped_by.convert2melody()
        self.assertEqual(melody[0].delay, rhy.Unit(5))
        self.assertEqual(melody[0].duration, rhy.Unit(1))


class PolyTest(unittest.TestCase):
    p0 = ji.r(5, 4)
    p1 = ji.r(3, 2)
    p2 = ji.r(1, 1)
    p3 = ji.r(6, 5)
    p4 = ji.r(7, 4)
    p5 = ji.r(9, 8)
    t0 = old.Tone(p0, rhy.Unit(1))
    t1 = old.Tone(p1, rhy.Unit(1))
    t2 = old.Tone(p2, rhy.Unit(1))
    t3 = old.Tone(p3, rhy.Unit(1))
    t3 = old.Tone(p3, rhy.Unit(1))
    t4 = old.Tone(p4, rhy.Unit(1))
    t5 = old.Tone(p5, rhy.Unit(1))
    t6 = old.Tone(p0, rhy.Unit(2))
    t7 = old.Tone(p0, rhy.Unit(0.5))
    t8 = old.Tone(p0, rhy.Unit(1.5))
    t9 = old.Tone(p1, rhy.Unit(1.5))
    t10 = old.Tone(p5, rhy.Unit(0.5))
    t11 = old.Tone(p2, rhy.Unit(1))
    melody0 = old.JIMelody((t0, t1))
    melody1 = old.JIMelody((t2, t3))
    melody2 = old.JIMelody((t6, t6, t0, t7))  # duration 5.5
    melody3 = old.JIMelody((t7, t6, t2, t2))  # duration 4.5
    melody4 = old.JIMelody((t7, t7, t7, t2, t2))  # duration 3.5
    melody5 = old.JIMelody((t10, t9, t3, t8, t0))  # duration 5.5
    melody6 = old.JIMelody((t6, t6, t2, t7))  # duration 5.5

    poly0 = old.Polyphon([melody0, melody1])
    poly1 = old.Polyphon([melody2, melody3, melody4])
    poly2 = old.Polyphon([melody6, melody5])

    def test_chordify(self):
        chord0 = old.Chord(ji.JIHarmony([self.t0, self.t2]), rhy.Unit(1))
        chord1 = old.Chord(ji.JIHarmony([self.t1, self.t3]), rhy.Unit(1))
        cadence0 = old.Cadence([chord0, chord1])
        self.assertEqual(cadence0, self.poly0.chordify())

        chord0 = old.Chord(ji.JIHarmony([self.p0, self.p5]), rhy.Unit(0.5))
        chord1 = old.Chord(ji.JIHarmony([self.p0, self.p1]), rhy.Unit(1.5))
        chord2 = old.Chord(ji.JIHarmony([self.p0, self.p3]), rhy.Unit(1))
        chord3 = old.Chord(ji.JIHarmony([self.p0]), rhy.Unit(1))
        chord4 = old.Chord(ji.JIHarmony([self.p0, self.p2]), rhy.Unit(0.5))
        chord5 = old.Chord(ji.JIHarmony([self.p0]), rhy.Unit(0.5))
        expected = old.Cadence([chord0, chord1, chord2, chord3, chord4, chord4, chord5])
        result = self.poly2.chordify(
            harmony_class=ji.JIHarmony, cadence_class=old.JICadence, add_longer=True
        )
        self.assertEqual(expected, result)

    def test_find_simultan_events(self):
        simultan_events0 = self.poly0.find_simultan_events(0, 0)
        self.assertEqual(simultan_events0, (self.poly0[1].convert2absolute()[0],))
        simultan_events1 = self.poly0.find_simultan_events(1, 1)
        self.assertEqual(simultan_events1, (self.poly0[0].convert2absolute()[1],))
        simultan_events2 = self.poly1.find_simultan_events(0, 1)
        simultan_events2_comp = (
            self.poly1[1].convert2absolute()[1],
            self.poly1[1].convert2absolute()[2],
            self.poly1[1].convert2absolute()[3],
            self.poly1[2].convert2absolute()[-2],
            self.poly1[2].convert2absolute()[-1],
        )
        self.assertEqual(simultan_events2, simultan_events2_comp)
        simultan_events3 = self.poly1.find_simultan_events(1, 1)
        simultan_events3_comp = (
            self.poly1[0].convert2absolute()[0],
            self.poly1[0].convert2absolute()[1],
            self.poly1[2].convert2absolute()[1],
            self.poly1[2].convert2absolute()[2],
            self.poly1[2].convert2absolute()[3],
        )
        self.assertEqual(simultan_events3, simultan_events3_comp)

    def test_find_exact_simultan_events(self):
        poly2 = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 2), old.Tone(ji.r(1, 1), 3)]),
                old.Melody([old.Tone(ji.r(3, 2), 3), old.Tone(ji.r(3, 2), 2)]),
                old.Melody([old.Tone(ji.r(4, 3), 1), old.Tone(ji.r(4, 3), 2)]),
            )
        )
        simultan_events4 = poly2.find_exact_simultan_events(0, 1)
        simultan_events4_expected = (
            old.Tone(ji.r(3, 2), 1, 1),
            old.Tone(ji.r(3, 2), 2, 2),
            old.Tone(ji.r(4, 3), 1, 1),
        )
        self.assertEqual(simultan_events4, simultan_events4_expected)

        simultan_events0 = self.poly0.find_exact_simultan_events(0, 0)
        self.assertEqual(simultan_events0, (self.poly0[1][0],))
        simultan_events1 = self.poly0.find_exact_simultan_events(0, 0, False)
        self.assertEqual(simultan_events1, (self.poly0[1].convert2absolute()[0],))
        simultan_events2 = self.poly1.find_exact_simultan_events(1, 0)
        simultan_events2_expected = (self.poly1[2][0], self.poly1[2][0])
        self.assertEqual(simultan_events2, simultan_events2_expected)
        simultan_events3 = self.poly1.find_exact_simultan_events(1, 1)
        simultan_events3_expected = (self.t8, self.t7, self.t7, self.t7, self.t2)
        self.assertEqual(simultan_events3, simultan_events3_expected)

    def test_cut_up_by_time(self):
        poly0 = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 2), old.Tone(ji.r(1, 1), 3)]),
                old.Melody([old.Tone(ji.r(3, 2), 3), old.Tone(ji.r(3, 2), 2)]),
                old.Melody([old.Tone(ji.r(4, 3), 1), old.Tone(ji.r(4, 3), 2)]),
            )
        )
        poly0_cut = poly0.cut_up_by_time(1, 3)
        poly0_cut_expected = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 1), old.Tone(ji.r(1, 1), 1)]),
                old.Melody([old.Tone(ji.r(3, 2), 2)]),
                old.Melody([old.Tone(ji.r(4, 3), 2)]),
            )
        )
        self.assertEqual(poly0_cut, poly0_cut_expected)

        poly1_cut = poly0.cut_up_by_time(1, 3, add_earlier=False)
        poly1_cut_expected = old.Polyphon(
            (
                old.Melody([old.Rest(1), old.Tone(ji.r(1, 1), 1)]),
                old.Melody([old.Rest(2)]),
                old.Melody([old.Tone(ji.r(4, 3), 2)]),
            )
        )
        self.assertEqual(poly1_cut, poly1_cut_expected)

        poly2_cut = poly0.cut_up_by_time(1, 3, hard_cut=False)
        poly2_cut_expected = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 2), old.Tone(ji.r(1, 1), 3)]),
                old.Melody([old.Tone(ji.r(3, 2), 3)]),
                old.Melody([old.Rest(1), old.Tone(ji.r(4, 3), 2)]),
            )
        )
        self.assertEqual(poly2_cut, poly2_cut_expected)

    def test_cut_up_by_idx(self):
        poly0 = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 2), old.Tone(ji.r(1, 1), 3)]),
                old.Melody([old.Tone(ji.r(3, 2), 3), old.Tone(ji.r(3, 2), 2)]),
                old.Melody([old.Tone(ji.r(4, 3), 1), old.Tone(ji.r(4, 3), 2)]),
            )
        )
        poly0_cut = poly0.cut_up_by_idx(2, 1)
        poly0_cut_expected = old.Polyphon(
            (
                old.Melody([old.Tone(ji.r(1, 1), 1), old.Tone(ji.r(1, 1), 1)]),
                old.Melody([old.Tone(ji.r(3, 2), 2)]),
                old.Melody([old.Tone(ji.r(4, 3), 2)]),
            )
        )
        self.assertEqual(poly0_cut, poly0_cut_expected)


if __name__ == "__main__":
    unittest.main()
