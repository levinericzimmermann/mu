import bisect
import itertools
from mu.utils import music21
from mu.mel import mel


class Period(object):
    def __init__(self, *stressed):
        self.__period = itertools.accumulate((0,) + stressed)
        self.__period = tuple(self.__period)
        self.__period_gen = itertools.cycle(stressed)
        self.__period_length = len(stressed)
        self.__period_sum = sum(stressed)
        self.__counter = 0
        self.__current_rhythm = 0

    def __repr__(self):
        return repr(self.__period)

    def __getitem__(self, idx):
        return self.__period[idx]

    def __len__(self):
        return self.__period_length

    def summed(self):
        return self.__period_sum

    def summed_rest(self):
        return self.__period_sum - self.current_rhythm

    @property
    def current_beat(self):
        """
        Return number of current beat.
        """
        return self.__counter

    @property
    def current_rhythm(self):
        return self.__current_rhythm

    def current_rhythm_add(self, val):
        added = float(self.current_rhythm + val)
        diff = float(added - self.summed())
        if diff >= 0:
            self.__current_rhythm = abs(diff)
        else:
            self.__current_rhythm = added

    @staticmethod
    def is_dotted(duration):
        possible_values = [1 / (2 ** n) for n
                           in range(8)]
        possible_values += [1 * (2 ** n) for n
                            in range(10)]
        return duration / 3 in possible_values

    def apply_rhythm(self, duration) -> list:
        def convert_cut2distribution(cut):
            def is_useable(item):
                # check 1. is duration a multiple of 2
                # (1/4, 1/2, 1/8 etc.)
                # or is duration a bindend duration
                return any((item % 2 == 0,
                           Period.is_dotted(item)))
            distribution = []
            cut = list(list(item) for item in cut)
            for i, item in enumerate(reversed(cut)):
                idx = - (i + 1)
                if item[1] is True and item[2] != "first":
                    summed = cut[idx-1][0] + item[0]
                    addable = cut[idx-1][1]
                    swapable = is_useable(summed)
                    if swapable is False and addable is True:
                        distribution.append(cut[idx -1][0])
                        cut[idx - 1][0] = item[0]
                    elif addable is True:
                        cut[idx - 1][0] += item[0]
                    else:
                        distribution.append(item[0])
                else:
                    # Actually you could go even deeper now
                    distribution.append(item[0])
            return list(reversed(distribution))

        try:
            assert(duration <= self.summed_rest())
        except AssertionError:
            msg = "APPLY_RHYTHM - Method can only "
            msg += "apply durations which are smaller "
            msg += "than the remaining beat."
            raise ValueError(msg)
        start = self.current_rhythm
        end = start + duration
        b_left = bisect.bisect_left(
            self.__period, start)
        b_right = bisect.bisect_right(
            self.__period, end)
        if b_right == 0:
            print("WARNING--Zero Duration rhythm")
            return []
        inbetween = self.__period[b_left:b_right]
        if inbetween:
            cut = []
            st = float(start)
            first = True
            for scissors in inbetween:
                diff = scissors - st
                if first is True:
                    if st in self.__period:
                        full = True
                    else:
                        full = False
                    msg = "first"
                else:
                    full = True
                    msg = "middle"
                cut.append((diff, full, msg))
                first = False
                st = float(scissors)
            if end != inbetween[-1]:
                cut.append((end - inbetween[-1],
                           False, "last"))
            distribution = convert_cut2distribution(cut)
            distribution = filter(lambda n: n != 0, distribution)
            distribution = list(distribution)
        else:
            distribution = [duration]
        self.current_rhythm_add(duration)
        return distribution


class LineConverter(object):
    def __call__(self, line):
        period = line.period
        line = line.split()
        line = line.tie_pauses()
        assert(period is not None)
        period = Period(*period)
        stream = music21.m21.stream.Stream()
        for event in line:
            try:
                pitch = tuple(event.pitch)
            except TypeError:
                pitch = (event.pitch,)
            duration = float(event.duration)
            durations = LineConverter.distribute_duration(
                duration, period)
            durations_len = len(durations)
            volume = event.volume
            if volume is not None:
                volume_m21 = music21.m21.dynamics.dynamicStrFromDecimal(volume)
                volume_m21 = music21.m21.dynamics.Dynamic(
                        volume_m21)
                stream.append(volume_m21)
            for i, d in enumerate(durations):
                d = float(d)
                duration_m21 = music21.m21.duration.Duration(d)
                contain_pitches = any(map(lambda p: p != mel.EmptyPitch(),
                                        pitch))
                if contain_pitches is True:
                    pitches = tuple(p.convert2music21() for p in pitch
                                  if p != mel.EmptyPitch())
                    item = music21.m21.chord.Chord(pitches,
                            duration=duration_m21)
                    if durations_len > 1:
                        if i == 0:
                            tie_msg = "start"
                        elif i + 1 == durations_len:
                            tie_msg = "stop"
                        else:
                            tie_msg = "continue"
                        item.tie = music21.m21.tie.Tie(tie_msg)
                else:
                    item = music21.m21.note.Rest(duration=duration_m21)
                stream.append(item)
        return stream


    @staticmethod
    def distribute_duration(duration, period) -> list:
        if duration > 0:
            available_rest = period.summed_rest()
            if duration > available_rest:
                rest_duration = duration - available_rest
                duration = available_rest
            else:
                rest_duration = 0
            distributed = period.apply_rhythm(duration)
            return distributed + LineConverter.distribute_duration(
                rest_duration, period)
        else:
            return []

