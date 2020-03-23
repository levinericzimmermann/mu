from mu.mel.abstract import AbstractPitch
from mu.mel import mel
from mu.rhy import rhy
from mu.sco import old

import abc
import bisect
import functools
import itertools
import operator
import os
import subprocess
from typing import Optional

import mido

__directory = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(__directory, "", "../mel/12edo"), "r") as f:
    _12edo_freq = tuple(float(line[:-1]) for line in f.readlines())


class MidiTone(old.Tone):
    _init_args = {}

    def __init__(
        self,
        pitch: Optional[AbstractPitch],
        delay: rhy.Unit,
        duration: Optional[rhy.Unit] = None,
        volume: Optional = None,
        glissando: old.GlissandoLine = None,
        vibrato: old.VibratoLine = None,
        tuning: tuple = tuple([]),
    ) -> None:
        old.Tone.__init__(
            self,
            pitch,
            delay,
            duration,
            volume=volume,
            glissando=glissando,
            vibrato=vibrato,
        )
        if tuning:
            self.tuning = tuple(tuning)
        else:
            self.tuning = tuple([])

    def control_messages(self, channel: int) -> tuple:
        """Generate control messages. Depending on specific channel."""
        messages = []
        for arg in self._init_args:
            value = getattr(self, arg)
            if value is not None:
                boundaries, control_number = self._init_args[arg]
                difference = boundaries[1] - boundaries[0]
                normalized = value - boundaries[0]
                percent = normalized / difference
                value = int(127 * percent)
                message = mido.Message(
                    "control_change",
                    time=0,
                    control=control_number,
                    value=value,
                    channel=channel,
                )
                messages.append(message)
        return tuple(messages)


class SynthesizerMidiTone(abc.ABCMeta):
    tone_args = (
        "pitch",
        "delay",
        "duration",
        "volume",
        "glissando",
        "vibrato",
        "tuning",
    )

    def __new__(cls, name, bases, attrs):
        def auto_init(self, *args, **kwargs):
            arg_names = cls.tone_args + tuple(self._init_args.keys())
            length_tone_args = len(cls.tone_args)
            length_args = len(args)
            for counter, arg_val, arg_name in zip(range(length_args), args, arg_names):
                if counter > length_tone_args:
                    tolerance = self.__init_args[arg_name][0]
                    try:
                        assert arg_val <= tolerance[0]
                        assert arg_val >= tolerance[1]
                    except AssertionError:
                        msg = "The value for '{0}' has to be >= {1} and <= {2}.".format(
                            arg_name, tolerance[0], tolerance[1]
                        )
                        raise ValueError(msg)
                setattr(self, arg_name, arg_val)

            for arg_name in arg_names[length_args:]:
                if arg_name not in kwargs.keys():
                    kwargs.update({arg_name: None})

            self.__dict__.update(kwargs)

            MidiTone.__init__(
                self,
                self.pitch,
                self.delay,
                self.duration,
                self.volume,
                self.glissando,
                self.vibrato,
                self.tuning,
            )

        attrs["__init__"] = auto_init
        return super(SynthesizerMidiTone, cls).__new__(cls, name, bases, attrs)


class PyteqTone(MidiTone, metaclass=SynthesizerMidiTone):
    """Tone object to work with Pianoteq"""

    _init_args = {
        "unison_width": ((0, 20), 2),
        "hammer_noise": ((0.20, 3), 3),
        "diapason": ((220, 880), 5),
        "octave_stretching": ((0.95, 3), 6),
        "unison_balance": ((-1, 1), 7),
        "direct_sound_duration": ((0, 5), 8),
        "hammer_hard_piano": ((0, 2), 9),
        "spectrum_profile_1": ((-15, 15), 10),
        "spectrum_profile_2": ((-15, 15), 11),
        "spectrum_profile_3": ((-15, 15), 12),
        "spectrum_profile_4": ((-15, 15), 13),
        "spectrum_profile_5": ((-15, 15), 14),
        "spectrum_profile_6": ((-15, 15), 15),
        "spectrum_profile_7": ((-15, 15), 16),
        "spectrum_profile_8": ((-15, 15), 17),
        "strike_point": ((1 / 64, 1 / 2), 18),
        "pinch_harmonic_point": ((1 / 64, 1 / 2), 19),
        "pickup_symmetry": ((0, 1), 20),
        "pickup_distance": ((0.20, 2), 21),
        "soft_level": ((0, 1), 22),
        "impedance": ((0.3, 3), 23),
        "cutoff": ((0.3, 3), 24),
        "q_factor": ((0.2, 5), 25),
        "string_length": ((0.8, 10), 26),
        "sympathetic_resonance": ((0, 5), 27),
        "duplex_scale_resonance": ((0, 20), 28),
        "quadratic_effect": ((0, 20), 29),
        "damper_noise": ((-85, 12), 30),
        "damper_position": ((1 / 64, 1 / 2), 31),
        "last_damper": ((0, 128), 32),
        "pedal_noise": ((-70, 25), 33),
        "key_release_noise": ((-70, 25), 34),
        "damping_duration": ((0.03, 10), 35),
        "mute": ((0, 1), 36),
        "clavinet_low_mic": ((-1, 1), 37),
        "clavinet_high_mic": ((-1, 1), 38),
        "equalizer_switch": ((0, 1), 39),
        "hammer_tine_nose": ((-100, 25), 40),
        "blooming_energy": ((0, 2), 41),
        "blooming_inertia": ((0.10, 3), 42),
        "aftertouch": ((0, 1), 43),
        "post_effect_gain": ((-12, 12), 44),
        "bounce_switch": ((0, 1), 45),
        "bounce_delay": ((10, 250), 46),
        "bounce_sync": ((0, 1), 47),
        "bounce_sync_speed": ((16, 1 / 8), 48),
        "bounce_velocity_speed": ((0, 100), 49),
        "bounce_delay_loss": ((0, 100), 50),
        "bounce_velocity_loss": ((0, 100), 51),
        "bounce_humanization": ((0, 100), 52),
        "sustain_pedal": ((0, 1), 53),
        "soft_pedal": ((0, 1), 54),
        "sostenuto_pedal": ((0, 1), 55),
        "harmonic_pedal": ((0, 1), 56),
        "rattle_pedal": ((0, 1), 57),
        "buff_stop_pedal": ((0, 1), 58),
        "celeste_pedal": ((0, 1), 59),
        "super_sostenuto": ((0, 1), 60),
        "pinch_harmonic_pedal": ((0, 1), 61),
        "glissando_pedal": ((0, 1), 62),
        "harpsichord_register_1": ((0, 1), 63),
        "harpsichord_register_2": ((0, 1), 64),
        "harpsichord_register_3": ((0, 1), 65),
        "reversed_sustain": ((0, 1), 66),
        "mozart_rail": ((0, 1), 67),
        "pedal_1": ((0, 1), 68),
        "pedal_2": ((0, 1), 69),
        "pedal_3": ((0, 1), 70),
        "pedal_4": ((0, 1), 71),
        "stereo_width": ((0, 5), 72),
        "lid_position": ((0, 1), 73),
        "output_mode": ((0, 3), 74),
        "mic_level_compensation": ((0, 1), 75),
        "mic_delay_compensation": ((0, 1), 76),
        "head_x_position": ((-10, 10), 77),
        "head_y_position": ((-6, 6), 78),
        "head_z_position": ((0, 3.5), 79),
        "head_diameter": ((10, 50), 80),
        "head_angle": ((-180, 180), 81),
        "mic_1_mic_switch": ((0, 1), 82),
        "mic_1_x_position": ((-10, 10), 83),
        "mic_1_y_position": ((-6, 6), 84),
        "mic_1_z_position": ((0, 3.5), 85),
        "mic_1_azimuth": ((-180, 180), 86),
        "mic_1_elevation": ((-180, 180), 87),
        "mic_1_level_1": ((-85, 6), 90),
        "reverb_switch": ((0, 1), 91),
        "sound_speed": ((200, 500), 92),
        "wall_distance": ((0, 6), 93),
        "hammer_hard_mezzo": ((0, 2), 94),
        "hammer_hard_forte": ((0, 2), 95),
    }


class DivaTone(MidiTone, metaclass=SynthesizerMidiTone):
    """Tone object to work with U-He Diva."""

    # control value 2 in voice 1 is already used for fine_tune_cents
    # _init_args = {"fine_tune_cents": ((-100, 100), 2)}

    def control_messages(self, channel: int, midi_key: int) -> tuple:
        messages = super(DivaTone, self).control_messages(channel)
        cent_deviation = AbstractPitch.ratio2ct(self.pitch.freq / _12edo_freq[midi_key])
        assert cent_deviation > -100 and cent_deviation < 100
        percentage = (cent_deviation + 100) / 200
        value = int(percentage * 127)
        messages += tuple(
            [mido.Message("control_change", time=0, control=2, value=value, channel=0)]
        )
        return messages


class MidiFile(abc.ABC):
    maximum_cent_deviation = 1200
    maximum_pitch_bending = 16382
    maximum_pitch_bending_positive = 8191

    # somehow control messages take some time until they
    # became valid in Pianoteq. Therefore there has to be
    # a delay between the last control change and the next
    # NoteOn - message.
    delay_between_control_messages_and_note_on_message = 40

    # for some weird reason pianoteq don't use channel 9
    available_channel = tuple(i for i in range(16) if i != 9)

    def __init__(
        self, sequence: tuple, available_midi_notes: tuple = tuple(range(128))
    ):
        self.__available_midi_notes = available_midi_notes
        sequence = MidiFile.discard_pauses_and_tie_sequence(tuple(sequence))
        filtered_sequence = tuple(t for t in sequence if t.pitch != mel.TheEmptyPitch)
        gridsize = 0.001  # 1 milisecond
        self.__duration = float(sum(t.delay for t in sequence))
        n_hits = int(self.__duration // gridsize)
        n_hits += self.delay_between_control_messages_and_note_on_message + 2
        self.__grid = tuple(i * gridsize for i in range(0, n_hits))
        self.__gridsize = gridsize
        self.__grid_position_per_tone = MidiFile.detect_grid_position(
            sequence, self.__grid, self.__duration
        )
        self.__amount_available_midi_notes = len(available_midi_notes)
        self.__sequence = sequence
        self.__overlapping_dict = MidiFile.mk_overlapping_dict(filtered_sequence)
        self.__midi_keys_dict = MidiFile.mk_midi_key_dictionary(
            set(t.pitch for t in filtered_sequence),
            available_midi_notes,
            self.__amount_available_midi_notes,
        )
        self.keys = MidiFile.distribute_tones_on_midi_keys(
            filtered_sequence,
            self.__amount_available_midi_notes,
            available_midi_notes,
            self.__overlapping_dict,
            self.__midi_keys_dict,
        )
        pitch_data = MidiFile.mk_pitch_sequence(filtered_sequence)
        self.__pitch_sequence = pitch_data[0]
        self.__tuning_sequence = pitch_data[1]
        self.__midi_pitch_dictionary = pitch_data[2]
        self.__control_messages = self.mk_control_messages_per_tone(filtered_sequence)
        self.__note_on_off_messages = MidiFile.mk_note_on_off_messages(
            filtered_sequence, self.keys
        )
        self.__pitch_bending_per_tone = MidiFile.detect_pitch_bending_per_tone(
            filtered_sequence, self.__gridsize, self.__grid_position_per_tone
        )
        self.__pitch_bending_per_channel = self.distribute_pitch_bends_on_channels(
            self.__pitch_bending_per_tone,
            self.__grid,
            self.__grid_position_per_tone,
            self.__gridsize,
        )
        self.__filtered_sequence = filtered_sequence

    @abc.abstractmethod
    def mk_tuning_messages(
        self, sequence, keys, available_midi_notes, overlapping_dict, midi_pitch_dict
    ) -> tuple:
        raise NotImplementedError

    @staticmethod
    def discard_pauses_and_tie_sequence(sequence):
        """this will change the init sequence!!!"""
        new = []
        first = True
        for event in sequence:
            if first is False and event.pitch == mel.TheEmptyPitch:
                information = event.delay
                new[-1].duration += information
                new[-1].delay += information
            else:
                new.append(event)
            first = False
        return tuple(new)

    def distribute_pitch_bends_on_channels(
        self, pitch_bends_per_tone, grid, grid_position_per_tone, gridsize
    ) -> tuple:
        channels = itertools.cycle(range(len(MidiFile.available_channel)))
        pitches_per_channels = list(
            list(0 for j in range(len(grid))) for i in MidiFile.available_channel
        )
        for position, pitch_bends in zip(grid_position_per_tone, pitch_bends_per_tone):
            channel = next(channels)
            start = (
                position[0] + self.delay_between_control_messages_and_note_on_message
            )
            end = position[1] + self.delay_between_control_messages_and_note_on_message
            pitches_per_channels[channel][start:end] = pitch_bends

        # transform to pitch_bending midi - messages
        first = True
        pitch_bending_messages = []
        total_range = MidiFile.maximum_cent_deviation * 2
        # total_range = MidiFile.maximum_cent_deviation * 1
        warn = Warning(
            "Maximum pitch bending is {0} cents up or down!".format(
                MidiFile.maximum_pitch_bending
            )
        )
        standardmessage0 = tuple(
            mido.Message("pitchwheel", channel=channel_number, pitch=0, time=0)
            for channel_number in MidiFile.available_channel
        )
        standardmessage1 = tuple(
            mido.Message("pitchwheel", channel=channel_number, pitch=0, time=1)
            for channel_number in MidiFile.available_channel
        )
        for channel_number, channel in zip(
            reversed(MidiFile.available_channel), reversed(pitches_per_channels)
        ):
            pitch_bending_messages_sub_channel = []
            for cent_deviation in channel:
                if first is True:
                    time = 1
                else:
                    time = 0
                if cent_deviation != 0:
                    pitch_percent = (
                        cent_deviation + MidiFile.maximum_cent_deviation
                    ) / total_range
                    if pitch_percent > 1:
                        pitch_percent = 1
                        raise warn
                    if pitch_percent < 0:
                        pitch_percent = 0
                        raise warn
                    midi_pitch = int(MidiFile.maximum_pitch_bending * pitch_percent)
                    midi_pitch -= MidiFile.maximum_pitch_bending_positive
                    msg = mido.Message(
                        "pitchwheel",
                        channel=channel_number,
                        pitch=midi_pitch,
                        time=time,
                    )
                else:
                    if time == 0:
                        msg = standardmessage0[
                            MidiFile.available_channel.index(channel_number)
                        ]
                    else:
                        msg = standardmessage1[
                            MidiFile.available_channel.index(channel_number)
                        ]
                pitch_bending_messages_sub_channel.append(msg)
            pitch_bending_messages.append(pitch_bending_messages_sub_channel)
            first = False
        pitch_bending_messages = tuple(reversed(pitch_bending_messages))
        return pitch_bending_messages

    @staticmethod
    def detect_pitch_bending_per_tone(
        sequence, gridsize, grid_position_per_tone
    ) -> tuple:
        """Return tuple filled with tuples that contain cent deviation per step."""

        def mk_interpolation(obj, size):
            if obj:
                obj = list(obj.interpolate(gridsize))
            else:
                obj = []
            while len(obj) > size:
                obj = obj[:-1]
            while len(obj) < size:
                obj.append(0)
            return obj

        pitch_bending = []
        for tone, start_end in zip(sequence, grid_position_per_tone):
            size = start_end[1] - start_end[0]
            glissando = mk_interpolation(tone.glissando, size)
            vibrato = mk_interpolation(tone.vibrato, size)
            resulting_cents = tuple(a + b for a, b in zip(glissando, vibrato))
            pitch_bending.append(resulting_cents)
        return tuple(pitch_bending)

    @staticmethod
    def mk_note_on_off_messages(sequence, keys) -> tuple:
        """Generate Note on / off messages for every tone.

        Resulting tuple has the form:
        ((note_on0, note_off0), (note_on1, note_off1), ...)
        """
        assert len(sequence) == len(keys)
        channels = itertools.cycle(MidiFile.available_channel)
        messages = []
        for tone, key in zip(sequence, keys):
            if tone.pitch != mel.TheEmptyPitch:
                if tone.volume:
                    velocity = int((tone.volume / 1) * 127)
                else:
                    velocity = 64
                chnl = next(channels)
                msg0 = mido.Message(
                    "note_on", note=key, velocity=velocity, time=0, channel=chnl
                )
                msg1 = mido.Message(
                    "note_off", note=key, velocity=velocity, time=0, channel=chnl
                )
                messages.append((msg0, msg1))
        return tuple(messages)

    @staticmethod
    def detect_grid_position(sequence, grid, duration):
        def find_closest_point(points, time):
            pos = bisect.bisect_right(points, time)
            try:
                return min(
                    (
                        (abs(time - points[pos]), pos),
                        (abs(time - points[pos - 1]), pos - 1),
                    ),
                    key=operator.itemgetter(0),
                )[1]
            except IndexError:
                # if pos is len(points) + 1
                return pos - 1

        delays = tuple(float(tone.delay) for tone in sequence)
        starts = tuple(itertools.accumulate((0,) + delays))[:-1]
        durations = tuple(float(tone.duration) for tone in sequence)
        endings = tuple(s + d for s, d in zip(starts, durations))
        start_points = tuple(find_closest_point(grid, s) for s in starts)
        end_points = tuple(find_closest_point(grid, e) for e in endings)
        zipped = tuple(zip(start_points, end_points))
        return tuple(
            start_end
            for start_end, tone in zip(zipped, sequence)
            if tone.pitch != mel.TheEmptyPitch
        )

    @property
    def sequence(self) -> tuple:
        return tuple(self.__sequence)

    def mk_control_messages_per_tone(self, sequence) -> tuple:
        channels = itertools.cycle(MidiFile.available_channel)
        return tuple(tone.control_messages(next(channels)) for tone in sequence)

    @staticmethod
    def mk_midi_track(messages) -> mido.MidiFile:
        mid = mido.MidiFile(type=0)
        bpm = 120
        ticks_per_second = 1000
        ticks_per_minute = ticks_per_second * 60
        ticks_per_beat = int(ticks_per_minute / bpm)
        mid.ticks_per_beat = ticks_per_beat
        track = mido.MidiTrack()
        mid.tracks.append(track)
        track.append(mido.MetaMessage("instrument_name", name="Acoustic Grand Piano"))
        for i in MidiFile.available_channel:
            track.append(mido.Message("program_change", program=0, time=0, channel=i))
        for message in messages:
            track.append(message)
        return mid

    @staticmethod
    def mk_overlapping_dict(sequence) -> dict:
        delays = tuple(float(t.delay) for t in sequence)
        absolute_delays = tuple(
            b - a for a, b in zip(delays, delays[1:] + (sum(delays),))
        )
        overlapping_dict = {i: [] for i, t in enumerate(sequence)}
        for i, tone in enumerate(sequence):
            if tone.delay < tone.duration:
                ending = delays[i] + tone.duration
                for j, abs_delay in enumerate(absolute_delays[i + 1 :]):
                    if abs_delay < ending:
                        overlapping_dict[i + j + 1].append(i)
                    else:
                        break
        return overlapping_dict

    @staticmethod
    def distribute_tones_on_midi_keys(
        sequence,
        amount_available_midi_notes,
        available_midi_notes,
        overlapping_dict,
        midi_keys_dict,
    ) -> tuple:
        def convert_keys(keys) -> tuple:
            return tuple(midi_keys_dict[t.pitch][key] for t, key in zip(sequence, keys))

        def is_alright(keys, overlapping_dict) -> bool:
            converted_keys = convert_keys(keys)
            for tone in tuple(overlapping_dict.keys())[: len(keys)]:
                simultan_tones = overlapping_dict[tone]
                current_keys = tuple(
                    converted_keys[idx] for idx in simultan_tones + [tone]
                )
                if len(current_keys) - len(set(current_keys)) != 0:
                    return False
            return True

        keys = [0]
        amount_tones = len(sequence)
        while len(keys) < amount_tones:
            if is_alright(keys, overlapping_dict) is True:
                keys.append(0)
            else:
                while keys[-1] + 1 == amount_available_midi_notes:
                    keys = keys[:-1]
                    if len(keys) == 0:
                        raise ValueError("No solution found! Too many simultan tones.")
                keys[-1] += 1
        converted_keys = convert_keys(keys)
        return tuple(available_midi_notes[key] for key in converted_keys)

    @staticmethod
    def mk_pitch_sequence(sequence) -> "pitch_sequence, tuning_sequence, midi_dict":
        pitch_sequence = tuple(t.pitch for t in sequence)
        tuning_sequence = tuple(t.tuning for t in sequence)
        pitches = set(functools.reduce(operator.add, tuning_sequence) + pitch_sequence)
        midi_dict = MidiFile.mk_midi_pitch_dictionary(pitches)
        return pitch_sequence, tuning_sequence, midi_dict

    @staticmethod
    def mk_midi_pitch_dictionary(pitches: set) -> dict:
        return {
            pitch: pitch.convert2midi_tuning()
            for pitch in pitches
            if pitch != mel.TheEmptyPitch
        }

    @staticmethod
    def mk_midi_key_dictionary(
        pitches: set, available_midi_notes, amount_available_midi_notes
    ) -> dict:
        def evaluate_rating(pitch):
            freq = pitch.freq
            closest = bisect.bisect_right(available_frequencies, freq) - 1
            higher = tuple(range(closest + 1, amount_available_midi_notes))
            lower = tuple(range(closest - 1, -1, -1))
            ranking = (closest,) + tuple(
                functools.reduce(operator.add, zip(higher, lower))
            )
            len_h, len_l = len(higher), len(lower)
            if len_h > len_l:
                ranking += higher[len_l:]
            else:
                ranking += lower[len_h:]
            return ranking

        available_frequencies = tuple(_12edo_freq[idx] for idx in available_midi_notes)
        return {pitch: evaluate_rating(pitch) for pitch in pitches}

    def mk_complete_messages(
        self,
        filtered_sequence,
        gridsize,
        grid_position_per_tone,
        control_messages,
        note_on_off_messages,
        pitch_bending_per_channel,
        tuning_messages,
    ) -> tuple:
        length_seq = len(filtered_sequence)
        assert length_seq == len(control_messages)
        assert length_seq == len(note_on_off_messages)
        assert length_seq == len(tuning_messages)
        messages_per_tick = list(zip(*reversed(pitch_bending_per_channel)))
        messages_per_tick = [list(s) for s in messages_per_tick]
        for note_on_off, control, tuning, grid_position in zip(
            note_on_off_messages,
            control_messages,
            tuning_messages,
            grid_position_per_tone,
        ):
            note_on, note_off = note_on_off
            start, stop = grid_position
            messages_per_tick[
                start + self.delay_between_control_messages_and_note_on_message
            ].append(note_on)
            messages_per_tick[start].extend(tuning)
            messages_per_tick[start].extend(control)
            messages_per_tick[
                stop + self.delay_between_control_messages_and_note_on_message
            ].append(note_off)
        messages_per_tick = tuple(tuple(reversed(tick)) for tick in messages_per_tick)
        return tuple(item for sublist in messages_per_tick for item in sublist)

    @property
    def miditrack(self) -> mido.MidiFile:
        return self.__miditrack

    def export(self, name: str = "test.mid") -> None:
        """save content of object to midi-file."""

        tuning_messages = self.mk_tuning_messages(
            self.__filtered_sequence,
            self.keys,
            self.__available_midi_notes,
            self.__overlapping_dict,
            self.__midi_pitch_dictionary,
        )

        messages = self.mk_complete_messages(
            self.__filtered_sequence,
            self.__gridsize,
            self.__grid_position_per_tone,
            self.__control_messages,
            self.__note_on_off_messages,
            self.__pitch_bending_per_channel,
            tuning_messages,
        )

        miditrack = MidiFile.mk_midi_track(messages)
        miditrack.save(name)


class SysexTuningMidiFile(MidiFile):
    """MidiFile for synthesizer that understand Sysex tuning messages."""

    def mk_tuning_messages(
        self, sequence, keys, available_midi_notes, overlapping_dict, midi_pitch_dict
    ) -> tuple:
        def check_for_available_midi_notes(
            available_midi_notes, overlapping_dict, keys, tone_index
        ) -> tuple:
            """Return tuple with two elements:

            1. midi_number for playing tone
            2. remaining available midi numbers for retuning
            """
            played_key = keys[tone_index]
            busy_keys = tuple(keys[idx] for idx in tuple(overlapping_dict[tone_index]))
            busy_keys += (played_key,)
            remaining_keys = tuple(
                key for key in available_midi_notes if key not in busy_keys
            )
            return (played_key, remaining_keys)

        def mk_tuning_messages_for_tone(tone, local_midi_notes) -> tuple:
            if tone.pitch != mel.TheEmptyPitch:
                midi_pitch = midi_pitch_dict[tone.pitch]
                played_midi_note, remaining_midi_notes = local_midi_notes
                tuning = tone.tuning
                if not tuning:
                    tuning = (tone.pitch,)
                tuning = list(tuning)
                amount_remaining_midi_notes = len(remaining_midi_notes)
                tuning_gen = itertools.cycle(tuning)
                while len(tuning) < amount_remaining_midi_notes:
                    tuning.append(next(tuning_gen))
                while len(tuning) > amount_remaining_midi_notes:
                    tuning = tuning[:-1]
                tuning = sorted(tuning)
                midi_tuning = tuple(midi_pitch_dict[pitch] for pitch in tuning)
                key_tuning_pairs = tuple(zip(sorted(remaining_midi_notes), midi_tuning))
                key_tuning_pairs = ((played_midi_note, midi_pitch),) + key_tuning_pairs
                messages = []
                for key, tuning in key_tuning_pairs:
                    msg = mido.Message(
                        "sysex",
                        data=(
                            127,
                            127,
                            8,
                            2,
                            0,
                            1,
                            key,
                            tuning[0],
                            tuning[1],
                            tuning[2],
                        ),
                        time=0,
                    )
                    messages.append(msg)
                return tuple(messages)
            else:
                return tuple([])

        available_midi_notes_per_tone = tuple(
            check_for_available_midi_notes(
                available_midi_notes, overlapping_dict, keys, i
            )
            for i in range(len(sequence))
        )
        tuning_messages_per_tone = tuple(
            mk_tuning_messages_for_tone(tone, local_midi_notes)
            for tone, local_midi_notes in zip(sequence, available_midi_notes_per_tone)
        )
        return tuning_messages_per_tone


class NonSysexTuningMidiFile(MidiFile):
    """MidiFile for synthesizer that can't understand Sysex tuning messages."""

    def mk_tuning_messages(
        self, sequence, keys, available_midi_notes, overlapping_dict, midi_pitch_dict
    ) -> tuple:
        """Make empty tuning messages."""
        tuning_messages_per_tone = tuple(tuple([]) for tone in zip(sequence))
        return tuning_messages_per_tone


class Pianoteq(SysexTuningMidiFile):
    software_path = "pianoteq"

    def __init__(
        self, sequence: tuple, available_midi_notes: tuple = tuple(range(128))
    ):
        super(Pianoteq, self).__init__(sequence, available_midi_notes)

    def export2wav(
        self, name, nchnls=1, preset=None, fxp=None, sr=44100, verbose: bool = False
    ):
        self.export("{0}.mid".format(name))
        cmd = [
            "./{}".format(self.software_path),
            "--rate {}".format(sr),
            "--bit-depth 32",
            "--midimapping complete",
        ]

        if verbose is False:
            cmd.append("--quiet")

        if nchnls == 1:
            cmd.append("--mono")

        if preset is not None:
            cmd.append("--preset {}".format(preset))

        if fxp is not None:
            cmd.append("--fxp {} ".format(fxp))

        cmd.append("--midi {0}.mid --wav {0}.wav".format(name))
        subprocess.call(" ".join(cmd), shell=True)


class Diva(NonSysexTuningMidiFile):
    def __init__(self, sequence: tuple):
        super(Diva, self).__init__(sequence, tuple(range(128)))

    def mk_control_messages_per_tone(self, sequence) -> tuple:
        channels = itertools.cycle(MidiFile.available_channel)
        return tuple(
            tone.control_messages(next(channels), key)
            for tone, key in zip(sequence, self.keys)
        )
