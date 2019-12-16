import functools
import operator

from mu.mel import ji
from mu.mel import mel


"""Module for JI-Pitch shortwriting.

Pitch notation: 7+   ->  7/4
                3+7+ -> 21/16
                5-   -> 8/5
                3+5- -> 6/5

Octave notation: 7+. -> 7/2
                 .7+ -> 7/8

Chord notation: (7+ 1+) -> (7/4, 1/1)

Pause notation X -> mel.TheEmptyPitch
         or    x -> mel.TheEmptyPitch

Setting standard exponent:  !+ 3 -> 3/2
                            !- 3 -> 4/3
Writing comment: # This is a comment
        (only for translate_from_file valid)

Example:

    # This is an example
    !+ 3 1 5 3 (7 1.) 5+3- 3- (5 1) x (7- .7--) 1
"""


def translate2pitch(
    info: str, standard=1, idx=None, decodex: dict = None, inverse=False
) -> ji.JIPitch:
    def change2pitch(current_num, current_exp) -> tuple:
        if current_num:
            number = int(current_num)
        else:
            msg = "NUMBER FORGOTTEN IN ELEMENT {0}".format(info)
            if idx:
                msg += " ({0} element)".format(idx)
            raise ValueError(msg)
        if current_exp:
            exp = sum(current_exp)
        else:
            exp = standard
        if exp > 0:
            ret = True
        else:
            ret = False
        return number ** abs(exp), ret

    splited_by_octave_remarks = info.split(".")
    octave = 0
    before = True
    pitch = None
    for item in splited_by_octave_remarks:
        if item:
            if pitch:
                msg = "UNEXPECTED FORM: '.' in between {0}".format(info)
                raise ValueError(msg)
            else:
                pitch = item
            before = False
        else:
            if before:
                octave -= 1
            else:
                octave += 1

    numbers = tuple(str(i) for i in range(10))
    positive, negative = [[1], [1]]
    is_seperating = False
    current_num = ""
    current_exp = []
    if decodex:
        pitch = decodex[pitch]
    else:
        for element in pitch:
            if element in numbers:
                if is_seperating:
                    fac, pos = change2pitch(current_num, current_exp)
                    if pos:
                        positive.append(fac)
                    else:
                        negative.append(fac)
                    current_num = element
                    current_exp = []
                    is_seperating = False
                else:
                    current_num += element
            elif element == "+":
                is_seperating = True
                current_exp.append(1)
            elif element == "-":
                is_seperating = True
                current_exp.append(-1)
            else:
                msg = "UNKNOWN SIGN {0} IN {1}".format(element, pitch)
                if idx:
                    msg += " ({0} element)".format(idx)
                raise ValueError(msg)

        fac, pos = change2pitch(current_num, current_exp)
        if pos:
            positive.append(fac)
        else:
            negative.append(fac)

        pos_and_neg = (positive, negative)
        if inverse:
            pos_and_neg = reversed(pos_and_neg)
        pitch = ji.r(
            *tuple(functools.reduce(operator.mul, n) for n in pos_and_neg)
        ).normalize()

    if octave > 0:
        ocp = ji.r(2 ** octave, 1)
    else:
        ocp = ji.r(1, 2 ** abs(octave))

    typp = type(pitch)
    if typp == ji.JIPitch or typp == mel.SimplePitch:
        pitch = pitch + ocp

    else:
        msg = "Unknown pitch type {0}".format(typp)
        raise TypeError(msg)

    return pitch


def translate2line(information: str, standard=1, decodex=None, inverse=False) -> tuple:
    divided = tuple(info for info in information.split(" ") if info)
    pitches = []
    for idx, info in enumerate(divided):
        if info[0] == "!":
            if info[1] == "+":
                standard = 1
            elif info[1] == "-":
                standard = -1
            else:
                msg = "UNKNOWN SYMBOL {0} IN COMMAND {1} ({2}).".format(
                    info[1], info, idx
                )
                raise ValueError(msg)
        elif info[0].upper() == "X":
            pitches.append(mel.TheEmptyPitch)
        else:
            pitches.append(translate2pitch(info, standard, idx, decodex, inverse))
    return tuple(pitches), standard


def translate(
    information: str, allow_chords=True, decodex: dict = None, inverse: bool = False
) -> mel.Cadence:
    if allow_chords:
        not_closed_msg = "Paranthesis not closed for one chord in {0}".format(
            information
        )
        not_opened_msg = "Paranthesis not opened for one chord in {0}".format(
            information
        )
        id_line = "line"
        id_chord = "chord"
        lines_and_chords = []
        is_chord = False
        current_line = ""
        for character in information:
            if character == "(":
                if is_chord:
                    raise ValueError(not_closed_msg)
                if current_line:
                    lines_and_chords.append((id_line, str(current_line)))
                    current_line = ""
                is_chord = True
            elif character == ")":
                if is_chord:
                    lines_and_chords.append((id_chord, str(current_line)))
                    is_chord = False
                    current_line = ""
                else:
                    raise ValueError(not_opened_msg)
            else:
                current_line += character

        if is_chord:
            raise ValueError(not_closed_msg)
        else:
            if current_line:
                lines_and_chords.append((id_line, str(current_line)))

        cadence = []
        standard = 1
        for identity, line in lines_and_chords:
            translation, standard = translate2line(
                line, standard, decodex=decodex, inverse=inverse
            )
            if identity == id_line:
                for pitch in translation:
                    if pitch != mel.TheEmptyPitch:
                        cadence.append(mel.Harmony([pitch]))
                    else:
                        cadence.append(mel.Harmony([]))
            elif identity == id_chord:
                cadence.append(
                    mel.Harmony(
                        tuple(p for p in translation if p != mel.TheEmptyPitch)
                    )
                )
            else:
                raise ValueError("UNKNOWN IDENTITY {0}".format(identity))

        return mel.Cadence(cadence)
    else:
        return translate2line(information, decodex=decodex, inverse=inverse)[0]


def translate_from_file(path: str, decodex: dict = None) -> tuple:
    with open(path, "r") as content:
        lines = content.read().splitlines()
        # deleting comments
        lines = tuple(l for l in lines if l and l[0] != "#")
        content = " ".join(lines)
        return translate(content, decodex=decodex)
