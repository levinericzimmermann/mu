"""This module implements activity-levels in Python.

They are copied from M. Edwards activity-levels concepts implemented in
his Common Lisp software 'slippery chicken'. For more information see:
    https://michael-edwards.org/sc/
"""

import functools
import itertools
import operator

from mutools import schillinger


class ActivityLevel(object):
    """Activity Levels is a concept derived from M. Edwards.

    This is a Python implementation of the Activity Levels.
    Quoting Michael Edwards Activity Levels are an "object for determining
    (deterministically) on a call-by-call basis whether a process is active
    or not (boolean).  This is determined by nine 10-element lists
    (actually three versions of each) of hand-coded 1s and 0s, each list
    representing an 'activity-level' (how active the process should be).
    The first three 10-element lists have only one 1 in them, the rest being zeros.
    The second three have two 1s, etc. Activity-levels of 0 and 10 would return
    never active and always active respectively."
    (see https://michael-edwards.org/sc/robodoc/activity-levels_lsp.html#robo23)
    """

    # tuples copied from
    # github.com/mdedwards/slippery-chicken/blob/master/activity-levels.lsp
    __activity_levels = (
        # 0
        ((0,),),
        # 1
        (
            (1, 0, 0, 0, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 1, 0, 0, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 0, 0, 0),
        ),
        # 2
        (
            (1, 0, 0, 0, 0, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 0, 0, 0, 0),
            (0, 0, 0, 0, 0, 0, 1, 1, 0, 0),
        ),
        # 3
        (
            (1, 0, 0, 0, 1, 0, 1, 0, 0, 0),
            (0, 0, 0, 1, 0, 1, 1, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 1, 1, 0, 0),
        ),
        # 4
        (
            (1, 0, 0, 0, 1, 0, 1, 1, 0, 0),
            (0, 1, 0, 1, 0, 1, 1, 0, 0, 0),
            (0, 0, 1, 0, 0, 0, 1, 1, 0, 1),
        ),
        # 5
        (
            (1, 1, 0, 0, 1, 0, 1, 1, 0, 0),
            (0, 1, 0, 1, 0, 1, 1, 0, 0, 1),
            (0, 0, 1, 0, 1, 0, 1, 1, 0, 1),
        ),
        # 6
        (
            (1, 1, 0, 1, 1, 0, 1, 1, 0, 0),
            (0, 1, 0, 1, 0, 1, 1, 0, 1, 1),
            (0, 1, 1, 0, 1, 0, 1, 1, 0, 1),
        ),
        # 7
        (
            (1, 1, 0, 1, 1, 0, 1, 1, 0, 1),
            (1, 1, 0, 1, 0, 1, 1, 0, 1, 1),
            (1, 1, 1, 0, 1, 0, 1, 1, 0, 1),
        ),
        # 8
        (
            (1, 1, 0, 1, 1, 1, 1, 1, 0, 1),
            (1, 1, 1, 1, 0, 1, 1, 0, 1, 1),
            (1, 1, 1, 0, 1, 1, 1, 1, 0, 1),
        ),
        # 9
        (
            (1, 1, 0, 1, 1, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 0, 1, 1, 1, 1, 1),
            (1, 1, 1, 1, 1, 1, 1, 1, 0, 1),
        ),
        # 10
        ((1,),),
    )

    __allowed_range = tuple(range(11))

    def __init__(self, start_at: int = 0, is_inverse: bool = False) -> None:
        try:
            assert start_at in (0, 1, 2)
        except AssertionError:
            msg = "start_at has to be either 0, 1 or 2 and not {}, ".format(start_at)
            msg += "because there are only three different tuples defined per level."
            raise ValueError(msg)

        # inversing index 1 and 2, because the schillinger.permute_cyclic
        # function also returns an inversed result.
        start_at = (0, 2, 1)[start_at]

        self.__activity_levels = tuple(
            itertools.cycle(
                functools.reduce(
                    operator.add, schillinger.permute_cyclic(levels)[start_at]
                )
            )
            for levels in self.__activity_levels
        )
        self.__is_inverse = is_inverse

    @property
    def is_inverse(self) -> bool:
        return self.__is_inverse

    def __repr__(self) -> str:
        return "ActivityLevel({})".format(self.is_inverse)

    def __call__(self, lv: int) -> bool:
        try:
            assert lv in self.__allowed_range
        except AssertionError:
            msg = "lv is {} but has to be in range {}!".format(lv, self.__allowed_range)
            raise ValueError(msg)

        val = bool(next(self.__activity_levels[lv]))

        if self.is_inverse:
            return not val
        else:
            return val
