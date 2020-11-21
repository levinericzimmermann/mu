"""This module provides several useful small functions."""

import bisect
import decimal
import functools
import itertools
import math
import operator
import os
import types

import numpy as np
from scipy.stats import norm

try:
    import quicktions as fractions
except ImportError:
    import fractions


def accumulate_from_n(iterator: tuple, n: float) -> tuple:
    return tuple(itertools.accumulate((n,) + (tuple(iterator))))


def accumulate_from_zero(iterator: tuple) -> tuple:
    return accumulate_from_n(iterator, 0)


def find_closest_index(item: float, data: tuple, key=None) -> int:
    """Return index of element in data with smallest difference to item"""

    if key is not None:
        research_data = tuple(map(key, data))

    else:
        research_data = tuple(data)

    solution = bisect.bisect_left(research_data, item)
    if solution == len(data):
        return solution - 1
    elif solution == 0:
        return solution
    else:
        indices = (solution, solution - 1)
        differences = tuple(abs(item - research_data[n]) for n in indices)
        return indices[differences.index(min(differences))]


def igmkdir(path: str) -> None:
    """mkdir that ignores FileExistsError."""
    try:
        os.mkdir(path)
    except FileExistsError:
        pass


def find_closest_item(item: float, data: tuple, key=None) -> float:
    """Return element in data with smallest difference to item"""
    return data[find_closest_index(item, data, key=key)]


def brownian(x0, n, dt, delta, out=None, random_state=None):
    """Generate an instance of Brownian motion (i.e. the Wiener process):

        X(t) = X(0) + N(0, delta**2 * t; 0, t)

    where N(a,b; t0, t1) is a normally distributed random variable with mean a and
    variance b.  The parameters t0 and t1 make explicit the statistical
    independence of N on different time intervals; that is, if [t0, t1) and
    [t2, t3) are disjoint intervals, then N(a, b; t0, t1) and N(a, b; t2, t3)
    are independent.

    Written as an iteration scheme,

        X(t + dt) = X(t) + N(0, delta**2 * dt; t, t+dt)


    If `x0` is an array (or array-like), each value in `x0` is treated as
    an initial condition, and the value returned is a numpy array with one
    more dimension than `x0`.

    Arguments
    ---------
    x0 : float or numpy array (or something that can be converted to a numpy array
         using numpy.asarray(x0)).
        The initial condition(s) (i.e. position(s)) of the Brownian motion.
    n : int
        The number of steps to take.
    dt : float
        The time step.
    delta : float
        delta determines the "speed" of the Brownian motion.  The random variable
        of the position at time t, X(t), has a normal distribution whose mean is
        the position at time t=0 and whose variance is delta**2*t.
    out : numpy array or None
        If `out` is not None, it specifies the array in which to put the
        result.  If `out` is None, a new numpy array is created and returned.

    Returns
    -------
    A numpy array of floats with shape `x0.shape + (n,)`.

    Note that the initial value `x0` is not included in the returned array.
    """

    x0 = np.asarray(x0)

    # For each element of x0, generate a sample of n numbers from a
    # normal distribution.
    r = norm.rvs(
        size=x0.shape + (n,), scale=delta * math.sqrt(dt), random_state=random_state
    )

    # If `out` was not given, create an output array.
    if out is None:
        out = np.empty(r.shape)

    # This computes the Brownian motion by forming the cumulative sum of
    # the random samples.
    np.cumsum(r, axis=-1, out=out)

    # Add the initial condition.
    out += np.expand_dims(x0, axis=-1)

    return out


def np_scale(a, minima=0, maxima=15) -> np.array:
    if len(a) > 0:
        return np.interp(a, (a.min(), a.max()), (minima, maxima))
    else:
        return np.array([])


def scale(iterable: tuple, minima: float = 0, maxima=1) -> tuple:
    return tuple(
        float(n) for n in np_scale(np.array(iterable), minima=minima, maxima=maxima)
    )


def graycode(length: int, modulus: int) -> tuple:
    """Returns the n-tuple reverse Gray code mod m.

    source: https://yetanothermathblog.com/tag/gray-codes/
    """
    n, m = length, modulus
    F = range(m)
    if n == 1:
        return [[i] for i in F]
    L = graycode(n - 1, m)
    M = []
    for j in F:
        M = M + [ll + [j] for ll in L]
    k = len(M)
    Mr = [0] * m
    for i in range(m - 1):
        i1 = i * int(k / m)
        i2 = (i + 1) * int(k / m)
        Mr[i] = M[i1:i2]
    Mr[m - 1] = M[(m - 1) * int(k / m) :]
    for i in range(m):
        if i % 2 != 0:
            Mr[i].reverse()
    M0 = []
    for i in range(m):
        M0 = M0 + Mr[i]
    return M0


def euclid(size: int, distribution: int) -> tuple:
    standard_size = size // distribution
    rest = size % distribution
    data = (standard_size for n in range(distribution))
    if rest:
        added = accumulate_from_zero(euclid(distribution, rest))
        return tuple(s + 1 if idx in added else s for idx, s in enumerate(data))
    else:
        return tuple(data)


def euclidic_interlocking(*iterable: tuple) -> tuple:
    lengths = tuple(len(it) for it in iterable)
    indices = tuple(0 for i in range(lengths[0]))

    for idx, length in enumerate(lengths[1:]):
        current_length = len(indices)
        indices_iter = iter(indices)
        current_idx = idx + 1
        indices = tuple(
            next(indices_iter) if distribution else current_idx
            for distribution in euclid(current_length, current_length + length)
        )

    iterables = [iter(it) for it in iterable]
    return tuple(next(iterables[idx]) for idx in indices)


def make_growing_series_with_sum_n(requested_sum: int) -> tuple:
    ls = []
    add_idx = iter([])
    while sum(ls) < requested_sum:
        try:
            ls[next(add_idx)] += 1
        except StopIteration:
            ls = [1] + ls
            add_idx = reversed(tuple(range(len(ls))))
    return tuple(ls)


def make_falling_series_with_sum_n(requested_sum: int) -> tuple:
    return tuple(reversed(make_growing_series_with_sum_n(requested_sum)))


def interlock_tuples(t0: tuple, t1: tuple) -> tuple:
    size0, size1 = len(t0), len(t1)
    difference = size0 - size1
    indices = functools.reduce(
        operator.add, ((0, 1) for n in range(min((size0, size1))))
    )
    if difference > 0:
        indices = tuple(0 for i in range(difference)) + indices
    else:
        indices = indices + tuple(1 for i in range(abs(difference)))
    t0_it = iter(t0)
    t1_it = iter(t1)
    return tuple(next(t0_it) if idx == 0 else next(t1_it) for idx in indices)


def not_fibonacci_transition(size0: int, size1: int, element0=0, element1=1) -> tuple:
    def write_to_n_element(it, element) -> tuple:
        return tuple(tuple(element for n in range(x)) for x in it)

    if size0 == 0 and size1 == 0:
        return tuple([])

    elif size0 == 0:
        return tuple([element1 for n in range(size1)])

    elif size1 == 0:
        return tuple([element0 for n in range(size0)])

    else:
        return functools.reduce(
            operator.add,
            interlock_tuples(
                *tuple(
                    write_to_n_element(s, el)
                    for s, el in zip(
                        (
                            make_falling_series_with_sum_n(size0),
                            make_growing_series_with_sum_n(size1),
                        ),
                        (element0, element1),
                    )
                )
            ),
        )


def gcd(*arg):
    return functools.reduce(math.gcd, arg)


def lcm(*arg: int) -> int:
    """from

    https://stackoverflow.com/questions/37237954/
    calculate-the-lcm-of-a-list-of-given-numbers-in-python
    """
    lcm = arg[0]
    for i in arg[1:]:
        lcm = lcm * i // gcd(lcm, i)
    return lcm


def cyclic_permutation(iterable: tuple) -> tuple:
    return (iterable[x:] + iterable[0:x] for x in range(len(iterable)))


def backtracking(elements: tuple, tests: tuple, return_indices: bool = False) -> tuple:
    """General backtracking algorithm function."""

    def convert_indices2elements(indices: tuple) -> tuple:
        current_elements = tuple(elements)
        resulting_elements = []
        for idx in indices:
            resulting_elements.append(current_elements[idx])
            current_elements = tuple(
                p for i, p in enumerate(current_elements) if i != idx
            )
        return tuple(resulting_elements)

    def is_valid(indices: tuple) -> bool:
        resulting_elements = convert_indices2elements(tuple(element_indices))
        return all(tuple(test(resulting_elements) for test in tests))

    amount_available_elements = len(elements)
    aapppppi = tuple(reversed(tuple(range(1, amount_available_elements + 1))))
    element_indices = [0]
    while True:
        if is_valid(tuple(element_indices)):
            if len(element_indices) < amount_available_elements:
                element_indices.append(0)
            else:
                break
        else:
            while element_indices[-1] + 1 == aapppppi[len(element_indices) - 1]:
                element_indices = element_indices[:-1]
                if len(element_indices) == 0:
                    raise ValueError("No solution found")
            element_indices[-1] += 1

    res = convert_indices2elements(element_indices)
    if return_indices:
        return res, element_indices
    else:
        return res


def complex_backtracking(
    elements_per_item: tuple, tests: tuple, return_indices: bool = False
) -> tuple:
    """Backtracking algorithm function where each item has a different set of elements."""

    def convert_indices2elements(indices: tuple) -> tuple:
        resulting_elements = []
        for idx, elements in zip(indices, elements_per_item):
            resulting_elements.append(elements[idx])
        return tuple(resulting_elements)

    def is_valid(indices: tuple) -> bool:
        resulting_elements = convert_indices2elements(tuple(element_indices))
        return all(tuple(test(resulting_elements) for test in tests))

    amount_available_elements_per_item = tuple(
        len(elements) for elements in elements_per_item
    )
    amount_available_items = len(elements_per_item)
    element_indices = [0]
    while True:
        if is_valid(tuple(element_indices)):
            if len(element_indices) < amount_available_items:
                element_indices.append(0)
            else:
                break
        else:
            while (
                element_indices[-1] + 1
                == amount_available_elements_per_item[len(element_indices) - 1]
            ):
                element_indices = element_indices[:-1]
                if len(element_indices) == 0:
                    raise ValueError("No solution found")

            element_indices[-1] += 1

    res = convert_indices2elements(element_indices)
    if return_indices:
        return res, element_indices
    else:
        return res


def fib(x: int) -> int:
    """Fast fibonacci function

    written by https://www.codespeedy.com/find-fibonacci-series-in-python/
    """
    return round(math.pow((math.sqrt(5) + 1) / 2, x) / math.sqrt(5))


def split_iterable_by_function(iterable: tuple, function) -> tuple:
    seperate_indices = tuple(idx + 1 for idx, v in enumerate(iterable) if function(v))
    if seperate_indices:
        size = len(iterable)
        zip0 = (0,) + seperate_indices
        zip1 = seperate_indices + (
            (size,) if seperate_indices[-1] != size else tuple([])
        )
        return type(iterable)(iterable[i:j] for i, j in zip(zip0, zip1))
    else:
        return type(iterable)((tuple(iterable),))


def split_iterable_by_n(iterable: tuple, n) -> tuple:
    return split_iterable_by_function(iterable, lambda x: x == n)


def find_all_indices_of_n(n, iterable: tuple) -> tuple:
    return tuple(idx for idx, item in enumerate(iterable) if item == n)


def cyclic_perm(iterable: tuple):
    """Cyclic permutation of an iterable. Return a generator object.

    Adapted function from the reply of Paritosh Singh here
    https://stackoverflow.com/questions/56171246/cyclic-permutation-operators-in-python/56171531
    """

    def reorder_from_idx(idx: int, iterable: tuple) -> tuple:
        return iterable[idx:] + iterable[:idx]

    return (
        functools.partial(reorder_from_idx, i)(iterable) for i in range(len(iterable))
    )


def round_percentage(percentage_per_item: tuple, n: int) -> tuple:
    float_per_item = tuple(percentage * n for percentage in percentage_per_item)
    rounded_items = [int(float_value) for float_value in float_per_item]
    difference = n - sum(rounded_items)
    items_sorted_by_decimals = sorted(
        float_per_item, key=lambda a: a - int(a), reverse=True
    )

    for i in range(abs(difference)):
        rounded_items[float_per_item.index(items_sorted_by_decimals[i])] += 1

    return tuple(rounded_items)


# from https://stackoverflow.com/questions/7267226/range-for-floats/7267287
def drange(x, y, jump):
    while x < y:
        yield float(x)
        x += decimal.Decimal(jump)


def frange(x: fractions.Fraction, y: fractions.Fraction, stepsize: fractions.Fraction):
    while x < y:
        yield x
        x += stepsize


def find_attributes_of_object(obj, omit_private_attributes: bool = True) -> tuple:
    return tuple(
        attribute
        for attribute in dir(obj)
        # no private attributes
        if omit_private_attributes and attribute[0] != "_"
        # no methods
        and not isinstance(getattr(obj, attribute), types.MethodType)
    )


def db2a(db: float) -> float:
    return 10 ** (db / 20)


def a2db(a: float) -> float:
    try:
        return 20 * math.log(a, 10)
    except ValueError:
        return -120


def round_to_next_power_of_n(x: float, n: int) -> float:
    return pow(n, math.ceil(math.log(x) / math.log(n)))


def round_to_next_power_of_2(x: float) -> float:
    return round_to_next_power_of_n(x, 2)
