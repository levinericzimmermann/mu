import functools
import math
import operator
import itertools
import scipy
import numpy as np


def accumulate_from_zero(iterator) -> tuple:
    return tuple(itertools.accumulate((0,) + (tuple(iterator))))


def brownian(x0, n, dt, delta, out=None, random_state=None):
    """
    Generate an instance of Brownian motion (i.e. the Wiener process):

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
    r = scipy.stats.norm.rvs(
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


def scale(a, minima=0, maxima=15):
    return np.interp(a, (a.min(), a.max()), (minima, maxima))


def graycode(length, modulus):
    """
    Returns the n-tuple reverse Gray code mod m.
    https://yetanothermathblog.com/tag/gray-codes/
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


def euclid(size, distribution) -> tuple:
    standard_size = size // distribution
    rest = size % distribution
    data = (standard_size for n in range(distribution))
    if rest:
        added = accumulate_from_zero(euclid(distribution, rest))
        return tuple(s + 1 if idx in added else s for idx, s in enumerate(data))
    else:
        return tuple(data)


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


def interlock_tuples(t0, t1) -> tuple:
    size0, size1 = len(t0), len(t1)
    difference = size0 - size1
    indices = functools.reduce(
        operator.add, ((0, 1) for n in range(min((size0, size1))))
    )
    if difference > 0:
        indices = tuple(0 for i in range(difference)) + indices
    else:
        indices = indices + tuple(1 for i in range(difference))
    t0_it = iter(t0)
    t1_it = iter(t1)
    return tuple(next(t0_it) if idx == 0 else next(t1_it) for idx in indices)


def not_fibonacci_transition(size0: int, size1: int, element0=0, element1=1) -> tuple:
    def write_to_n_element(it, element) -> tuple:
        return tuple(tuple(element for n in range(x)) for x in it)

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
