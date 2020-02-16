import collections
import functools
import math
import operator

from mu.utils import prime_factors


def basic_indispensability(n, p):
    if p == 2:
        return p - n
    elif n == p - 1:
        return int(p / 4)
    else:
        factorised = tuple(sorted(prime_factors.factorise(p - 1), reverse=True))
        q = indispensability(n - int(n / p), factorised)
        return int(q + (2 * math.sqrt((q + 1) / p)))


def detect_order(p: tuple):
    p = sorted(p)
    if p[-1] != 2:
        return len(p)
    else:
        return 0


def indispensability(n, p: tuple):
    z = len(p)
    p = (1,) + p + (1,)
    r_sum = []
    for r in range(0, z):
        up = (n - 2) % functools.reduce(
            operator.mul, tuple(p[j] for j in range(1, z + 1))
        )
        down = functools.reduce(operator.mul, tuple(p[z + 1 - k] for k in range(r + 1)))
        local_result = 1 + (int(1 + (up / down)) % p[z - r])
        base_indispensability = basic_indispensability(local_result, p[z - r])
        product = functools.reduce(operator.mul, tuple(p[i] for i in range(0, z - r)))
        r_sum.append(product * base_indispensability)
    return sum(r_sum)


def indispensability_for_bar(p: tuple):
    length = functools.reduce(operator.mul, p)
    return tuple(indispensability(i + 1, p) for i in range(length))


def bar_indispensability2indices(bar_indispensability):
    bar_indispensability = tuple((i, n) for i, n in enumerate(bar_indispensability))
    ig0 = operator.itemgetter(0)
    ig1 = operator.itemgetter(1)
    return tuple(
        ig0(ind) for ind in sorted(bar_indispensability, key=ig1, reverse=True)
    )


def convert_dominant_prime_and_length2p(dominant_prime, length):
    sorted_factors = sorted(prime_factors.factorise(length))
    defactorised_prime = sorted(prime_factors.factorise(dominant_prime), reverse=True)
    remaining_factors = collections.Counter(sorted_factors) - collections.Counter(
        defactorised_prime
    )
    remaining_factors = functools.reduce(
        operator.add,
        tuple(tuple(n for i in range(m)) for n, m in tuple(remaining_factors.items())),
    )
    return tuple(sorted(remaining_factors)) + tuple(defactorised_prime)


def convert_p2dominant_prime_and_length(p):
    length = functools.reduce(operator.mul, p)
    dominant_prime = functools.reduce(operator.mul, [])
    return dominant_prime, length
