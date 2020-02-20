import collections
import functools
import math
import operator

from mu.utils import prime_factors


def detect_order(primes: tuple) -> int:
    primes = sorted(primes)
    if primes[-1] != 2:
        return len(primes)
    else:
        return 0


def basic_indispensability(beat_idx: int, prime_number: int) -> int:
    """Calculate indispensability for the nth beat of a metre thats composed of

    only one single prime number.
    """
    if prime_number == 2:
        return prime_number - beat_idx
    elif beat_idx == prime_number - 1:
        return int(prime_number / 4)
    else:
        factorised = tuple(
            sorted(prime_factors.factorise(prime_number - 1), reverse=True)
        )
        q = indispensability(beat_idx - int(beat_idx / prime_number), factorised)
        return int(q + (2 * math.sqrt((q + 1) / prime_number)))


def indispensability(beat_idx: int, primes: tuple) -> int:
    """Calculate indispensability for the nth beat of a metre.

    The metre is defined with the primes argument.
    """
    z = len(primes)
    primes = (1,) + primes + (1,)
    r_sum = []
    for r in range(0, z):
        up = (beat_idx - 2) % functools.reduce(
            operator.mul, tuple(primes[j] for j in range(1, z + 1))
        )
        down = functools.reduce(
            operator.mul, tuple(primes[z + 1 - k] for k in range(r + 1))
        )
        local_result = 1 + (int(1 + (up / down)) % primes[z - r])
        base_indispensability = basic_indispensability(local_result, primes[z - r])
        product = functools.reduce(
            operator.mul, tuple(primes[i] for i in range(0, z - r))
        )
        r_sum.append(product * base_indispensability)
    return sum(r_sum)


def indispensability_for_bar(primes: tuple) -> tuple:
    """Convert indispensability for each beat of a particular metre.

    The metre is defined via the primes argument and is the product
    of the relevant primes.
    """
    length = functools.reduce(operator.mul, primes)
    return tuple(indispensability(i + 1, primes) for i in range(length))


def bar_indispensability2indices(bar_indispensability: tuple) -> tuple:
    bar_indispensability = tuple((i, n) for i, n in enumerate(bar_indispensability))
    ig0 = operator.itemgetter(0)
    ig1 = operator.itemgetter(1)
    return tuple(
        ig0(ind) for ind in sorted(bar_indispensability, key=ig1, reverse=True)
    )


def convert_dominant_prime_and_length2p(dominant_prime: int, length: int) -> tuple:
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


def convert_p2dominant_prime_and_length(p) -> tuple:
    length = functools.reduce(operator.mul, p)
    dominant_prime = functools.reduce(operator.mul, [])
    return dominant_prime, length
