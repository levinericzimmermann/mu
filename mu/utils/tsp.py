import itertools
import math

import crosstrainer
from mu.mel import mel


def solve(distance_matrix: tuple, circular=True, add_progressbar=False) -> tuple:
    def calc_distance(per):
        return sum(distance_matrix[i][j] for i, j in zip(per, per[1:]))

    def is_reverse(ind0, ind1):
        return ind0 == tuple(reversed(ind1))

    def is_cyclic_permutation(ind0, ind1):
        ind0 = ind0 + ind0
        return ind1 in ind0

    def append_circular(hof, per):
        per = (0,) + per + (0,)
        addable = True
        for per1 in hof._items:
            if is_reverse(per, per1 + (0,)) is True:
                addable = False
        if addable is True:
            d = calc_distance(per)
            hof.append(per[:-1], d)

    def append_non_circular(hof, per):
        addable = True
        for per1 in hof._items:
            tests = (
                is_reverse(per, per1) is True,
                is_cyclic_permutation(per, per1) is True,
                is_reverse(per + (per[0],), per1 + (per1[0],)) is True,
            )
            if any(tests) is True:
                addable = False
        if addable is True:
            d = calc_distance(per)
            hof.append(per, d)

    if add_progressbar is True:
        import progressbar
    else:
        progressbar = False

    amount_elements = len(distance_matrix)
    elements = tuple(range(amount_elements))
    if circular is True:
        permutations = itertools.permutations(elements[1:])
        append_function = append_circular
        amount_permutations = math.factorial(amount_elements - 1)
    else:
        permutations = itertools.permutations(elements)
        append_function = append_non_circular
        amount_permutations = math.factorial(amount_elements)
    hof = crosstrainer.MultiDimensionalRating(
        size=amount_permutations, fitness=[-1], condition_to_add_if_not_full_yet=False
    )
    if add_progressbar is True:
        with progressbar.ProgressBar(max_value=amount_permutations) as bar:
            for i, per in enumerate(permutations):
                append_function(hof, per)
                bar.update(i)
    else:
        for per in permutations:
            append_function(hof, per)
    minima = min(hof._fitness)
    # print(hof._fitness)
    best = [item for item, fit in zip(hof._items, hof._fitness) if fit == minima]
    return best


def solve_tuples_with_hamming_distance(
    tuples, circular=True, add_progressbar=False
) -> tuple:
    from scipy.spatial import distance

    distance_matrix = []
    for t0 in tuples:
        ldm = []
        for t1 in tuples:
            d = distance.hamming(t0, t1)
            ldm.append(d)
        distance_matrix.append(ldm)
    solutions = solve(distance_matrix, circular, add_progressbar)
    return solutions


def solve_mel_object(
    mel_object: mel.Mel, circular=True, add_progressbar=False
) -> tuple:
    mel_object_copied = mel_object.copy()
    mel_object_copied.val_border = 2
    tuples = [tuple(p.monzo) for p in mel_object_copied]
    max_length = max(len(m) for m in tuples)
    tuples = [m + tuple(0 for i in range(max_length - len(m))) for m in tuples]
    solutions = solve_tuples_with_hamming_distance(tuples, circular, add_progressbar)
    mels = []
    for sol in solutions:
        mels.append([mel_object[idx] for idx in sol])
    return mels
