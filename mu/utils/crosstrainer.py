import operator


class Stack(object):
    def __init__(self, size: int=10, fitness: str="min",
                 condition_to_add_if_not_full_yet: bool=False):
        try:
            assert(fitness == "min" or fitness == "max")
        except AssertionError:
            raise ValueError("Fitness can only be 'min' or 'max'.")
        self.size = size
        self.fitness = fitness
        self._ls = []
        self._condition = condition_to_add_if_not_full_yet

    def __repr__(self) -> str:
        return repr(self._ls)

    def append(self, item, fitness: float) -> None:
        if self._ls:
            current_fitness = tuple(p[1] for p in self._ls)
            minima, maxima = min(current_fitness), max(current_fitness)
            if self.fitness == "min":
                reverse = False
                if fitness <= maxima:
                    addable = True
                else:
                    addable = False
            else:
                reverse = True
                if fitness >= minima:
                    addable = True
                else:
                    addable = False
            if self.size is None:
                if addable is True:
                    self._ls.append((item, fitness))
            else:
                if len(self._ls) < self.size:
                    if self._condition is False or addable is True:
                        self._ls.append((item, fitness))
                elif addable is True:
                    self._ls = self._ls[:-1]
                    self._ls.append((item, fitness))
            self._ls = sorted(self._ls, key=operator.itemgetter(1),
                              reverse=reverse)
        else:
            self._ls.append((item, fitness))

    def convert2list(self, filter_only_highest=False) -> list:
        ls = self._ls
        if filter_only_highest is True and ls:
            highest = ls[0][1]
            ls = filter(lambda item: item[1] == highest, ls)
        return list(ls)

    @property
    def best(self):
        return self._ls[0]


class MultiDimensionalRating(object):
    def __init__(self, size: int=10, fitness: list=[-1],
                 condition_to_add_if_not_full_yet: bool=False):
        self.amount_items = len(fitness)
        self.size = size
        self.fitness = fitness
        self._condition = condition_to_add_if_not_full_yet
        self._items = []
        self._fitness = []

    def __repr__(self):
        return repr(self._items)

    def is_bigger(self, fitness0, fitness1) -> bool:
        """
        Checks whether the second fitness is
        better than the first one.
        Return True if true and False if false.
        """
        win = 0
        for item0, item1, fit in zip(fitness0, fitness1, self.fitness):
            if fit >= 0:
                if item0 > item1:
                    return False
                elif item0 < item1:
                    win -= 1
            else:
                if item0 > item1:
                    win -= 1
                elif item0 < item1:
                    return False
        return win < 0

    def append(self, item, *fitness):
        def trim_data(new, data, i):
            data = data[0:-i] + [new] + data[-i:]
            return data[:self.size]

        try:
            assert(len(fitness) == self.amount_items)
        except AssertionError:
            msg = "These has to be {0} fitness attributes.".format(
                    self.amount_items)
            raise ValueError(msg)
        if self._items:
            already = False
            for i1 in self._items:
                if i1 == item:
                    already = True
                    break
            if already is False:
                i = 0
                fitness_items = self._fitness[-1]
                while self.is_bigger(fitness_items, fitness) is True:
                    i += 1
                    try:
                        fitness_items = self._fitness[-(i + 1)]
                    except IndexError:
                        break
                if i > 0:
                    self._items = trim_data(item, self._items, i)
                    self._fitness = trim_data(fitness, self._fitness, i)
        else:
            self._items.append(item)
            self._fitness.append(fitness)

    def extend(self, iterable):
        for pair in iterable:
            self.append(pair[0], *pair[1])

    def convert2list(self, filter_only_highest=False) -> list:
        items = self._items
        values = self._fitness
        ls = zip(items, values)
        if filter_only_highest is True and items:
            highest = values[0]
            items = filter(lambda item: item[1] == highest, ls)
        return list(ls)
