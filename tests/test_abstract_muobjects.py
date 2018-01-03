import unittest
from mu.abstract import muobjects


class AbstractTest(unittest.TestCase):
    def test_construction(self):
        assert(muobjects.MUList([1, 2, 3]))
        assert(muobjects.MUSet([1, 2, 3]))
        assert(muobjects.MUDict({"hi": 10, "hey": 20}))
        assert(muobjects.MUFloat(2.429))
        assert(muobjects.MUTuple([1, 2, 3]))
