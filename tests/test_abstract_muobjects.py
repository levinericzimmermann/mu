import unittest
from mu.abstract import muobjects


class AbstractTest(unittest.TestCase):
    def test_construction(self):
        assert(muobjects.MUList([1, 2, 3]))
        assert(muobjects.MUSet([1, 2, 3]))
        assert(muobjects.MUDict({"hi": 10, "hey": 20}))
        assert(muobjects.MUFloat(2.429))
        assert(muobjects.MUTuple([1, 2, 3]))
        assert(muobjects.MUOrderedSet([1, 2, 3]))


class MUListTest(unittest.TestCase):
    def test_getitem(self):
        ls = muobjects.MUList([1, 2, 3])
        self.assertEqual(ls[0], 1)
        self.assertEqual(ls[0:1], muobjects.MUList([1]))

        class TestClass(muobjects.MUList):
            def __init__(self, iter, x=1):
                muobjects.MUList.__init__(self, iter)
                self.x = x

            def copy(self):
                copied = muobjects.MUList.copy(self)
                copied = type(self)(copied)
                copied.x = self.x
                return copied

            def __eq__(self, other):
                try:
                    return tuple(self) == tuple(other) and self.x == other.x
                except AttributeError:
                    return False

        testls0 = TestClass([1, 2, 3], 10)
        testls1 = TestClass([1, 2], 10)
        self.assertEqual(testls0[:2], testls1)
