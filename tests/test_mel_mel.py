import unittest
from mu.mel import mel


class MelTest(unittest.TestCase):
    def test_hash(self):
        h = mel.Mel([])
        self.assertEqual(hash(h), hash(tuple([])))


class HarmonyTest(unittest.TestCase):
    def test_hash(self):
        h = mel.Harmony()
        self.assertEqual(hash(h), hash(tuple([])))
