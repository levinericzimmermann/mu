import unittest
from mu.utils import prime_factors


class TestFactorise(unittest.TestCase):
    def test_factorise(self):
        self.assertEqual(prime_factors.factorise(6), [2, 3])
        self.assertEqual(prime_factors.factorise(9), [3, 3])
        self.assertEqual(prime_factors.factorise(15), [3, 5])
        self.assertEqual(prime_factors.factorise(36), [2, 2, 3, 3])
