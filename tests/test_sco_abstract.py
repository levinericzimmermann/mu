import unittest
from mu.sco import abstract


class EventTest(unittest.TestCase):
    def test_abstract_error(self):
        self.assertRaises(TypeError, abstract.Event)

    def test_is_uniform_error(self):
        self.assertRaises(NotImplementedError, abstract.Event.is_uniform)

    def test_duration_error(self):
        self.assertRaises(TypeError, abstract.Event.duration)


class UniformEventTest(unittest.TestCase):
    def test_is_uniform(self):
        self.assertEqual(abstract.UniformEvent.is_uniform(), True)


class ComplexEventTest(unittest.TestCase):
    def test_is_uniform(self):
        self.assertEqual(abstract.ComplexEvent.is_uniform(), False)


if __name__ == "__main__":
    unittest.main()
