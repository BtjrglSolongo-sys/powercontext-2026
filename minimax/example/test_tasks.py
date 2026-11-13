import unittest
from tasks import TASKS, list_tasks


class RegressionTests(unittest.TestCase):
    def test_default_returns_all_in_order(self):
        self.assertEqual(list_tasks(), TASKS)

    def test_results_are_independent(self):
        result = list_tasks()
        result[0]["title"] = "changed"
        self.assertEqual(TASKS[0]["title"], "Write specification")


class FeatureTests(unittest.TestCase):
    def test_open_filter(self):
        self.assertEqual([task["id"] for task in list_tasks("open")], [2, 3])

    def test_done_filter(self):
        self.assertEqual([task["id"] for task in list_tasks("done")], [1])

    def test_invalid_status(self):
        with self.assertRaises(ValueError):
            list_tasks("unknown")


if __name__ == "__main__":
    unittest.main()
