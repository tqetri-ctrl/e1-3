import unittest

from npu_simulator import (Grid, judge, mac, normalize_label, validate_matrix,
                           LABEL_CROSS, LABEL_X, LABEL_UNDECIDED)

CROSS = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
XPAT = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class Phase1Test(unittest.TestCase):
    def test_mac_mission_examples(self):
        self.assertEqual(mac(CROSS, CROSS), 5)
        self.assertEqual(mac(CROSS, XPAT), 1)
        self.assertEqual(mac(XPAT, CROSS), 1)
        self.assertEqual(mac(XPAT, XPAT), 5)

    def test_mac_size_mismatch(self):
        with self.assertRaises(ValueError):
            mac(CROSS, [[1, 2], [3, 4]])

    def test_validate(self):
        self.assertEqual(validate_matrix(CROSS, 3), 3)
        for bad in ([[1, 2, 3], [1, 2, 3]], [[1, 2], [3]], [], [[1, "a"], [2, 3]]):
            with self.assertRaises(ValueError):
                validate_matrix(bad)
        with self.assertRaises(ValueError):
            validate_matrix(CROSS, 5)

    def test_grid(self):
        g = Grid.zeros(5)
        g.set(2, 3, 1.5)
        self.assertEqual(g.get(2, 3), 1.5)
        self.assertEqual(len(g.flatten()), 25)

    def test_normalize(self):
        for raw in ("+", "cross", "Cross", "CROSS", " cross "):
            self.assertEqual(normalize_label(raw), LABEL_CROSS)
        for raw in ("x", "X"):
            self.assertEqual(normalize_label(raw), LABEL_X)
        for bad in ("circle", "", None, 1):
            with self.assertRaises(ValueError):
                normalize_label(bad)

    def test_judge(self):
        self.assertEqual(judge(1.0, 5.0), "B")
        self.assertEqual(judge(5.0, 1.0), "A")
        self.assertEqual(judge(5.0, 1.0, LABEL_CROSS, LABEL_X), LABEL_CROSS)
        # 부동소수점 미세 오차 -> 동점
        self.assertNotEqual(0.1 + 0.2, 0.3)
        self.assertEqual(judge(0.1 + 0.2, 0.3), LABEL_UNDECIDED)
        self.assertEqual(judge(0.9000000000000000, 0.8999999999999999), LABEL_UNDECIDED)


if __name__ == "__main__":
    unittest.main()
