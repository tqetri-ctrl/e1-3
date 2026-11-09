import unittest

from bonus import (compare_2d_1d, flatten, format_compare_table, generate_cross,
                   generate_x, mac_1d, read_size, run_bonus)
from main import select_mode
from npu_simulator import mac

CROSS3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


def feeder(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


class Phase6Test(unittest.TestCase):
    def test_generators_match_mission_examples(self):
        self.assertEqual(generate_cross(3), CROSS3)
        self.assertEqual(generate_x(3), X3)

    def test_generator_shape_and_counts(self):
        for n in (3, 5, 13, 25):
            c, x = generate_cross(n), generate_x(n)
            self.assertEqual((len(c), {len(r) for r in c}), (n, {n}))
            self.assertEqual(sum(map(sum, c)), 2 * n - 1)
            self.assertEqual(sum(map(sum, x)), 2 * n - 1)  # 홀수 n

    def test_generator_rejects_bad_size(self):
        for bad in (0, -1, 2.5, "3", True):
            with self.assertRaises(ValueError):
                generate_cross(bad)

    def test_generated_patterns_are_discriminated(self):
        for n in (5, 13, 25):
            c, x = generate_cross(n), generate_x(n)
            self.assertGreater(mac(c, c), mac(c, x))
            self.assertGreater(mac(x, x), mac(x, c))

    def test_mac_1d_equals_2d(self):
        for n in (3, 5, 13, 25):
            c, x = generate_cross(n), generate_x(n)
            self.assertEqual(mac_1d(flatten(c), flatten(x)), mac(c, x))
        with self.assertRaises(ValueError):
            mac_1d([1, 2], [1])

    def test_compare_rows_and_table(self):
        rows = compare_2d_1d([3, 5], repeat=3)
        self.assertEqual([r["n"] for r in rows], [3, 5])
        self.assertTrue(all(r["ms_2d"] > 0 and r["ms_1d"] > 0 for r in rows))
        self.assertEqual(len(format_compare_table(rows)), 4)
        with self.assertRaises(ValueError):
            compare_2d_1d([3], repeat=0)

    def test_read_size_retries(self):
        out = []
        self.assertEqual(read_size(feeder(["a", "2", "-5", "7"]), out.append), 7)
        self.assertEqual(len(out), 3)

    def test_run_bonus(self):
        out = []
        res = run_bonus(feeder(["x", "7"]), out.append)
        text = "\n".join(out)
        self.assertEqual(res["n"], 7)
        self.assertEqual(res["verdicts"], {"Cross": "Cross", "X": "X"})
        self.assertIn("2D/1D", text)
        self.assertIn("7×7", text)

    def test_select_mode_accepts_3(self):
        self.assertEqual(select_mode(feeder(["3"]), lambda s: None), 3)


if __name__ == "__main__":
    unittest.main()
