import unittest

from data_loader import parse_row, read_matrix
from main import run_mode1, select_mode


def feeder(lines):
    it = iter(lines)
    return lambda prompt="": next(it)


class Phase2Test(unittest.TestCase):
    def test_parse_row_ok(self):
        self.assertEqual(parse_row("0 1  0.5", 3), [0.0, 1.0, 0.5])

    def test_parse_row_errors(self):
        for bad in ("1 2", "1 2 3 4", "a b c", "1 2 x", "", "nan 1 2", "inf 1 2"):
            with self.assertRaises(ValueError):
                parse_row(bad, 3)

    def test_read_matrix_retries(self):
        msgs = []
        m = read_matrix("필터", 3,
                        feeder(["1 2", "a b c", "1 0 1", "0 1 0", "1 2 3 4", "1 0 1"]),
                        msgs.append)
        self.assertEqual(m, [[1, 0, 1], [0, 1, 0], [1, 0, 1]])
        self.assertEqual(sum("입력 형식 오류" in s for s in msgs), 3)

    def run_mode1(self, pattern_lines):
        out = []
        lines = ["0 1 0", "1 1 1", "0 1 0", "1 0 1", "0 1 0", "1 0 1"] + pattern_lines
        res = run_mode1(feeder(lines), out.append)
        return res, "\n".join(out)

    def test_mode1_x_pattern(self):
        res, text = self.run_mode1(["1 0 1", "0 1 0", "1 0 1"])
        self.assertEqual((res["score_a"], res["score_b"], res["verdict"]), (1.0, 5.0, "B"))
        self.assertIn("A 점수: 1.0", text)
        self.assertIn("판정: B", text)
        self.assertIn("ms", text)

    def test_mode1_cross_pattern(self):
        res, _ = self.run_mode1(["0 1 0", "1 1 1", "0 1 0"])
        self.assertEqual(res["verdict"], "A")

    def test_mode1_undecided(self):
        res, text = self.run_mode1(["0 0 0", "0 0 0", "0 0 0"])
        self.assertEqual(res["verdict"], "UNDECIDED")
        self.assertIn("판정 불가", text)

    def test_mode1_recovers_from_bad_input(self):
        out = []
        lines = ["0 1", "0 1 0", "1 1 1", "0 1 0",
                 "1 0 1", "0 q 0", "0 1 0", "1 0 1",
                 "1 0 1", "0 1 0", "1 0 1"]
        res = run_mode1(feeder(lines), out.append)
        self.assertEqual(res["verdict"], "B")

    def test_select_mode(self):
        out = []
        self.assertEqual(select_mode(feeder(["x", "3", "2"]), out.append), 2)
        self.assertEqual(sum("입력 오류" in s for s in out), 2)


if __name__ == "__main__":
    unittest.main()
