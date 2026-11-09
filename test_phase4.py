import unittest

from benchmark import (REPEAT, SIZES, format_table, make_sample, measure_avg_ms,
                       print_benchmark, run_benchmark)
from main import run_mode1, run_mode2


class Phase4Test(unittest.TestCase):
    def test_sample_shape(self):
        for n in SIZES:
            m = make_sample(n)
            self.assertEqual((len(m), {len(r) for r in m}), (n, {n}))
            self.assertTrue(all(v != 0 for r in m for v in r))

    def test_repeat_count_and_average(self):
        calls = []
        measure_avg_ms([[1]], [[1]], 10, lambda p, f: calls.append(1))
        self.assertEqual(len(calls), 10)
        with self.assertRaises(ValueError):
            measure_avg_ms([[1]], [[1]], 0)

    def test_rows(self):
        rows = run_benchmark()
        self.assertEqual([r["n"] for r in rows], [3, 5, 13, 25])
        self.assertEqual([r["ops"] for r in rows], [9, 25, 169, 625])
        self.assertTrue(all(r["avg_ms"] > 0 for r in rows))

    def test_table_format(self):
        out = []
        print_benchmark(out.append)
        text = "\n".join(out)
        for s in ("3×3", "5×5", "13×13", "25×25", "9", "625", "평균 시간(ms)"):
            self.assertIn(s, text)
        self.assertEqual(len(format_table(run_benchmark())), 6)

    def test_modes_print_table(self):
        out = []
        lines = iter(["0 1 0", "1 1 1", "0 1 0", "1 0 1", "0 1 0", "1 0 1",
                      "1 0 1", "0 1 0", "1 0 1"])
        run_mode1(lambda p="": next(lines), out.append)
        text = "\n".join(out)
        self.assertIn("성능 분석", text)
        self.assertIn("3×3", text)
        self.assertNotIn("25×25", text)  # 모드 1은 3×3만

        out = []
        run_mode2("data.json", out.append)
        text = "\n".join(out)
        for s in ("3×3", "5×5", "13×13", "25×25"):
            self.assertIn(s, text)


if __name__ == "__main__":
    unittest.main()
