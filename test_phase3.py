import os
import tempfile
import unittest

from data_loader import analyze_all, load_json, parse_filters
from main import run_mode2

C3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
HERE = os.path.dirname(os.path.abspath(__file__))


def base(patterns, filters=None):
    return {"filters": filters or {"size_3": {"cross": C3, "x": X3}},
            "patterns": patterns}


def results_of(patterns, filters=None):
    return {r["id"]: r for r in analyze_all(base(patterns, filters))[2]}


class Phase3Test(unittest.TestCase):
    def test_pass_with_label_variants(self):
        r = results_of({"size_3_1": {"input": C3, "expected": "+"},
                        "size_3_2": {"input": X3, "expected": "x"},
                        "size_3_3": {"input": C3, "expected": "Cross"}})
        self.assertTrue(all(v["status"] == "PASS" for v in r.values()))
        self.assertEqual(r["size_3_1"]["score_cross"], 5)
        self.assertEqual(r["size_3_1"]["score_x"], 1)

    def test_filter_key_normalized(self):
        f, p = parse_filters({"size_3": {"+": C3, "X": X3}})
        self.assertEqual(sorted(f[3]), ["Cross", "X"])
        self.assertEqual(p, {})

    def test_wrong_verdict_is_fail(self):
        r = results_of({"size_3_1": {"input": C3, "expected": "x"}})["size_3_1"]
        self.assertEqual(r["status"], "FAIL")

    def test_tie_is_undecided_fail(self):
        zero = [[0] * 3 for _ in range(3)]
        r = results_of({"size_3_1": {"input": zero, "expected": "x"}})["size_3_1"]
        self.assertEqual((r["verdict"], r["status"]), ("UNDECIDED", "FAIL"))
        self.assertIn("동점", r["reason"])

    def test_float_noise_tie(self):
        # 0.1+0.2 vs 0.3 같은 미세 오차는 동점으로 처리된다
        f = {"size_1": {"cross": [[1]], "x": [[1]]}}
        r = results_of({"size_1_1": {"input": [[0.1 + 0.2]], "expected": "x"}}, f)
        self.assertEqual(r["size_1_1"]["verdict"], "UNDECIDED")

    def test_faults_do_not_stop_run(self):
        r = results_of({
            "size_3_1": {"input": [[1, 2], [3, 4]], "expected": "+"},   # 크기 불일치
            "size_3_2": {"input": C3},                                   # expected 누락
            "size_3_3": {"input": C3, "expected": "?"},                  # 알 수 없는 라벨
            "size_7_1": {"input": C3, "expected": "+"},                  # 필터 없음
            "bad_key": {"input": C3, "expected": "+"},                   # 키 형식 오류
            "size_3_4": "oops",                                          # 항목 형식 오류
            "size_3_5": {"input": [[1, "a", 0]] * 3, "expected": "+"},   # 숫자 아님
            "size_3_6": {"input": C3, "expected": "+"},                  # 정상
        })
        for k in ("size_3_1", "size_3_2", "size_3_3", "size_7_1", "bad_key",
                  "size_3_4", "size_3_5"):
            self.assertEqual(r[k]["status"], "FAIL", k)
            self.assertTrue(r[k]["reason"], k)
        self.assertEqual(r["size_3_6"]["status"], "PASS")

    def test_broken_filter_fails_only_its_cases(self):
        f = {"size_3": {"cross": C3},  # x 필터 누락
             "size_5": {"cross": [[0] * 5] * 5, "x": [[0] * 5] * 5}}
        r = results_of({"size_3_1": {"input": C3, "expected": "+"}}, f)
        self.assertEqual(r["size_3_1"]["status"], "FAIL")
        self.assertIn("x", r["size_3_1"]["reason"].lower())

    def test_load_json_errors(self):
        with self.assertRaises(ValueError):
            load_json("no_such_file.json")
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "bad.json")
            for content in ("{not json", "[]", '{"filters": {}}'):
                with open(p, "w", encoding="utf-8") as f:
                    f.write(content)
                with self.assertRaises(ValueError):
                    load_json(p)

    def test_mode2_handles_bad_file(self):
        out = []
        self.assertEqual(run_mode2("no_such_file.json", out.append), [])
        self.assertIn("ERROR", out[0])

    def test_real_data_json(self):
        out = []
        res = run_mode2(os.path.join(HERE, "data.json"), out.append)
        self.assertEqual(len(res), 7)
        self.assertTrue(all(r["status"] == "PASS" for r in res))


if __name__ == "__main__":
    unittest.main()
