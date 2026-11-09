import json
import os
import tempfile
import unittest

from main import run_mode2, summarize

C3 = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
X3 = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]


class Phase5Test(unittest.TestCase):
    def test_summarize(self):
        s = summarize([{"id": "a", "status": "PASS", "reason": ""},
                       {"id": "b", "status": "FAIL", "reason": "크기 불일치"},
                       {"id": "c", "status": "FAIL", "reason": ""}])
        self.assertEqual((s["total"], s["passed"], s["failed"]), (3, 1, 2))
        self.assertEqual(s["failures"], [("b", "크기 불일치"), ("c", "사유 미상")])

    def test_console_summary_with_failures(self):
        data = {"filters": {"size_3": {"cross": C3, "x": X3}},
                "patterns": {"size_3_1": {"input": C3, "expected": "+"},
                             "size_3_2": {"input": [[1, 2], [3, 4]], "expected": "x"},
                             "size_3_3": {"input": [[0] * 3] * 3, "expected": "x"}}}
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "d.json")
            with open(p, "w", encoding="utf-8") as f:
                json.dump(data, f)
            out = []
            run_mode2(p, out.append)
        text = "\n".join(out)
        self.assertIn("총 테스트: 3개", text)
        self.assertIn("통과: 1개", text)
        self.assertIn("실패: 2개", text)
        self.assertIn("- size_3_2: 크기 불일치", text)
        self.assertIn("- size_3_3: 동점", text)

    def test_console_summary_all_pass(self):
        out = []
        run_mode2("data.json", out.append)
        text = "\n".join(out)
        self.assertIn("총 테스트: 7개", text)
        self.assertIn("실패: 0개", text)
        self.assertNotIn("실패 케이스:", text)


if __name__ == "__main__":
    unittest.main()
