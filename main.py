"""Mini NPU Simulator - 실행 엔트리포인트.

모드 1: 사용자 입력 3×3 / 모드 2: data.json 일괄 판정 (성능 표·요약은 Phase 4~5).
"""

from typing import Callable, List

from bonus import run_bonus
from benchmark import REPEAT, measure_avg_ms as measure_mac_ms, print_benchmark
from data_loader import analyze_all, load_json, read_matrix
from npu_simulator import judge, mac

SIZE = 3
DATA_PATH = "data.json"
BAR = "#" + "-" * 39


def format_matrix(m: List[List[float]]) -> str:
    return "\n".join("  " + " ".join(f"{v:g}" for v in row) for row in m)


def run_mode1(input_fn: Callable[[str], str] = input,
              output_fn: Callable[[str], None] = print) -> dict:
    out = output_fn
    out(BAR)
    out("# [1] 필터 입력")
    out(BAR)
    filter_a = read_matrix("필터 A", SIZE, input_fn, out)
    out("")
    filter_b = read_matrix("필터 B", SIZE, input_fn, out)
    out("")
    out("[OK] 필터 A, B 저장 완료")
    out(f"필터 A:\n{format_matrix(filter_a)}")
    out(f"필터 B:\n{format_matrix(filter_b)}")
    out("")
    out(BAR)
    out("# [2] 패턴 입력")
    out(BAR)
    pattern = read_matrix("패턴", SIZE, input_fn, out)
    out("")

    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    # 두 필터의 연산을 모두 포함한 1회 평균 시간
    avg_ms = (measure_mac_ms(pattern, filter_a) + measure_mac_ms(pattern, filter_b)) / 2
    verdict = judge(score_a, score_b)

    out(BAR)
    out("# [3] MAC 결과" + (" (판정 불가)" if verdict == "UNDECIDED" else ""))
    out(BAR)
    out(f"A 점수: {score_a!r}")
    out(f"B 점수: {score_b!r}")
    out(f"연산 시간(평균/{REPEAT}회): {avg_ms:.3f} ms")
    if verdict == "UNDECIDED":
        out(f"판정: 판정 불가 (|A-B| < 1e-9)")
    else:
        out(f"판정: {verdict}")
    out("")
    out(BAR)
    out(f"# [4] 성능 분석 (평균/{REPEAT}회)")
    out(BAR)
    print_benchmark(out, sizes=(SIZE,))
    return {"score_a": score_a, "score_b": score_b, "avg_ms": avg_ms, "verdict": verdict}


def summarize(results: List[dict]) -> dict:
    """총/통과/실패 개수와 실패 케이스 목록(id, 사유)을 집계한다."""
    failed = [r for r in results if r["status"] != "PASS"]
    return {"total": len(results), "passed": len(results) - len(failed),
            "failed": len(failed),
            "failures": [(r["id"], r["reason"] or "사유 미상") for r in failed]}


def print_summary(results: List[dict], output_fn: Callable[[str], None] = print) -> dict:
    s = summarize(results)
    output_fn(BAR)
    output_fn("# [4] 결과 요약")
    output_fn(BAR)
    output_fn(f"총 테스트: {s['total']}개")
    total = s["total"]
    pass_rate = s["passed"] / total * 100 if total else 0.0
    fail_rate = s["failed"] / total * 100 if total else 0.0
    output_fn(f"통과: {s['passed']}개 ({pass_rate:.1f}%)")
    output_fn(f"실패: {s['failed']}개 ({fail_rate:.1f}%)")
    if s["failures"]:
        output_fn("")
        output_fn("실패 케이스:")
        for case_id, reason in s["failures"]:
            output_fn(f"- {case_id}: {reason}")
    return s


def run_mode2(path: str = DATA_PATH,
              output_fn: Callable[[str], None] = print) -> list:
    out = output_fn
    try:
        data = load_json(path)
    except ValueError as e:
        out(f"[ERROR] {path} 로드 실패: {e}")
        return []

    filters, problems, results = analyze_all(data)

    out(BAR)
    out("# [1] 필터 로드")
    out(BAR)
    for n in sorted(set(filters) | set(problems)):
        if n in filters:
            out(f"[OK] size_{n} 필터 로드 완료 (Cross, X)")
        else:
            out(f"[X]  size_{n} {problems[n]}")
    out("")
    out(BAR)
    out("# [2] 패턴 분석 (라벨 정규화 적용)")
    out(BAR)
    for r in results:
        out(f"--- {r['id']} ---")
        if r["score_cross"] is not None:
            out(f"Cross 점수: {r['score_cross']!r}")
            out(f"X 점수: {r['score_x']!r}")
            out(f"판정: {r['verdict']} | expected: {r['expected']} | {r['status']}"
                + (f" ({r['reason']})" if r["status"] == "FAIL" else ""))
        else:
            out(f"FAIL: {r['reason']}")
    out("")
    out(BAR)
    out(f"# [3] 성능 분석 (평균/{REPEAT}회)")
    out(BAR)
    print_benchmark(out)
    out("")
    print_summary(results, out)
    return results


def select_mode(input_fn: Callable[[str], str] = input,
                output_fn: Callable[[str], None] = print) -> int:
    output_fn("[모드 선택]")
    output_fn("")
    output_fn("1. 사용자 입력 (3x3)")
    output_fn("2. data.json 분석")
    output_fn("3. 보너스: 패턴 생성기 + 2D/1D 성능 비교")
    while True:
        choice = input_fn("선택: ").strip()
        if choice in ("1", "2", "3"):
            return int(choice)
        output_fn("입력 오류: 1, 2, 3 중 하나를 입력하세요.")


def main() -> None:
    print("=== Mini NPU Simulator ===")
    print()
    try:
        mode = select_mode()
        if mode == 1:
            run_mode1()
        elif mode == 2:
            run_mode2()
        else:
            run_bonus()
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")


if __name__ == "__main__":
    main()
