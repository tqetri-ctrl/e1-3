"""Mini NPU Simulator - 실행 엔트리포인트.

Phase 2: 모드 1 (사용자 입력 3×3). 모드 2(data.json)는 Phase 3에서 구현.
"""

import time
from typing import Callable, List

from data_loader import read_matrix
from npu_simulator import judge, mac

SIZE = 3
REPEAT = 10
BAR = "#" + "-" * 39


def measure_mac_ms(pattern, filt, repeat: int = REPEAT) -> float:
    """MAC 연산 구간만 repeat회 측정해 평균(ms)을 반환한다 (I/O 제외)."""
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac(pattern, filt)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


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
    return {"score_a": score_a, "score_b": score_b, "avg_ms": avg_ms, "verdict": verdict}


def select_mode(input_fn: Callable[[str], str] = input,
                output_fn: Callable[[str], None] = print) -> int:
    output_fn("[모드 선택]")
    output_fn("")
    output_fn("1. 사용자 입력 (3x3)")
    output_fn("2. data.json 분석")
    while True:
        choice = input_fn("선택: ").strip()
        if choice in ("1", "2"):
            return int(choice)
        output_fn("입력 오류: 1 또는 2를 입력하세요.")


def main() -> None:
    print("=== Mini NPU Simulator ===")
    print()
    try:
        mode = select_mode()
        if mode == 1:
            run_mode1()
        else:
            print("모드 2(data.json 분석)는 아직 구현되지 않았습니다. (Phase 3 예정)")
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 종료되어 프로그램을 마칩니다.")


if __name__ == "__main__":
    main()
