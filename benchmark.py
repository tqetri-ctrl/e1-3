"""성능 분석 (Phase 4): 크기별 MAC 연산 평균 시간(ms)과 연산 횟수(N²) 표.

시간은 mac() 호출 구간만 perf_counter로 측정하며 I/O는 포함하지 않는다.
"""

import time
from typing import Callable, Dict, List, Sequence

from npu_simulator import mac

SIZES = (3, 5, 13, 25)
REPEAT = 10


def make_sample(n: int) -> List[List[float]]:
    """측정용 n×n 입력. 값이 0이 아닌 결정적 패턴(곱셈이 실제로 일어나도록)."""
    return [[(i * n + j) % 7 + 1 for j in range(n)] for i in range(n)]


def measure_avg_ms(pattern, filt, repeat: int = REPEAT,
                   mac_fn: Callable = mac) -> float:
    """mac_fn(pattern, filt)을 repeat회 호출한 1회 평균 시간(ms)."""
    if repeat < 1:
        raise ValueError("repeat는 1 이상이어야 합니다.")
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        mac_fn(pattern, filt)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


def run_benchmark(sizes: Sequence[int] = SIZES, repeat: int = REPEAT) -> List[Dict]:
    rows = []
    for n in sizes:
        pattern = make_sample(n)
        filt = make_sample(n)
        rows.append({"n": n, "avg_ms": measure_avg_ms(pattern, filt, repeat),
                     "ops": n * n})
    return rows


def _pad(text: str, width: int, right: bool = False) -> str:
    """한글(전각)을 2칸으로 계산해 정렬한다."""
    w = sum(2 if ord(c) > 0x2E80 else 1 for c in text)
    gap = " " * max(width - w, 0)
    return gap + text if right else text + gap


def format_table(rows: List[Dict], repeat: int = REPEAT) -> List[str]:
    lines = [_pad("크기", 8) + _pad("평균 시간(ms)", 16, True)
             + _pad("연산 횟수(N²)", 16, True) + _pad("반복", 6, True),
             "-" * 46]
    for r in rows:
        label = f"{r['n']}×{r['n']}"
        lines.append(f"{label:<8}{r['avg_ms']:>16.4f}{r['ops']:>16d}{repeat:>6d}")
    return lines


def print_benchmark(output_fn: Callable[[str], None] = print,
                    sizes: Sequence[int] = SIZES, repeat: int = REPEAT) -> List[Dict]:
    rows = run_benchmark(sizes, repeat)
    for line in format_table(rows, repeat):
        output_fn(line)
    return rows
