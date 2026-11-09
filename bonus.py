"""보너스 과제 (Phase 6): N×N 패턴 생성기, 1D(평탄화) MAC, 2D vs 1D 성능 비교."""

import time
from typing import Callable, Dict, List, Sequence

from benchmark import REPEAT, SIZES, _pad
from npu_simulator import Grid, judge, mac

Matrix = List[List[float]]


# ------------------------------------------------------------ 패턴 생성기
def _check_size(n: int) -> None:
    if isinstance(n, bool) or not isinstance(n, int) or n < 1:
        raise ValueError(f"크기 N은 1 이상의 정수여야 합니다: {n!r}")


def generate_cross(n: int) -> Matrix:
    """가운데 행과 가운데 열이 1인 N×N 십자가 패턴."""
    _check_size(n)
    mid = n // 2
    return [[1 if i == mid or j == mid else 0 for j in range(n)] for i in range(n)]


def generate_x(n: int) -> Matrix:
    """두 대각선이 1인 N×N X 패턴."""
    _check_size(n)
    return [[1 if i == j or i + j == n - 1 else 0 for j in range(n)] for i in range(n)]


# ------------------------------------------------------- 1D 평탄화 MAC
def flatten(matrix: Sequence[Sequence[float]]) -> List[float]:
    return Grid(matrix).flatten()


def mac_1d(flat_pattern: Sequence[float], flat_filter: Sequence[float]) -> float:
    """길이 N² 배열 두 개를 단일 루프로 곱해 누적한다."""
    if len(flat_pattern) != len(flat_filter):
        raise ValueError(f"길이 불일치: {len(flat_pattern)} vs {len(flat_filter)}")
    total = 0.0
    for k in range(len(flat_pattern)):
        total += flat_pattern[k] * flat_filter[k]
    return total


# ------------------------------------------------------------ 성능 비교
def _avg_ms(fn: Callable, a, b, repeat: int) -> float:
    total = 0.0
    for _ in range(repeat):
        start = time.perf_counter()
        fn(a, b)
        total += time.perf_counter() - start
    return total / repeat * 1000.0


def compare_2d_1d(sizes: Sequence[int] = SIZES, repeat: int = REPEAT) -> List[Dict]:
    """동일 입력·동일 반복 횟수로 2D MAC과 1D MAC의 평균 시간(ms)을 비교한다.

    평탄화 변환은 측정 밖에서 한 번만 수행한다 (연산 구간만 비교).
    두 방식의 결과 점수가 다르면 ValueError.
    """
    if repeat < 1:
        raise ValueError("repeat는 1 이상이어야 합니다.")
    rows = []
    for n in sizes:
        pattern, filt = generate_cross(n), generate_x(n)
        fp, ff = flatten(pattern), flatten(filt)
        if mac(pattern, filt) != mac_1d(fp, ff):
            raise ValueError(f"{n}×{n}: 2D/1D 결과가 다릅니다.")
        t2d = _avg_ms(mac, pattern, filt, repeat)
        t1d = _avg_ms(mac_1d, fp, ff, repeat)
        rows.append({"n": n, "ms_2d": t2d, "ms_1d": t1d,
                     "speedup": t2d / t1d if t1d > 0 else float("inf")})
    return rows


def format_compare_table(rows: List[Dict]) -> List[str]:
    lines = [_pad("크기", 8) + _pad("2D(ms)", 12, True) + _pad("1D(ms)", 12, True)
             + _pad("2D/1D", 10, True), "-" * 42]
    for r in rows:
        lines.append(f"{str(r['n']) + '×' + str(r['n']):<8}{r['ms_2d']:>12.4f}"
                     f"{r['ms_1d']:>12.4f}{r['speedup']:>9.2f}x")
    return lines


# ------------------------------------------------------------- 모드 3 흐름
def read_size(input_fn: Callable[[str], str] = input,
              output_fn: Callable[[str], None] = print) -> int:
    while True:
        raw = input_fn("패턴 크기 N (3 이상의 정수): ").strip()
        try:
            n = int(raw)
            if n < 3:
                raise ValueError
            return n
        except ValueError:
            output_fn("입력 오류: 3 이상의 정수를 입력하세요.")


def run_bonus(input_fn: Callable[[str], str] = input,
              output_fn: Callable[[str], None] = print) -> dict:
    out = output_fn
    bar = "#" + "-" * 39
    n = read_size(input_fn, out)
    cross, xpat = generate_cross(n), generate_x(n)

    out(bar)
    out(f"# [1] {n}×{n} 패턴 생성")
    out(bar)
    for name, pat in (("Cross", cross), ("X", xpat)):
        out(f"{name}:")
        for row in pat:
            out("  " + " ".join(str(v) for v in row))

    out(bar)
    out("# [2] 생성 패턴 판정 (생성 필터 사용)")
    out(bar)
    verdicts = {}
    for name, pat in (("Cross", cross), ("X", xpat)):
        sc, sx = mac(pat, cross), mac(pat, xpat)
        verdicts[name] = judge(sc, sx, "Cross", "X")
        out(f"{name} 패턴 -> Cross 점수: {sc!r}, X 점수: {sx!r}, 판정: {verdicts[name]}")

    out(bar)
    out(f"# [3] 2D vs 1D 성능 비교 (평균/{REPEAT}회)")
    out(bar)
    sizes = sorted(set(SIZES) | {n})
    rows = compare_2d_1d(sizes)
    for line in format_compare_table(rows):
        out(line)
    return {"n": n, "verdicts": verdicts, "rows": rows}
