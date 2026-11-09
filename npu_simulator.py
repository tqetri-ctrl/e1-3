"""Mini NPU Simulator - 핵심 로직 (Phase 1)

데이터 구조 검증, MAC 연산, 라벨 정규화, epsilon 기반 점수 비교.
표준 라이브러리만 사용한다.
"""

from typing import List, Optional, Sequence

EPSILON = 1e-9

LABEL_CROSS = "Cross"
LABEL_X = "X"
LABEL_UNDECIDED = "UNDECIDED"

Matrix = List[List[float]]

# 정규화 표 (키는 소문자/공백 제거 후 조회)
_LABEL_MAP = {
    "+": LABEL_CROSS,
    "cross": LABEL_CROSS,
    "x": LABEL_X,
}


# ---------------------------------------------------------------- 데이터 구조
def validate_matrix(matrix: Sequence[Sequence[float]], n: Optional[int] = None) -> int:
    """n×n 정방 행렬인지 검증하고 크기 n을 반환한다.

    n을 주면 해당 크기와도 일치해야 한다. 위반 시 ValueError.
    """
    if not isinstance(matrix, (list, tuple)) or len(matrix) == 0:
        raise ValueError("행렬이 비어 있거나 2차원 배열이 아닙니다.")
    rows = len(matrix)
    for i, row in enumerate(matrix):
        if not isinstance(row, (list, tuple)):
            raise ValueError(f"{i + 1}행이 배열이 아닙니다.")
        if len(row) != rows:
            raise ValueError(
                f"행/열 불일치: {rows}행 행렬인데 {i + 1}행의 열 수가 {len(row)}입니다."
            )
        for j, v in enumerate(row):
            # bool은 int의 하위 타입이므로 명시적으로 제외
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                raise ValueError(f"({i + 1},{j + 1}) 값이 숫자가 아닙니다: {v!r}")
    if n is not None and rows != n:
        raise ValueError(f"크기 불일치: 기대 {n}×{n}, 실제 {rows}×{rows}")
    return rows


class Grid:
    """n×n 2차원 숫자 배열 (패턴/필터 공용)."""

    def __init__(self, data: Sequence[Sequence[float]]):
        self.n = validate_matrix(data)
        self._data: Matrix = [list(row) for row in data]

    @classmethod
    def zeros(cls, n: int) -> "Grid":
        return cls([[0] * n for _ in range(n)])

    def get(self, row: int, col: int) -> float:
        return self._data[row][col]

    def set(self, row: int, col: int, value: float) -> None:
        self._data[row][col] = value

    def to_list(self) -> Matrix:
        return [list(row) for row in self._data]

    def flatten(self) -> List[float]:
        return [v for row in self._data for v in row]


# ------------------------------------------------------------------ MAC 연산
def mac(pattern: Sequence[Sequence[float]], filt: Sequence[Sequence[float]]) -> float:
    """위치별 곱셈 후 누적 합산 (이중 for 루프, 외부 라이브러리 금지).

    두 행렬의 크기가 다르면 ValueError.
    """
    n = len(pattern)
    if len(filt) != n:
        raise ValueError(f"크기 불일치: 패턴 {n}행, 필터 {len(filt)}행")
    total = 0.0
    for i in range(n):
        if len(pattern[i]) != n or len(filt[i]) != n:
            raise ValueError(f"{i + 1}행의 열 수가 {n}이 아닙니다.")
        for j in range(n):
            total += pattern[i][j] * filt[i][j]
    return total


# ------------------------------------------------------------- 라벨 정규화
def normalize_label(raw) -> str:
    """'+', 'cross', 'Cross', 'x', 'X' 등을 표준 라벨(Cross/X)로 변환한다.

    알 수 없는 값이면 ValueError.
    """
    if not isinstance(raw, str):
        raise ValueError(f"라벨이 문자열이 아닙니다: {raw!r}")
    key = raw.strip().lower()
    if key not in _LABEL_MAP:
        raise ValueError(f"알 수 없는 라벨: {raw!r}")
    return _LABEL_MAP[key]


# ---------------------------------------------------------- epsilon 비교/판정
def is_tie(a: float, b: float, epsilon: float = EPSILON) -> bool:
    return abs(a - b) < epsilon


def judge(score_a: float, score_b: float, label_a: str = "A", label_b: str = "B",
          epsilon: float = EPSILON) -> str:
    """두 점수를 비교해 label_a / label_b / UNDECIDED 중 하나를 반환한다.

    모드 1: judge(sa, sb) -> 'A' | 'B' | 'UNDECIDED'
    모드 2: judge(sc, sx, LABEL_CROSS, LABEL_X) -> 'Cross' | 'X' | 'UNDECIDED'
    """
    if is_tie(score_a, score_b, epsilon):
        return LABEL_UNDECIDED
    return label_a if score_a > score_b else label_b
