"""콘솔 입력 파싱/검증 (Phase 2).

표준 라이브러리만 사용한다. input_fn을 주입할 수 있어 테스트가 쉽다.
"""

import math
from typing import Callable, List

Matrix = List[List[float]]


def parse_row(line: str, n: int) -> List[float]:
    """공백으로 구분된 n개의 숫자 한 줄을 파싱한다. 실패 시 ValueError."""
    tokens = line.split()
    if len(tokens) != n:
        raise ValueError(
            f"입력 형식 오류: 각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요. "
            f"(입력된 개수: {len(tokens)})"
        )
    row = []
    for tok in tokens:
        try:
            value = float(tok)
        except ValueError:
            raise ValueError(
                f"입력 형식 오류: '{tok}' 은(는) 숫자가 아닙니다. "
                f"각 줄에 {n}개의 숫자를 공백으로 구분해 입력하세요."
            ) from None
        if not math.isfinite(value):
            raise ValueError(f"입력 형식 오류: '{tok}' 은(는) 유한한 숫자가 아닙니다.")
        row.append(value)
    return row


def read_matrix(title: str, n: int = 3,
                input_fn: Callable[[str], str] = input,
                output_fn: Callable[[str], None] = print) -> Matrix:
    """n줄을 입력받아 n×n 행렬을 반환한다.

    줄 단위로 검증하며, 오류가 나면 안내 후 해당 줄부터 다시 입력받는다.
    줄 수가 모자라거나 남는 경우는 줄 단위 재입력으로 자연스럽게 방지된다
    (정확히 n줄을 받을 때까지 반복). EOF는 호출자에게 전파한다.
    """
    output_fn(f"{title} ({n}줄 입력, 공백 구분)")
    matrix: Matrix = []
    while len(matrix) < n:
        line = input_fn("")
        try:
            matrix.append(parse_row(line, n))
        except ValueError as e:
            output_fn(f"  {e}  [{len(matrix) + 1}번째 줄부터 다시 입력]")
    return matrix
