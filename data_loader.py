"""콘솔 입력 파싱/검증 (Phase 2).

표준 라이브러리만 사용한다. input_fn을 주입할 수 있어 테스트가 쉽다.
"""

import json
import math
import re
from typing import Callable, Dict, List, Optional, Tuple

from npu_simulator import (LABEL_CROSS, LABEL_X, judge, mac, normalize_label,
                           validate_matrix)

_FILTER_KEY_RE = re.compile(r"^size_(\d+)$")
_PATTERN_KEY_RE = re.compile(r"^size_(\d+)_(\d+)$")

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


# ------------------------------------------------------------ data.json (Phase 3)
def load_json(path: str) -> dict:
    """data.json을 읽는다. 파일/JSON/최상위 구조 오류는 ValueError로 통일."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise ValueError(f"파일을 찾을 수 없습니다: {path}") from None
    except (json.JSONDecodeError, UnicodeDecodeError) as e:
        raise ValueError(f"JSON 파싱 실패: {e}") from None
    except OSError as e:
        raise ValueError(f"파일을 읽을 수 없습니다: {e}") from None
    if not isinstance(data, dict):
        raise ValueError("최상위 구조가 객체가 아닙니다.")
    for key in ("filters", "patterns"):
        if not isinstance(data.get(key), dict):
            raise ValueError(f"'{key}' 객체가 없거나 형식이 올바르지 않습니다.")
    return data


def parse_filters(raw_filters: dict) -> Tuple[Dict[int, Dict[str, list]], Dict[int, str]]:
    """filters를 {N: {Cross: 행렬, X: 행렬}}로 변환한다 (필터 키 라벨 정규화).

    문제가 있는 size_N은 결과에서 빼고 {N: 사유}로 따로 돌려준다.
    """
    filters: Dict[int, Dict[str, list]] = {}
    problems: Dict[int, str] = {}
    for key, entry in raw_filters.items():
        m = _FILTER_KEY_RE.match(key)
        if not m:
            continue  # size_N 형식이 아닌 키는 무시
        n = int(m.group(1))
        try:
            if not isinstance(entry, dict):
                raise ValueError("필터 항목이 객체가 아닙니다.")
            by_label: Dict[str, list] = {}
            for raw_label, matrix in entry.items():
                label = normalize_label(raw_label)
                validate_matrix(matrix, n)
                by_label[label] = matrix
            for needed in (LABEL_CROSS, LABEL_X):
                if needed not in by_label:
                    raise ValueError(f"{needed} 필터가 없습니다.")
            filters[n] = by_label
        except ValueError as e:
            problems[n] = f"필터 {key} 오류: {e}"
    return filters, problems


def evaluate_pattern(case_id: str, entry, filters: Dict[int, Dict[str, list]],
                     filter_problems: Optional[Dict[int, str]] = None) -> dict:
    """패턴 1건을 검증/판정한다. 어떤 오류도 예외로 새지 않고 FAIL 결과로 반환."""
    result = {"id": case_id, "status": "FAIL", "verdict": None, "expected": None,
              "score_cross": None, "score_x": None, "reason": ""}
    try:
        m = _PATTERN_KEY_RE.match(case_id)
        if not m:
            raise ValueError("키 형식 오류 (size_{N}_{idx} 필요)")
        n = int(m.group(1))
        if not isinstance(entry, dict):
            raise ValueError("패턴 항목이 객체가 아닙니다.")
        if "input" not in entry or "expected" not in entry:
            raise ValueError("input 또는 expected 누락")
        if n in (filter_problems or {}):
            raise ValueError(filter_problems[n])
        if n not in filters:
            raise ValueError(f"size_{n} 필터가 없습니다.")
        expected = normalize_label(entry["expected"])
        result["expected"] = expected
        validate_matrix(entry["input"], n)  # 키의 N과 입력 크기 일치 검증

        score_c = mac(entry["input"], filters[n][LABEL_CROSS])
        score_x = mac(entry["input"], filters[n][LABEL_X])
        verdict = judge(score_c, score_x, LABEL_CROSS, LABEL_X)
        result.update(score_cross=score_c, score_x=score_x, verdict=verdict)
        if verdict == expected:
            result["status"] = "PASS"
        elif verdict == "UNDECIDED":
            result["reason"] = "동점(UNDECIDED) 규칙에 따라 FAIL"
        else:
            result["reason"] = f"판정 {verdict} != expected {expected}"
    except ValueError as e:
        result["reason"] = str(e)
    except Exception as e:  # 예상 밖 오류도 케이스 단위로 격리
        result["reason"] = f"처리 중 예외: {type(e).__name__}: {e}"
    return result


def analyze_all(data: dict) -> Tuple[Dict[int, Dict[str, list]], Dict[int, str], List[dict]]:
    filters, problems = parse_filters(data["filters"])
    results = [evaluate_pattern(k, v, filters, problems)
               for k, v in data["patterns"].items()]
    return filters, problems, results
