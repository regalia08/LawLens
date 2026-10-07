"""국가법령정보센터 Open API로 법령 본문(JSON)을 받아 data/에 저장.

사용:
    python fetch_law.py                                  # 근로기준법 + 시행령 + 시행규칙, 오늘 기준
    python fetch_law.py "근로기준법 시행령" --date 2025-01-01

OC 키는 .env 의 LAW_OC (또는 환경변수 LAW_OC)에서 읽는다. 로그/에러에는 절대 찍지 않는다.

버전 선택: 목록(lawSearch, target=eflaw)은 같은 법령의 연혁/현행/시행예정 버전을
시행일자별로 모두 돌려준다. 기준일에 시행 중인 버전 = 시행일자 <= 기준일 중 가장 최근.
'현행' 코드 대신 날짜로 고르는 이유: 과거 시점 버전도 같은 규칙으로 받을 수 있어서.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.parse
import urllib.request
from datetime import date
from pathlib import Path

BASE_URL = "https://www.law.go.kr/DRF"
DEFAULT_LAWS = ["근로기준법", "근로기준법 시행령", "근로기준법 시행규칙"]
PAGE_SIZE = 100


def load_oc(env_path: Path = Path(".env")) -> str:
    if os.environ.get("LAW_OC"):
        return os.environ["LAW_OC"].strip()
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "LAW_OC":
                return value.strip().strip("'\"")
    sys.exit("LAW_OC가 없습니다. .env 에 LAW_OC=<키> 를 넣으세요.")


def get_json(endpoint: str, params: dict, oc: str) -> dict:
    url = f"{BASE_URL}/{endpoint}?" + urllib.parse.urlencode({"OC": oc, "type": "JSON", **params})
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            raw = resp.read().decode("utf-8")
    except Exception as e:  # URL에 OC가 들어 있으므로 메시지에서 가린다
        raise RuntimeError(f"{endpoint} 요청 실패: {str(e).replace(oc, '<OC>')}") from None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        # 미등록 IP 등은 HTML 안내 페이지로 온다
        raise RuntimeError(f"{endpoint} JSON 아님: {raw[:200].replace(oc, '<OC>')}") from None


def normalize_name(name: str) -> str:
    return "".join(name.split())  # '근로기준법시행규칙'(구 명칭)과 '근로기준법 시행규칙'을 같게 본다


def search_versions(name: str, oc: str) -> list[dict]:
    rows, page = [], 1
    while True:
        res = get_json(
            "lawSearch.do",
            {"target": "eflaw", "query": name, "display": PAGE_SIZE, "page": page},
            oc,
        ).get("LawSearch", {})
        if res.get("resultCode") not in (None, "00"):
            raise RuntimeError(f"lawSearch 오류: {res.get('resultCode')} / {res.get('resultMsg')}")
        batch = res.get("law") or []
        rows.extend(batch if isinstance(batch, list) else [batch])
        if page * PAGE_SIZE >= int(res.get("totalCnt") or 0) or not batch:
            return rows
        page += 1


def pick_version(rows: list[dict], name: str, on: date) -> dict | None:
    """이름이 정확히 같고(공백 무시) 시행일자 <= 기준일인 버전 중 가장 최근 것.

    시행일자가 같으면 공포일자, 법령일련번호가 큰 쪽(나중 개정)을 고른다.
    """
    target = normalize_name(name)
    cutoff = on.strftime("%Y%m%d")
    candidates = [
        r
        for r in rows
        if normalize_name(r.get("법령명한글", "")) == target and r.get("시행일자", "") <= cutoff
    ]
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (r["시행일자"], r.get("공포일자", ""), int(r.get("법령일련번호") or 0)),
    )


def fetch_body(row: dict, oc: str) -> dict:
    doc = get_json(
        "lawService.do",
        {"target": "eflaw", "MST": row["법령일련번호"], "efYd": row["시행일자"]},
        oc,
    )
    if "법령" not in doc:
        raise RuntimeError(f"lawService 오류 응답: {doc.get('result')} / {doc.get('msg')}")
    return doc


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("names", nargs="*", default=DEFAULT_LAWS, help="법령명 (기본: 근로기준법 3종)")
    ap.add_argument("--date", help="시행 기준일 (YYYY-MM-DD, 기본: 오늘)")
    ap.add_argument("--out", default="data")
    args = ap.parse_args(argv)

    on = date.fromisoformat(args.date) if args.date else date.today()
    oc = load_oc()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for name in args.names:
        row = pick_version(search_versions(name, oc), name, on)
        if row is None:
            print(f"[건너뜀] {name}: {on} 기준 시행 중인 버전 없음")
            continue
        doc = fetch_body(row, oc)
        path = out / f"{normalize_name(name)}_{row['시행일자']}.json"
        path.write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
        print(
            f"{row['법령명한글']} | 법령ID {row['법령ID']} | MST {row['법령일련번호']} | "
            f"시행 {row['시행일자']} ({row.get('현행연혁코드')}) -> {path}"
        )


if __name__ == "__main__":
    main()
