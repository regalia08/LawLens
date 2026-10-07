"""국가법령정보센터 lawService(type=JSON) 응답 -> 조문 정규화 / 청킹.

사용:
    python law_parser.py 근로기준법.json --today 2026-10-07 --out out

출력:
    out/articles.jsonl  정규화된 조문 (삭제/만료 상태 포함)
    out/chunks.jsonl    임베딩 대상 청크 (삭제/만료 제외)
    콘솔: 통계 + 만료 검토 대상 목록

MVP에서는 부칙, 개정문, 제개정이유, 별표를 읽지 않는다 (조문만 인덱싱).
"""
from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass, field
from datetime import date
from pathlib import Path

CIRCLED = "①②③④⑤⑥⑦⑧⑨⑩⑪⑫⑬⑭⑮⑯⑰⑱⑲⑳"

# <개정 2018.3.20, 2019.1.15>, <신설 2012.2.1>, <삭제 2019.1.15>, <2019.1.15> 처럼 날짜가 든 꺾쇠 태그
TAG_RE = re.compile(r"<[^<>]*\d{4}\.\d{1,2}\.\d{1,2}[^<>]*>")
DELETED_ARTICLE_RE = re.compile(r"^제\d+조(?:의\d+)?\s*삭제$")
# "2022년 12월 31일까지 유효함" 같은 조문참고자료 문구
EXPIRY_RE = re.compile(r"(\d{4})년\s*(\d{1,2})월\s*(\d{1,2})일까지\s*(?:유효|효력)")
# 본문 맨 앞의 "제43조의2(제목)" 접두 제거용
LABEL_PREFIX_RE = re.compile(r"^제\d+조(?:의\d+)?(?:\([^)]*\))?\s*")


# ---------------------------------------------------------------- 데이터 모델
@dataclass
class Item:
    no: str
    text: str
    children: list[Item] = field(default_factory=list)


@dataclass
class Paragraph:
    no: str | None  # "①" (호만 있는 항은 None)
    number: int | None
    text: str
    items: list[Item]
    amendments: list[str]
    deleted: bool = False
    expired: bool = False


@dataclass
class Article:
    key: str  # 조문키 (예: 0043021)
    label: str  # "제43조의2"
    number: int
    branch: int | None
    title: str
    chapter: str | None
    head: str  # 정제된 조문내용 (항이 없으면 본문 전체)
    text: str  # 삭제 항을 뺀 본문 전체
    paragraphs: list[Paragraph]
    amendments: list[str]
    effective_date: str
    changed_in_version: bool
    reference_note: str
    status: str  # valid | deleted | expired | partially_expired
    expiry: dict | None


# ---------------------------------------------------------------- 유틸
def as_list(value) -> list:
    """단일 객체로 오기도 하고 배열로 오기도 하는 필드를 항상 리스트로."""
    if value is None or value == "":
        return []
    return value if isinstance(value, list) else [value]


def flatten_text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return "\n".join(t for t in (flatten_text(v) for v in value) if t)
    return str(value)


def clean(text: str) -> str:
    text = TAG_RE.sub("", text)
    text = re.sub(r"[ \t\u3000]+", " ", text)
    text = re.sub(r"\s*\n\s*", "\n", text)
    return text.strip()


def extract_tags(text: str) -> list[str]:
    return [t[1:-1].strip() for t in TAG_RE.findall(text)]


def circled_to_int(label: str | None) -> int | None:
    if not label:
        return None
    label = label.strip()
    if label and label[0] in CIRCLED:
        return CIRCLED.index(label[0]) + 1
    m = re.match(r"\d+", label)
    return int(m.group()) if m else None


def to_int(value) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


def strip_label(head: str) -> str:
    return LABEL_PREFIX_RE.sub("", head, count=1).strip()


# ---------------------------------------------------------------- 항/호 파싱
def parse_items(ho_value) -> list[Item]:
    items = []
    for h in as_list(ho_value):
        if not isinstance(h, dict):
            continue
        children = []
        for m in as_list(h.get("목")):
            if isinstance(m, dict):
                children.append(
                    Item(
                        no=str(m.get("목번호", "")).strip(),
                        text=clean(flatten_text(m.get("목내용"))),
                    )
                )
        items.append(
            Item(
                no=str(h.get("호번호", "")).strip(),  # "9의2." 같은 값이 있어 문자열로 유지
                text=clean(flatten_text(h.get("호내용"))),
                children=children,
            )
        )
    return items


def parse_paragraphs(hang_value) -> list[Paragraph]:
    """항이 배열 / 단일 객체 / {"호": [...]} (항번호 없음) 어느 형태로 와도 처리."""
    paragraphs = []
    for h in as_list(hang_value):
        if not isinstance(h, dict):
            continue
        raw = flatten_text(h.get("항내용"))
        text = clean(raw)
        no = h.get("항번호")
        no = str(no).strip() if no else None
        body = re.sub(rf"^[{CIRCLED}]\s*", "", text)  # 항내용에 번호가 이미 들어 있음
        paragraphs.append(
            Paragraph(
                no=no,
                number=circled_to_int(no),
                text=text,
                items=parse_items(h.get("호")),
                amendments=extract_tags(raw),
                deleted=(body == "삭제"),
            )
        )
    return paragraphs


def build_text(head: str, paragraphs: list[Paragraph]) -> str:
    lines = [head] if head else []
    for p in paragraphs:
        if p.deleted:
            continue
        if p.text:
            lines.append(p.text)
        for it in p.items:
            lines.append(it.text)
            lines.extend(c.text for c in it.children)
    return "\n".join(lines)


# ---------------------------------------------------------------- 만료 판정
def self_ref_re(number: int, branch: int | None) -> re.Pattern:
    """조문참고자료에서 '이 조문'을 가리키는 참조 + 뒤따르는 항/호.

    - 제43조는 제43조의2와 구별한다 (가지번호 유무).
    - '법 제53조'(시행령이 모법을 가리킴), '부칙 제2조'는 이 조가 아니다.
    - '이 조 제1항제8호'처럼 '이 조'로 가리키기도 한다.
    """
    label = rf"제{number}조" + (rf"의{branch}(?!\d)" if branch else r"(?!의\d)")
    return re.compile(
        rf"(?<!법 )(?<!법)(?<!부칙 )(?<!부칙)(?:{label}|이\s*조)"
        r"(?:\s*제(\d+)항)?(\s*제\d+호)?"
    )


def parse_expiry(note: str, number: int, branch: int | None, today: date) -> dict | None:
    """조문참고자료의 'YYYY년 M월 D일까지 유효' 문구를 해석.

    scope (만료되는 범위):
      article    '제16조의 개정규정은 ...'                       -> 조 전체 만료
      paragraphs '제53조제3항, 제53조제6항의 개정규정은 ...'     -> 해당 항만 만료
      partial    '이 조 제1호의 개정규정 중 ... 부분은 ...'      -> 호/문구 일부. 지우지 않음
      unknown    이 조를 가리키는 참조를 못 찾음                -> 지우지 않음
    잘못 지우면 살아 있는 조문이 검색에서 사라지므로, 애매하면 남기는 쪽(partial/unknown)으로 판정한다.
    휴리스틱이므로 결과는 반드시 법제처 원문과 대조해 검토할 것.
    """
    m = EXPIRY_RE.search(note)
    if not m:
        return None
    until = date(int(m[1]), int(m[2]), int(m[3]))
    refs = list(self_ref_re(number, branch).finditer(note))
    paragraphs = sorted({int(r[1]) for r in refs if r[1]})
    if not refs:
        scope = "unknown"
    elif any(r[2] for r in refs) or "부분" in note:
        scope = "partial"
    elif paragraphs:
        scope = "paragraphs"
    else:
        scope = "article"
    return {
        "until": until.isoformat(),
        "scope": scope,
        "paragraphs": paragraphs,
        "expired": until < today,
        "note": note,
    }


# ---------------------------------------------------------------- 법령 파싱
def parse_law(doc: dict, today: date | None = None) -> tuple[dict, list[Article]]:
    today = today or date.today()
    root = doc.get("법령", doc)
    if "조문" not in root:
        if "result" in doc or "msg" in doc:
            raise RuntimeError(f"API 오류 응답: {doc.get('result')} / {doc.get('msg')}")
        raise ValueError("응답에 '조문' 키가 없습니다 (본문 조회 응답이 맞는지 확인)")

    base = root.get("기본정보", {})
    ministry = base.get("소관부처")
    if isinstance(ministry, dict):
        ministry = ministry.get("content")
    law = {
        "law_id": base.get("법령ID"),
        "name": base.get("법령명_한글"),
        "law_key": root.get("법령키"),
        "effective_date": base.get("시행일자"),
        "promulgation_date": base.get("공포일자"),
        "amendment_type": base.get("제개정구분"),
        "ministry": ministry,
    }

    articles: list[Article] = []
    chapter: str | None = None
    for unit in as_list((root.get("조문") or {}).get("조문단위")):
        content_raw = flatten_text(unit.get("조문내용"))

        # '전문' 행 = 장/절 제목. 조가 아니므로 청크로 만들지 않고 이후 조의 chapter로만 쓴다.
        if unit.get("조문여부") == "전문":
            chapter = clean(content_raw) or chapter
            continue

        number = to_int(unit.get("조문번호"))
        if number is None:
            continue
        branch = to_int(unit.get("조문가지번호"))
        label = f"제{number}조" + (f"의{branch}" if branch else "")

        head = strip_label(clean(content_raw))
        full_head = clean(content_raw)
        paragraphs = parse_paragraphs(unit.get("항"))
        note = clean(flatten_text(unit.get("조문참고자료")))

        status = "deleted" if DELETED_ARTICLE_RE.match(full_head) else "valid"
        expiry = parse_expiry(note, number, branch, today) if note else None
        if status == "valid" and expiry and expiry["expired"]:
            if expiry["scope"] == "article":
                status = "expired"
            elif expiry["scope"] == "paragraphs":
                targets = set(expiry["paragraphs"])
                hit = False
                for p in paragraphs:
                    if p.number in targets:
                        p.expired = True
                        hit = True
                status = "partially_expired" if hit else "valid"
            # partial / unknown: 조문은 그대로 두고 검토 목록에만 올린다

        amendments = extract_tags(content_raw)
        for p in paragraphs:
            amendments.extend(p.amendments)
        amendments = list(dict.fromkeys(amendments))

        articles.append(
            Article(
                key=str(unit.get("조문키", "")),
                label=label,
                number=number,
                branch=branch,
                title=str(unit.get("조문제목") or ""),
                chapter=chapter,
                head=head,
                text=build_text(head, paragraphs),
                paragraphs=paragraphs,
                amendments=amendments,
                effective_date=str(unit.get("조문시행일자") or ""),
                changed_in_version=(unit.get("조문변경여부") == "Y"),
                reference_note=note,
                status=status,
                expiry=expiry,
            )
        )
    return law, articles


# ---------------------------------------------------------------- 청킹
def chunk_article(law: dict, art: Article, max_chars: int = 1000) -> list[dict]:
    """조 단위가 기본. 길이가 max_chars를 넘고 살아있는 항이 2개 이상이면 항 단위로 쪼갠다.

    삭제/만료된 조·항은 청크에서 제외한다. 모든 청크는 '법령명 제N조(제목)' 헤더로 시작한다.
    """
    if art.status in ("deleted", "expired"):
        return []
    live = [p for p in art.paragraphs if not p.deleted and not p.expired]
    if art.paragraphs and not live:
        return []

    header = f"{law['name']} {art.label}" + (f"({art.title})" if art.title else "")
    base = {
        "law_id": law["law_id"],
        "law_name": law["name"],
        "law_key": law["law_key"],
        "article_key": art.key,
        "article_label": art.label,
        "title": art.title,
        "chapter": art.chapter,
        "status": art.status,
        "effective_date": art.effective_date,
    }

    whole = build_text(art.head, live)
    if len(header) + len(whole) <= max_chars or len(live) <= 1:
        return [
            {
                **base,
                "chunk_id": f"{law['law_id']}:{art.key}",
                "paragraph_no": None,
                "citation": header,
                "text": f"{header}\n{whole}".strip(),
            }
        ]

    chunks = []
    for p in live:
        part = build_text(art.head, [p])
        cite = header + (f" 제{p.number}항" if p.number else "")
        chunks.append(
            {
                **base,
                "chunk_id": f"{law['law_id']}:{art.key}:p{p.number or len(chunks) + 1}",
                "paragraph_no": p.number,
                "citation": cite,
                "text": f"{cite}\n{part}".strip(),
            }
        )
    return chunks


# ---------------------------------------------------------------- 통계
def summarize(articles: list[Article]) -> dict:
    lens = sorted(len(a.text) for a in articles if a.status != "deleted")
    stats = {
        "articles": len(articles),
        "valid": sum(a.status == "valid" for a in articles),
        "deleted": sum(a.status == "deleted" for a in articles),
        "expired": sum(a.status == "expired" for a in articles),
        "partially_expired": sum(a.status == "partially_expired" for a in articles),
        "expiry_notes": sum(a.expiry is not None for a in articles),
        "with_branch": sum(a.branch is not None for a in articles),
        "chapters": len({a.chapter for a in articles if a.chapter}),
    }
    if lens:
        stats["chars"] = {
            "min": lens[0],
            "median": lens[len(lens) // 2],
            "p90": lens[int(0.9 * (len(lens) - 1))],
            "max": lens[-1],
            "over_1000": sum(n > 1000 for n in lens),
        }
    return stats


# ---------------------------------------------------------------- CLI
def write_jsonl(path: Path, rows) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("input", help="lawService 본문 조회 JSON 파일")
    ap.add_argument("--today", help="만료 판정 기준일 (YYYY-MM-DD, 기본: 오늘)")
    ap.add_argument("--max-chars", type=int, default=1000)
    ap.add_argument("--out", default="out")
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    doc = json.loads(Path(args.input).read_text(encoding="utf-8"))
    law, articles = parse_law(doc, today)
    chunks = [c for a in articles for c in chunk_article(law, a, args.max_chars)]

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    write_jsonl(out / "articles.jsonl", (asdict(a) for a in articles))
    write_jsonl(out / "chunks.jsonl", chunks)

    print(f"{law['name']} (시행 {law['effective_date']}, 기준일 {today})")
    print(json.dumps(summarize(articles), ensure_ascii=False, indent=2))
    print(f"chunks: {len(chunks)}")
    review = [a for a in articles if a.expiry]
    if review:
        print("\n[만료 문구 검토 대상 - 법제처 원문과 대조 필요]")
        for a in review:
            e = a.expiry
            scope = {
                "article": "조 전체",
                "paragraphs": "항 " + ",".join(map(str, e["paragraphs"])),
                "partial": "호/문구 일부 (조문 유지)",
                "unknown": "대상 불명 (조문 유지)",
            }[e["scope"]]
            print(f"  {a.label}: ~{e['until']} / 대상: {scope} -> {a.status}")


if __name__ == "__main__":
    main()
