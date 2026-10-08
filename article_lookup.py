"""조문 번호 직접 조회: 질문에 "제26조", "근기법 23조", "시행령 제30조"처럼 조문 번호가 있으면
그 조문을 검색 결과 맨 앞에 고정하고, 나머지 자리는 기존 검색기 결과로 채운다.

왜 필요한가: "제26조가 무슨 내용이에요?"처럼 번호만 있고 주제어가 없는 질문은 임베딩 검색이 못 찾는다
(dev에서 제21조 31위, 제45조 11위). 번호는 사용자가 근거를 정확히 지정한 것이므로 검색보다 조회가 맞다.

규칙
- 법령 판별: 번호 바로 앞이 "시행령"이면 시행령, "시행규칙"이면 시행규칙, 그 외는 근로기준법.
- 다른 법 이름이 붙은 번호("남녀고용평등법 제14조의2", "기간제법 제4조")는 무시한다.
  근로기준법에도 같은 번호가 있어서, 잘못 고정하면 엉뚱한 근거로 답하게 된다.
- 인덱스에 없는 조문(삭제·만료, 없는 번호)은 고정하지 않는다.
- 근거 개수 k는 그대로다. 고정한 조문이 자리를 차지하는 만큼 기존 검색 결과가 뒤로 밀린다.
- 항 단위로 쪼개진 긴 조문은 청크를 전부 고정하지 않는다 (test t18에서 제74조 청크 10개가 근거 5자리를 독점해
  정답 제74조의2가 밀려난 결함). 항을 지정하면 그 항 청크 하나만, 지정하지 않으면 검색 순위가 높은 청크
  최대 MAX_CHUNKS_PER_ARTICLE개만 고정한다. 2는 "근거 5자리 중 3자리 이상을 검색 결과에 남긴다"는 기준으로
  미리 정한 값이다 (평가셋 점수로 고른 값 아님).
"""
from __future__ import annotations

import re

LAW = "근로기준법"
DECREE = "근로기준법 시행령"
RULE = "근로기준법 시행규칙"
OWN_LAW_WORDS = {"근로기준법", "근기법", "법"}
MAX_CHUNKS_PER_ARTICLE = 2
RANK_DEPTH = 50  # 고정할 청크를 고를 때 참고하는 기존 검색 결과 깊이

# "제 56 조 제3항", "23조", "제43조의2", "제76조의3 제6항"
ARTICLE_RE = re.compile(r"(?<!\d)제?\s*(\d+)\s*조(?:\s*의\s*(\d+))?(?:\s*제?\s*(\d+)\s*항)?")


def law_of(prefix: str) -> str | None:
    """번호 앞 문맥으로 법령을 정한다. 다른 법이면 None."""
    prefix = prefix.rstrip()
    if prefix.endswith("시행령"):
        return DECREE
    if prefix.endswith("시행규칙"):
        return RULE
    last = re.search(r"([가-힣A-Za-z]+)$", prefix)
    if last and last[1].endswith("법") and last[1] not in OWN_LAW_WORDS:
        return None
    return LAW


def parse_refs(question: str) -> list[tuple[str, str, int | None]]:
    """질문 속 조문 참조 -> [(법령명, '제N조(의M)', 항 번호 또는 None)] (등장 순서, 중복 제거)."""
    refs = []
    for m in ARTICLE_RE.finditer(question):
        law = law_of(question[: m.start()])
        if law is None:
            continue
        label = f"제{int(m[1])}조" + (f"의{int(m[2])}" if m[2] else "")
        ref = (law, label, int(m[3]) if m[3] else None)
        if ref not in refs:
            refs.append(ref)
    return refs


class ArticleLookupRetriever:
    def __init__(self, base, chunks: list[dict]):
        self.base = base
        self.by_article: dict[tuple[str, str], list[dict]] = {}
        for c in chunks:
            self.by_article.setdefault((c["law_name"], c["article_label"]), []).append(c)

    def pinned(self, question: str, rank: dict[str, int] | None = None) -> list[dict]:
        """질문이 지정한 조문의 청크. rank: chunk_id -> 기존 검색 순위 (청크가 여러 개인 조문에서 고를 때 사용)."""
        rank = rank or {}
        out = []
        for law, label, para in parse_refs(question):
            found = self.by_article.get((law, label), [])
            if len(found) > 1:
                exact = [c for c in found if para and c.get("paragraph_no") == para]
                if exact:
                    found = exact
                else:
                    by_rank = sorted(found, key=lambda c: (rank.get(c["chunk_id"], len(rank)), c.get("paragraph_no") or 0))
                    found = by_rank[: 1 if para else MAX_CHUNKS_PER_ARTICLE]
            out += [{**c, "score": 1.0, "pinned": True} for c in found if c["chunk_id"] not in {o["chunk_id"] for o in out}]
        return out

    def __call__(self, question: str, k: int) -> list[dict]:
        base = self.base(question, max(k, RANK_DEPTH))
        pinned = self.pinned(question, {c["chunk_id"]: i for i, c in enumerate(base)})
        ids = {c["chunk_id"] for c in pinned}
        return (pinned + [c for c in base if c["chunk_id"] not in ids])[:k]
