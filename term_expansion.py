"""일상용어 -> 법령용어 쿼리 확장 (법제처 지능형 법령정보지식베이스: 일상용어-법령용어 연계).

질문 "월급을 못 받았어요" -> 어절 '월급을' -> 사전에 있는 가장 긴 앞부분 '월급'
-> 연계 법령용어 [급료, 급여채권, ...] -> 이 중 우리 조문에 실제로 나오는 것만 검색어에 추가.

설계 메모
- 사전에서 무엇을 찾을지 사람이 고르지 않는다. 평가셋 오답을 보고 동의어를 손으로 넣으면
  평가셋에 과적합되어 숫자가 부풀려지므로, 질문 단어를 그대로 API에 조회한다 (실서비스와 같은 경로).
- 조사 제거: 형태소 분석기 없이, 어절의 앞부분 중 사전에 있는 가장 긴 것(2자 이상)을 쓴다.
- 반의어(해고 -> 고용)는 버린다. 확장어는 조문(공백 제거)에 실제로 나오는 것만 남긴다.
- API 응답은 data/terms/dlytrm_cache.json 에 캐시해 커밋한다. 평가는 캐시만으로 재현된다 (OC 불필요).
"""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

CACHE_PATH = Path("data/terms/dlytrm_cache.json")
DROP_RELATIONS = {"반의어"}
MIN_LEN = 2


def eojeols(query: str) -> list[str]:
    return re.findall(r"[가-힣]{%d,}" % MIN_LEN, query)


def prefixes(word: str) -> list[str]:
    """'월급을' -> ['월급을', '월급'] (긴 것부터)."""
    return [word[:n] for n in range(len(word), MIN_LEN - 1, -1)]


def nospace(text: str) -> str:
    return re.sub(r"\s+", "", text)


class TermLexicon:
    """일상용어 -> [(법령용어, 관계)] 조회. 캐시 우선, online=True 이면 없는 단어를 API로 채운다.

    캐시 값: 단어가 사전에 없으면 null, 있으면 연계용어 목록 (빈 목록일 수 있음).
    """

    def __init__(self, cache_path: Path = CACHE_PATH, online: bool = False, oc: str | None = None):
        self.cache_path = cache_path
        self.online = online
        self.oc = oc
        self.cache: dict[str, list[dict] | None] = (
            json.loads(cache_path.read_text(encoding="utf-8")) if cache_path.exists() else {}
        )
        self.dirty = False

    def lookup(self, word: str) -> list[dict] | None:
        if word in self.cache:
            return self.cache[word]
        if not self.online:
            return None
        try:
            value = self._fetch(word)
        except RuntimeError as e:  # 일시 오류는 캐시하지 않고 다음에 다시 시도
            print(f"[용어 조회 실패] {word}: {e}")
            return None
        self.cache[word] = value
        self.dirty = True
        return value

    def _fetch(self, word: str) -> list[dict] | None:
        from fetch_law import get_json  # 오프라인 평가/테스트에서는 네트워크 모듈이 필요 없도록 지연 import

        mst = None
        for page in range(1, 6):  # '임금'처럼 부분일치가 많으면 정확일치 항목이 뒤 페이지에 있다
            res = get_json("lawSearch.do", {"target": "dlytrm", "query": word, "display": 100, "page": page}, self.oc)
            res = res.get("dlytrmSearch", {})
            items = res.get("일상용어") or []
            items = items if isinstance(items, list) else [items]
            hit = next((i for i in items if i.get("일상용어명") == word), None)
            if hit:
                mst = hit["용어간관계링크"].split("MST=")[1]
                break
            if page * 100 >= int(res.get("검색결과개수") or 0):
                break
            time.sleep(0.1)
        if mst is None:
            return None
        res = get_json("lawService.do", {"target": "dlytrmRlt", "MST": mst}, self.oc)
        linked = (res.get("dlytrmRltService", {}).get("일상용어") or {}).get("연계용어") or []
        linked = linked if isinstance(linked, list) else [linked]
        time.sleep(0.1)
        return [{"term": t["법령용어명"], "rel": t["용어관계"]} for t in linked if t.get("법령용어명")]

    def save(self) -> None:
        if self.dirty:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            self.cache_path.write_text(
                json.dumps(dict(sorted(self.cache.items())), ensure_ascii=False, indent=1), encoding="utf-8"
            )
            self.dirty = False


class QueryExpander:
    def __init__(self, lexicon: TermLexicon, corpus_texts: list[str]):
        self.lexicon = lexicon
        self.corpus = "\n".join(nospace(t) for t in corpus_texts)

    def matched_words(self, query: str) -> list[tuple[str, list[dict]]]:
        """어절마다 사전에 있는 가장 긴 앞부분과 그 연계용어."""
        out = []
        for w in eojeols(query):
            for p in prefixes(w):
                linked = self.lexicon.lookup(p)
                if linked is not None:
                    out.append((p, linked))
                    break
        return out

    def expand(self, query: str) -> list[str]:
        """질문에 덧붙일 법령용어 (조문에 실제로 나오는 것만, 중복 제거, 등장 순서 유지)."""
        terms = []
        for _, linked in self.matched_words(query):
            for t in linked:
                if t["rel"] in DROP_RELATIONS:
                    continue
                if t["term"] not in terms and nospace(t["term"]) in self.corpus:
                    terms.append(t["term"])
        return terms
