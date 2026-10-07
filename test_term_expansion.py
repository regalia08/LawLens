"""일상용어 -> 법령용어 확장 테스트 (네트워크 없음, 캐시를 직접 넣어 검증).

연계용어 예시는 실제 법제처 dlytrmRlt 응답에서 가져옴.

실행: python -m unittest -v
"""
import unittest
from pathlib import Path

from eval_retrieval import Bm25Retriever, ExpandedBm25Retriever, bigrams
from term_expansion import QueryExpander, TermLexicon

CACHE = {
    "월급": [{"term": "급료", "rel": "상위어"}, {"term": "임금", "rel": "연관어"}],
    "해고": [{"term": "고용", "rel": "반의어"}, {"term": "면직", "rel": "연관어"}],
    "주휴수당": [{"term": "1주에평균1회이상의유급휴일", "rel": "연관어"}],
    "회사원": None,  # 사전에 없는 단어
}
CORPUS = [
    "근로기준법 제43조(임금 지급)\n임금은 통화로 직접 근로자에게 그 전액을 지급하여야 한다.",
    "근로기준법 제55조(휴일)\n사용자는 근로자에게 1주에 평균 1회 이상의 유급휴일을 보장하여야 한다.",
    "근로기준법 제54조(휴게)\n휴게시간은 근로자가 자유롭게 이용할 수 있다.",
    "근로기준법 제23조(해고 등의 제한)\n고용 관계에서 정당한 이유 없이 해고하지 못한다.",
]


def lexicon(cache=CACHE):
    lex = TermLexicon(cache_path=Path("__no_such_cache__.json"), online=False)
    lex.cache = dict(cache)
    return lex


class ExpanderTests(unittest.TestCase):
    def setUp(self):
        self.ex = QueryExpander(lexicon(), CORPUS)

    def test_longest_prefix_strips_josa(self):
        self.assertEqual([w for w, _ in self.ex.matched_words("월급을 못 받았어요")], ["월급"])

    def test_only_terms_present_in_corpus(self):
        # '급료'는 조문에 없으므로 빠지고 '임금'만 남는다
        self.assertEqual(self.ex.expand("월급을 못 받았어요"), ["임금"])

    def test_antonyms_dropped(self):
        # '고용'은 조문에 있지만 해고의 반의어라 확장하지 않는다
        self.assertEqual(self.ex.expand("해고당했어요"), [])

    def test_corpus_match_ignores_spaces(self):
        self.assertEqual(self.ex.expand("주휴수당 조건"), ["1주에평균1회이상의유급휴일"])

    def test_unknown_word_offline_is_skipped(self):
        self.assertEqual(self.ex.expand("회사원 휴게"), [])


class ExpandedRetrieverTests(unittest.TestCase):
    def setUp(self):
        self.chunks = [{"law_name": "근로기준법", "article_label": t.split("(")[0].split()[1], "text": t} for t in CORPUS]
        self.bm25 = Bm25Retriever(self.chunks)
        self.expanded = ExpandedBm25Retriever(self.bm25, QueryExpander(lexicon(), CORPUS), weight=0.5)

    def test_expansion_terms_get_lower_weight(self):
        terms = self.expanded.query_terms("월급")
        self.assertEqual(terms["월급"], 1)
        self.assertEqual(terms["임금"], 0.5)

    def test_expansion_finds_article_plain_bm25_misses(self):
        self.assertEqual(self.bm25("월급", 3), [])
        top = self.expanded("월급", 3)
        self.assertEqual(top[0]["article_label"], "제43조")

    def test_bigrams_of_single_char_word(self):
        self.assertEqual(bigrams("일 임금"), ["일", "임금"])


if __name__ == "__main__":
    unittest.main()
