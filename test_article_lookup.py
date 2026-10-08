"""조문 번호 직접 조회 테스트 (모델 없이 가짜 검색기 사용).

실행: python -m unittest -v
"""
import unittest

from article_lookup import DECREE, LAW, RULE, ArticleLookupRetriever, parse_refs


class ParseRefsTests(unittest.TestCase):
    def test_basic_and_variants(self):
        self.assertEqual(parse_refs("근로기준법 제60조는 어떤 내용이에요?"), [(LAW, "제60조", None)])
        self.assertEqual(parse_refs("근기법 23조 내용 알려주세요"), [(LAW, "제23조", None)])
        self.assertEqual(parse_refs("제 56 조 제3항에서 말하는 야간근로"), [(LAW, "제56조", 3)])

    def test_branch_article_and_paragraph(self):
        self.assertEqual(parse_refs("제43조의2에 따르면"), [(LAW, "제43조의2", None)])
        self.assertEqual(parse_refs("제76조의3 제6항은"), [(LAW, "제76조의3", 6)])

    def test_decree_and_rule(self):
        self.assertEqual(parse_refs("근로기준법 시행령 제30조가"), [(DECREE, "제30조", None)])
        self.assertEqual(parse_refs("시행령 제6조의 통상임금"), [(DECREE, "제6조", None)])
        self.assertEqual(parse_refs("근로기준법 시행규칙 제4조는"), [(RULE, "제4조", None)])

    def test_mixed_refs_in_order(self):
        self.assertEqual(
            parse_refs("근로기준법 제17조랑 시행령 제8조에서"), [(LAW, "제17조", None), (DECREE, "제8조", None)]
        )

    def test_other_law_is_ignored(self):
        self.assertEqual(parse_refs("남녀고용평등법 제14조의2에 따라"), [])
        self.assertEqual(parse_refs("기간제법 제4조에서"), [])

    def test_no_false_positive(self):
        for q in ["주 52시간제가 뭐예요?", "제3자에게 알려도 되나요?", "100분의 70 이상", "월급이 밀렸어요"]:
            with self.subTest(q):
                self.assertEqual(parse_refs(q), [])


def chunk(law, art, para=None, text=""):
    cid = f"{law}:{art}" + (f":p{para}" if para else "")
    return {"chunk_id": cid, "law_name": law, "article_label": art, "paragraph_no": para, "text": text}


CHUNKS = [
    chunk(LAW, "제21조"),
    chunk(LAW, "제23조", 1),
    chunk(LAW, "제23조", 2),
    chunk(LAW, "제60조"),
    chunk(DECREE, "제30조"),
]


class RetrieverTests(unittest.TestCase):
    def setUp(self):
        self.base_calls = []

        def base(q, k):
            self.base_calls.append(k)
            return [CHUNKS[3], CHUNKS[0], CHUNKS[4], CHUNKS[1]][:k]

        self.r = ArticleLookupRetriever(base, CHUNKS)

    def ids(self, q, k=3):
        return [c["chunk_id"] for c in self.r(q, k)]

    def test_referenced_article_pinned_first_and_deduped(self):
        self.assertEqual(self.ids("근로기준법 제21조 원문"), [f"{LAW}:제21조", f"{LAW}:제60조", f"{DECREE}:제30조"])

    def test_paragraph_chunk_first_when_paragraph_given(self):
        self.assertEqual(self.ids("제23조 제2항", k=2), [f"{LAW}:제23조:p2", f"{LAW}:제23조:p1"])

    def test_no_reference_returns_base_results(self):
        self.assertEqual(self.ids("월급이 밀렸어요"), [f"{LAW}:제60조", f"{LAW}:제21조", f"{DECREE}:제30조"])

    def test_unknown_or_other_law_article_not_pinned(self):
        self.assertEqual(self.ids("근로기준법 제999조"), self.ids("월급이 밀렸어요"))
        self.assertEqual(self.ids("남녀고용평등법 제21조"), self.ids("월급이 밀렸어요"))

    def test_k_is_respected(self):
        self.assertEqual(len(self.r("근로기준법 제17조랑 시행령 제30조, 제21조, 제23조", 2)), 2)

    def test_pinned_marked(self):
        top = self.r("제21조", 1)[0]
        self.assertTrue(top["pinned"])


if __name__ == "__main__":
    unittest.main()
