"""답변 생성 테스트. 가짜 검색기·가짜 LLM을 써서 모델 다운로드·GPU 없이 돈다.

실행: python -m unittest -v
"""
import json
import unittest

from answer import (
    DISCLAIMER,
    FORMAT_REMINDER,
    QAPipeline,
    build_answer,
    build_user_message,
    extract_json,
    format_answer,
    parse_citation,
)
from eval_answers import answer_row, score_answers


def chunk(law, art, title, body):
    cite = f"{law} {art}({title})"
    return {"law_name": law, "article_label": art, "citation": cite, "text": f"{cite}\n{body}", "chunk_id": f"{law}:{art}"}


CHUNKS = [
    chunk("근로기준법", "제60조", "연차 유급휴가", "② 1개월 개근 시 1일의 유급휴가를 주어야 한다."),
    chunk("근로기준법 시행령", "제30조", "휴일", "① 유급휴일은 1주 동안의 소정근로일을 개근한 자에게 주어야 한다."),
]


def reply(**obj):
    base = {"answerable": True, "reason": "", "answer": "", "citations": []}
    return json.dumps({**base, **obj}, ensure_ascii=False)


class FakeLLM:
    name = "fake@0"

    def __init__(self, raw):
        self.raw = raw
        self.calls = []

    def generate(self, system, user):
        self.calls.append((system, user))
        return self.raw


class ParsingTests(unittest.TestCase):
    def test_extract_json_from_code_fence(self):
        self.assertEqual(extract_json('```json\n{"answerable": false}\n```'), {"answerable": False})

    def test_extract_json_with_trailing_text(self):
        self.assertEqual(extract_json('{"a": 1} 이상입니다.'), {"a": 1})

    def test_extract_json_failure(self):
        self.assertIsNone(extract_json("죄송하지만 답할 수 없습니다"))
        self.assertIsNone(extract_json('{"answerable": tru'))

    def test_parse_citation_normalizes_law_and_drops_paragraph(self):
        self.assertEqual(parse_citation("근로기준법 제60조제2항"), ("근로기준법", "제60조"))
        self.assertEqual(parse_citation("근로기준법시행령 제7조의2"), ("근로기준법 시행령", "제7조의2"))
        self.assertEqual(parse_citation("근로기준법 시행규칙 제4조"), ("근로기준법 시행규칙", "제4조"))
        self.assertIsNone(parse_citation("제60조"))


class BuildAnswerTests(unittest.TestCase):
    def test_valid_answer_with_citations(self):
        raw = reply(answer="1개월 개근하면 1일이 생깁니다 [근로기준법 제60조].", citations=["근로기준법 제60조"])
        a = build_answer("연차?", raw, CHUNKS)
        self.assertTrue(a.answerable)
        self.assertEqual(a.citations, [("근로기준법", "제60조")])
        self.assertEqual(a.invalid_citations, [])

    def test_citation_in_answer_text_counts(self):
        raw = reply(answer="개근하면 유급휴일입니다 [근로기준법 시행령 제30조제1항].")
        a = build_answer("주휴?", raw, CHUNKS)
        self.assertEqual(a.citations, [("근로기준법 시행령", "제30조")])

    def test_citation_not_in_context_is_invalid(self):
        raw = reply(answer="... [근로기준법 제60조] [근로기준법 제61조]", citations=["근로기준법 제61조"])
        a = build_answer("연차?", raw, CHUNKS)
        self.assertEqual(a.citations, [("근로기준법", "제60조")])
        self.assertEqual(a.invalid_citations, [("근로기준법", "제61조")])

    def test_answer_without_valid_citation_is_downgraded_to_refusal(self):
        raw = reply(answer="연차는 15일입니다.", citations=["근로기준법 제61조"])
        a = build_answer("연차?", raw, CHUNKS)
        self.assertFalse(a.answerable)
        self.assertTrue(a.downgraded)
        self.assertEqual(a.answer, "")

    def test_refusal_passes_through(self):
        a = build_answer("소송 이길까요?", reply(answerable=False, reason="개별 사건 판단"), CHUNKS)
        self.assertFalse(a.answerable)
        self.assertFalse(a.downgraded)
        self.assertEqual(a.reason, "개별 사건 판단")

    def test_parse_error_is_refusal(self):
        a = build_answer("연차?", "연차는 15일입니다.", CHUNKS)
        self.assertFalse(a.answerable)
        self.assertTrue(a.parse_error)


class PipelineAndFormatTests(unittest.TestCase):
    def test_pipeline_passes_retrieved_articles_to_llm(self):
        llm = FakeLLM(reply(answer="... [근로기준법 제60조]"))
        qa = QAPipeline(lambda q, k: CHUNKS[:k], llm, k=2)
        a = qa.ask("연차 있나요?")
        _, user = llm.calls[0]
        self.assertIn("[조문 1]\n근로기준법 제60조(연차 유급휴가)", user)
        self.assertIn("[조문 2]\n근로기준법 시행령 제30조(휴일)", user)
        self.assertTrue(user.endswith(f"[질문]\n연차 있나요?\n\n{FORMAT_REMINDER}"))
        self.assertEqual(len(a.retrieved), 2)

    def test_disclaimer_always_present(self):
        ok = build_answer("q", reply(answer="... [근로기준법 제60조]"), CHUNKS)
        no = build_answer("q", reply(answerable=False, reason="범위 밖"), CHUNKS)
        bad = build_answer("q", "형식 오류", CHUNKS)
        for a in (ok, no, bad):
            self.assertTrue(format_answer(a).endswith(DISCLAIMER))

    def test_format_lists_sources_and_warnings(self):
        a = build_answer("q", reply(answer="... [근로기준법 제60조]", citations=["근로기준법 제99조"]), CHUNKS)
        text = format_answer(a)
        self.assertIn("- 근로기준법 제60조(연차 유급휴가)", text)
        self.assertIn("[검증 경고]", text)
        self.assertIn("근로기준법 제99조", text)

    def test_user_message_numbering(self):
        self.assertTrue(build_user_message("q", CHUNKS[:1]).startswith("[조문 1]\n"))


class ScoreTests(unittest.TestCase):
    def test_score_answers(self):
        good = build_answer("q", reply(answer="... [근로기준법 제60조]"), CHUNKS)
        refused = build_answer("q", reply(answerable=False, reason="x"), CHUNKS)
        qs = [
            {"id": "a", "category": "c", "question": "q", "gold": [{"law": "근로기준법", "article": "제60조"}]},
            {"id": "b", "category": "c", "question": "q", "gold": [{"law": "근로기준법", "article": "제23조"}]},
            {"id": "x", "category": "범위밖", "expected": "refuse", "question": "q", "gold": []},
        ]
        rows = [answer_row(qs[0], good, 1), answer_row(qs[1], refused, 1), answer_row(qs[2], refused, 1)]
        rows = [json.loads(json.dumps(r)) for r in rows]  # 파일 저장/로드 후와 같은 형태(튜플 -> 리스트)
        s = score_answers(rows)
        self.assertEqual((s["in_scope"], s["out_of_scope"]), (2, 1))
        self.assertEqual(s["refusal_rate"], 1.0)
        self.assertEqual(s["false_refusal_rate"], 0.5)
        self.assertEqual(s["gold_cited_rate"], 0.5)
        self.assertEqual(s["gold_in_context_rate"], 0.5)
        self.assertEqual(s["invalid_citation_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
