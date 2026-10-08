"""평가셋 무결성 + 지표 계산 테스트.

정답 조문이 실제 데이터에 없거나 삭제/만료되어 청크가 없으면, 검색이 아무리 좋아도 맞힐 수 없다.
데이터(법령 버전)를 바꿨을 때 평가셋이 깨졌는지 여기서 먼저 잡는다.

실행: python -m unittest -v
"""
import unittest
from datetime import date
from pathlib import Path

from eval_retrieval import SPLITS, evaluate, gold_refs, load_corpus, load_questions, ranked_articles

ROOT = Path(__file__).parent
TODAY = date(2026, 10, 7)


class EvalSetIntegrityTests(unittest.TestCase):
    """dev(eval/questions.jsonl)와 test(eval/test_questions.jsonl) 모두 검사."""

    @classmethod
    def setUpClass(cls):
        cls.splits = {name: load_questions(ROOT / path) for name, path in SPLITS.items()}
        cls.questions = [q for qs in cls.splits.values() for q in qs]
        chunks, _ = load_corpus(ROOT / "data", TODAY)
        cls.indexed = {(c["law_name"], c["article_label"]) for c in chunks}

    def test_ids_unique(self):
        # dev와 test를 합쳐도 id가 겹치지 않아야 결과 파일에서 섞이지 않는다
        ids = [q["id"] for q in self.questions]
        self.assertEqual(len(ids), len(set(ids)))

    def test_no_identical_question_across_splits(self):
        dev = {q["question"].strip() for q in self.splits["dev"]}
        self.assertEqual([q["id"] for q in self.splits["test"] if q["question"].strip() in dev], [])

    def test_every_gold_article_is_indexed(self):
        missing = [
            (q["id"], ref) for q in self.questions for ref in gold_refs(q) if ref not in self.indexed
        ]
        self.assertEqual(missing, [], "정답 조문이 인덱스에 없음 (오타, 삭제/만료, 데이터 누락)")

    def test_in_scope_questions_have_gold_and_out_of_scope_do_not(self):
        for q in self.questions:
            with self.subTest(q["id"]):
                if q.get("expected") == "refuse":
                    self.assertEqual(q["gold"], [])
                else:
                    self.assertTrue(q["gold"])

    def test_size_matches_plan(self):
        for name, qs in self.splits.items():
            with self.subTest(name):
                scoped = [q for q in qs if q.get("expected") != "refuse"]
                self.assertGreaterEqual(len(scoped), 30)
                self.assertGreaterEqual(len(qs) - len(scoped), 5)


def chunk(law, art):
    return {"law_name": law, "article_label": art}


class MetricTests(unittest.TestCase):
    QS = [
        {"id": "a", "category": "c", "question": "A", "gold": [{"law": "L", "article": "제1조"}]},
        {"id": "b", "category": "c", "question": "B", "gold": [{"law": "L", "article": "제9조"}, {"law": "L", "article": "제3조"}]},
        {"id": "x", "category": "범위밖", "expected": "refuse", "question": "X", "gold": []},
    ]
    RESULTS = {
        # 같은 조의 항 청크가 연달아 나와도 조문 순위는 1칸만 차지해야 한다
        "A": [chunk("L", "제2조"), chunk("L", "제2조"), chunk("L", "제2조"), chunk("L", "제1조")],
        "B": [chunk("L", "제5조"), chunk("L", "제6조"), chunk("L", "제7조"), chunk("L", "제3조")],
    }

    def test_ranked_articles_dedups_paragraph_chunks(self):
        self.assertEqual(ranked_articles(self.RESULTS["A"]), [("L", "제2조"), ("L", "제1조")])

    def test_hit_at_k_and_mrr(self):
        r = evaluate(self.QS, lambda q, k: self.RESULTS[q], ks=(1, 3, 5))
        self.assertEqual(r["questions"], 2)
        self.assertEqual(r["out_of_scope_skipped"], 1)
        self.assertEqual((r["hit@1"], r["hit@3"], r["hit@5"]), (0.0, 0.5, 1.0))
        self.assertEqual(r["mrr@10"], round((1 / 2 + 1 / 4) / 2, 3))

    def test_ctx_counts_chunks_not_articles(self):
        # 같은 조 청크 3개 뒤에 정답이 있으면 조 단위 순위는 2위지만 상위 청크 3개(근거) 안에는 없다
        r = evaluate(self.QS[:1], lambda q, k: self.RESULTS[q], ks=(3,), context_k=3)
        self.assertEqual(r["hit@3"], 1.0)
        self.assertEqual(r["ctx@3"], 0.0)


if __name__ == "__main__":
    unittest.main()
