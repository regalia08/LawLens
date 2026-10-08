"""임베딩 검색 / RRF 하이브리드 테스트. 실제 모델 대신 가짜 인코더를 써서 모델 다운로드·GPU 없이 돈다.

실행: python -m unittest -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from dense_retrieval import DenseRetriever, HybridRetriever, rrf_fuse

VOCAB = ["임금", "해고", "휴가", "근로시간"]


class FakeEncoder:
    """텍스트에 들어 있는 VOCAB 단어로 만든 정규화 벡터. 인코딩한 텍스트 수를 센다."""

    name = "fake"

    def __init__(self):
        self.encoded = 0

    def encode(self, texts):
        self.encoded += len(texts)
        vecs = np.array([[float(w in t) for w in VOCAB] for t in texts]) + 1e-6
        return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def chunk(cid, text):
    return {"chunk_id": cid, "law_name": "근로기준법", "article_label": cid, "text": text}


CHUNKS = [
    chunk("제23조", "근로기준법 제23조(해고 등의 제한) 정당한 이유 없이 해고하지 못한다."),
    chunk("제43조", "근로기준법 제43조(임금 지급) 임금은 통화로 직접 지급하여야 한다."),
    chunk("제60조", "근로기준법 제60조(연차 유급휴가) 15일의 유급휴가를 주어야 한다."),
]


class DenseRetrieverTests(unittest.TestCase):
    def test_ranks_by_cosine_similarity(self):
        r = DenseRetriever(CHUNKS, FakeEncoder(), cache_dir=None)
        self.assertEqual(r("임금을 못 받았어요", 1)[0]["chunk_id"], "제43조")
        self.assertEqual(r("해고 통보", 1)[0]["chunk_id"], "제23조")

    def test_chunk_embeddings_are_cached(self):
        with tempfile.TemporaryDirectory() as d:
            first = FakeEncoder()
            DenseRetriever(CHUNKS, first, cache_dir=Path(d))
            self.assertEqual(first.encoded, 3)
            second = FakeEncoder()
            DenseRetriever(CHUNKS, second, cache_dir=Path(d))
            self.assertEqual(second.encoded, 0)  # 캐시에서 읽음
            third = FakeEncoder()
            DenseRetriever(CHUNKS + [chunk("제50조", "근로시간은 40시간")], third, cache_dir=Path(d))
            self.assertEqual(third.encoded, 1)  # 새 청크만 인코딩


class RrfTests(unittest.TestCase):
    def test_rrf_scores_and_order(self):
        a = [chunk("A", ""), chunk("B", ""), chunk("C", "")]
        b = [chunk("C", ""), chunk("A", "")]
        fused = rrf_fuse([a, b], k=60)
        self.assertEqual([c["chunk_id"] for c in fused], ["A", "C", "B"])
        self.assertAlmostEqual(fused[0]["score"], 1 / 61 + 1 / 62)
        self.assertAlmostEqual(fused[2]["score"], 1 / 62)

    def test_item_found_by_only_one_retriever_is_kept(self):
        fused = rrf_fuse([[chunk("A", "")], [chunk("B", "")]])
        self.assertEqual({c["chunk_id"] for c in fused}, {"A", "B"})

    def test_hybrid_combines_retrievers(self):
        lexical = lambda q, k: [chunk("X", ""), chunk("Y", "")][:k]
        dense = lambda q, k: [chunk("Y", ""), chunk("Z", "")][:k]
        top = HybridRetriever([lexical, dense], depth=2)("q", 3)
        self.assertEqual(top[0]["chunk_id"], "Y")  # 두 검색기 모두에서 상위
        self.assertEqual(len(top), 3)


if __name__ == "__main__":
    unittest.main()
