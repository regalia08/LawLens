"""임베딩(dense) 검색 + BM25와의 하이브리드(RRF) 검색.

모델: BAAI/bge-m3 (다국어, 한국어 검색 성능이 좋고 입력 8192토큰까지 지원).
  512토큰 모델(multilingual-e5 등)은 1000자 안팎의 긴 조문 청크 뒷부분이 잘릴 수 있어 제외했다.
  dense 임베딩만 쓴다: [CLS] 벡터를 L2 정규화 -> 내적 = 코사인 유사도. 질의 접두 지시문 불필요.

벡터 저장소: 청크 251개라 numpy 행렬 전수 비교로 충분(ms 단위). FAISS/Chroma는 규모가 커지면 검토.

하이브리드: Reciprocal Rank Fusion, score = sum 1 / (k + rank), k=60 (Cormack et al. 2009 기본값).
  점수 대신 순위만 쓰므로 단위가 다른 BM25 점수와 코사인 유사도를 정규화 없이 합칠 수 있고,
  BM25:임베딩 비율 같은 가중치를 두지 않는다 (55문항 평가셋으로 가중치를 고르면 과적합).

모델 경로: .env 또는 환경변수 LAWLENS_MODEL_DIR, 없으면 프로젝트의 hf_cache/ (git 제외).
임베딩 캐시: .cache/embeddings/ (청크 텍스트 해시 -> 벡터, git 제외. 모델만 있으면 다시 만들 수 있음).
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

import numpy as np

MODEL_ID = "BAAI/bge-m3"
MODEL_REVISION = "5617a9f61b028005a4858fdac845db406aefb181"  # 재현성을 위해 고정
# 저장소에는 ONNX 사본(2.2GB)과 이미지도 있어 필요한 파일만 받는다 (~2.3GB)
MODEL_FILES = [
    "config.json",
    "pytorch_model.bin",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "sentencepiece.bpe.model",
]
EMBED_CACHE_DIR = Path(".cache/embeddings")
RRF_K = 60


def model_dir(env_path: Path = Path(".env")) -> Path:
    if os.environ.get("LAWLENS_MODEL_DIR"):
        return Path(os.environ["LAWLENS_MODEL_DIR"])
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            key, sep, value = line.partition("=")
            if sep and key.strip() == "LAWLENS_MODEL_DIR":
                return Path(value.strip().strip("'\""))
    return Path("hf_cache")


def text_key(text: str) -> str:
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------- 인코더
class BgeM3Encoder:
    """bge-m3 dense 인코더. torch/transformers는 실제로 쓸 때만 import (테스트는 가짜 인코더 사용)."""

    name = f"bge-m3@{MODEL_REVISION[:8]}"

    def __init__(self, cache_dir: Path | None = None, device: str | None = None, max_length: int = 1024):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModel, AutoTokenizer

        path = snapshot_download(
            MODEL_ID, revision=MODEL_REVISION, allow_patterns=MODEL_FILES, cache_dir=str(cache_dir or model_dir())
        )
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        dtype = torch.float16 if self.device == "cuda" else torch.float32
        self.model = AutoModel.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.max_length = max_length  # 청크는 최대 1000자 남짓이라 1024토큰이면 잘리지 않는다

    def encode(self, texts: list[str], batch_size: int = 16) -> np.ndarray:
        out = []
        with self.torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = self.tokenizer(
                    texts[i : i + batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                ).to(self.device)
                cls = self.model(**batch).last_hidden_state[:, 0]
                cls = self.torch.nn.functional.normalize(cls.float(), dim=-1)
                out.append(cls.cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, 0), dtype=np.float32)


# ---------------------------------------------------------------- 검색기
class DenseRetriever:
    def __init__(self, chunks: list[dict], encoder, cache_dir: Path | None = EMBED_CACHE_DIR):
        self.chunks = chunks
        self.encoder = encoder
        self.matrix = self._embed_chunks([c["text"] for c in chunks], cache_dir)

    def _embed_chunks(self, texts: list[str], cache_dir: Path | None) -> np.ndarray:
        cache: dict[str, np.ndarray] = {}
        path = cache_dir / f"{self.encoder.name}.npz" if cache_dir else None
        if path and path.exists():
            with np.load(path) as f:
                cache = {k: f[k] for k in f.files}
        missing = [t for t in dict.fromkeys(texts) if text_key(t) not in cache]
        if missing:
            for t, v in zip(missing, self.encoder.encode(missing)):
                cache[text_key(t)] = v
            if path:
                path.parent.mkdir(parents=True, exist_ok=True)
                np.savez(path, **cache)
        return np.stack([cache[text_key(t)] for t in texts])

    def __call__(self, query: str, k: int) -> list[dict]:
        q = self.encoder.encode([query])[0]
        scores = self.matrix @ q
        top = np.argsort(-scores)[:k]
        return [{**self.chunks[i], "score": float(scores[i])} for i in top]


def rrf_fuse(rankings: list[list[dict]], k: int = RRF_K) -> list[dict]:
    """여러 검색기의 청크 순위를 RRF로 합친다. 같은 청크는 chunk_id로 식별."""
    scores: dict[str, float] = {}
    first: dict[str, dict] = {}
    for ranking in rankings:
        for rank, c in enumerate(ranking, start=1):
            cid = c["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank)
            first.setdefault(cid, c)
    order = sorted(scores, key=lambda cid: -scores[cid])
    return [{**first[cid], "score": scores[cid]} for cid in order]


class HybridRetriever:
    """각 검색기에서 상위 depth개를 받아 RRF로 합친다."""

    def __init__(self, retrievers: list, depth: int = 50, k: int = RRF_K):
        self.retrievers, self.depth, self.k = retrievers, depth, k

    def __call__(self, query: str, k: int) -> list[dict]:
        return rrf_fuse([r(query, self.depth) for r in self.retrievers], self.k)[:k]
