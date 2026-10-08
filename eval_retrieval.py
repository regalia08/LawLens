"""검색 평가: eval/questions.jsonl 의 질문마다 정답 조문이 상위 k개 안에 드는지(hit@k) 측정.

사용:
    python eval_retrieval.py --today 2026-10-07          # 기준선(BM25) 평가
    python eval_retrieval.py --show-misses              # 놓친 질문 목록까지

평가 단위는 조문(법령명 + 조 번호). 청크가 항 단위로 쪼개져 있어도 같은 조는 하나로 본다.
정답이 여러 개면 그중 하나라도 상위 k개에 들면 hit (질문 하나에 근거 조문이 여럿일 수 있으므로).
expected=refuse 인 범위 밖 질문은 검색 지표에서 빼고, 이후 거절 규칙 평가에 쓴다.

기준선 BM25는 형태소 분석 없이 글자 2-gram으로 토큰화한다.
임베딩/하이브리드 검색을 붙였을 때 이 숫자보다 나아지는지 비교하기 위한 출발점이다.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import Counter, defaultdict
from datetime import date
from pathlib import Path
from typing import Callable

from law_parser import chunk_article, parse_law

Retriever = Callable[[str, int], list[dict]]  # (질문, k) -> 점수순 청크 목록


# ---------------------------------------------------------------- 데이터
def load_corpus(data_dir: Path, today: date, max_chars: int = 1000) -> tuple[list[dict], list]:
    """data/*.json 전체를 파싱. (청크 목록, [(law, articles)]) 반환."""
    chunks, laws = [], []
    for path in sorted(data_dir.glob("*.json")):
        law, articles = parse_law(json.loads(path.read_text(encoding="utf-8")), today)
        laws.append((law, articles))
        chunks.extend(c for a in articles for c in chunk_article(law, a, max_chars))
    return chunks, laws


# dev: 개발 중 오답을 보며 쓰는 셋 (튜닝·선택은 여기서만). test: 따로 만든 홀드아웃 셋 (최종 후보만 측정).
SPLITS = {"dev": "eval/questions.jsonl", "test": "eval/test_questions.jsonl"}


def load_questions(path: Path) -> list[dict]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def article_ref(chunk: dict) -> tuple[str, str]:
    return chunk["law_name"], chunk["article_label"]


def gold_refs(q: dict) -> set[tuple[str, str]]:
    return {(g["law"], g["article"]) for g in q["gold"]}


# ---------------------------------------------------------------- 기준선 BM25
def bigrams(text: str) -> list[str]:
    toks = []
    for word in re.findall(r"[가-힣]+|[a-zA-Z]+|\d+", text):
        toks.extend([word] if len(word) == 1 else [word[i : i + 2] for i in range(len(word) - 1)])
    return toks


class Bm25Retriever:
    def __init__(self, chunks: list[dict], k1: float = 1.5, b: float = 0.75):
        self.chunks = chunks
        self.k1, self.b = k1, b
        self.docs = [Counter(bigrams(c["text"])) for c in chunks]
        self.lens = [sum(d.values()) for d in self.docs]
        self.avg = sum(self.lens) / len(self.lens)
        df = Counter(t for d in self.docs for t in d)
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def __call__(self, query: str, k: int) -> list[dict]:
        return self.search(Counter(bigrams(query)), k)

    def search(self, terms: dict[str, float], k: int) -> list[dict]:
        """terms: 토큰 -> 질의 가중치 (확장어는 원래 질문보다 낮게 줄 수 있다)."""
        scored = []
        for i, (doc, length) in enumerate(zip(self.docs, self.lens)):
            s = 0.0
            for t, qf in terms.items():
                tf = doc.get(t)
                if tf:
                    norm = tf + self.k1 * (1 - self.b + self.b * length / self.avg)
                    s += self.idf[t] * tf * (self.k1 + 1) / norm * qf
            if s > 0:
                scored.append((s, i))
        scored.sort(reverse=True)
        return [{**self.chunks[i], "score": s} for s, i in scored[:k]]


class ExpandedBm25Retriever:
    """질문 토큰(가중치 1) + 일상용어->법령용어 확장 토큰(가중치 weight)으로 BM25 검색."""

    def __init__(self, bm25: Bm25Retriever, expander, weight: float = 0.5):
        self.bm25, self.expander, self.weight = bm25, expander, weight

    def query_terms(self, query: str) -> dict[str, float]:
        terms: dict[str, float] = Counter(bigrams(query))
        for term in self.expander.expand(query):
            for t in bigrams(term):
                terms[t] = terms.get(t, 0) + self.weight
        return terms

    def __call__(self, query: str, k: int) -> list[dict]:
        return self.bm25.search(self.query_terms(query), k)


# ---------------------------------------------------------------- 지표
def ranked_articles(chunks: list[dict]) -> list[tuple[str, str]]:
    """청크 순위 -> 조문 순위 (같은 조의 두 번째 청크부터는 건너뜀)."""
    seen, out = set(), []
    for c in chunks:
        ref = article_ref(c)
        if ref not in seen:
            seen.add(ref)
            out.append(ref)
    return out


def evaluate(questions: list[dict], retrieve: Retriever, ks=(1, 3, 5), depth: int = 50) -> dict:
    scoped = [q for q in questions if q.get("expected") != "refuse"]
    hits = {k: 0 for k in ks}
    rr_sum = 0.0
    by_cat = defaultdict(lambda: [0, 0])  # category -> [hit@3, total]
    rows = []
    for q in scoped:
        ranking = ranked_articles(retrieve(q["question"], depth))
        gold = gold_refs(q)
        rank = next((i + 1 for i, ref in enumerate(ranking) if ref in gold), None)
        for k in ks:
            hits[k] += rank is not None and rank <= k
        rr_sum += 1 / rank if rank and rank <= 10 else 0
        by_cat[q["category"]][0] += rank is not None and rank <= 3
        by_cat[q["category"]][1] += 1
        rows.append({"id": q["id"], "rank": rank, "question": q["question"], "gold": sorted(gold), "top3": ranking[:3]})
    n = len(scoped)
    return {
        "questions": n,
        "out_of_scope_skipped": len(questions) - n,
        **{f"hit@{k}": round(hits[k] / n, 3) for k in ks},
        "mrr@10": round(rr_sum / n, 3),
        "by_category_hit@3": {c: f"{h}/{t}" for c, (h, t) in sorted(by_cat.items())},
        "rows": rows,
    }


# ---------------------------------------------------------------- CLI
def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="data")
    ap.add_argument("--split", choices=list(SPLITS), default="dev", help="평가셋 (기본 dev)")
    ap.add_argument("--questions", help="평가셋 파일 직접 지정 (--split보다 우선)")
    ap.add_argument("--today", help="만료 판정 기준일 (YYYY-MM-DD, 기본: 오늘)")
    ap.add_argument("--show-misses", action="store_true", help="top-3 밖 질문 출력")
    ap.add_argument("--retriever", choices=["bm25", "bm25+terms", "dense", "hybrid"], default="bm25")
    ap.add_argument("--weight", type=float, default=0.5, help="확장어 가중치 (bm25+terms)")
    ap.add_argument(
        "--fetch-terms", action="store_true", help="캐시에 없는 단어를 법제처 API로 조회해 캐시에 추가 (.env의 LAW_OC 필요)"
    )
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    chunks, _ = load_corpus(Path(args.data), today)
    questions = load_questions(Path(args.questions or SPLITS[args.split]))
    bm25 = Bm25Retriever(chunks)
    expander = None
    if args.retriever == "bm25":
        retriever, name = bm25, "bm25-char-bigram"
    elif args.retriever == "dense":
        from dense_retrieval import BgeM3Encoder, DenseRetriever

        encoder = BgeM3Encoder()
        retriever, name = DenseRetriever(chunks, encoder), encoder.name
    else:
        from term_expansion import QueryExpander, TermLexicon

        oc = None
        if args.fetch_terms:
            from fetch_law import load_oc

            oc = load_oc()
        lexicon = TermLexicon(online=args.fetch_terms, oc=oc)
        expander = QueryExpander(lexicon, [c["text"] for c in chunks])
        retriever = ExpandedBm25Retriever(bm25, expander, args.weight)
        name = f"bm25-char-bigram + 일상용어 확장(w={args.weight})"
        if args.retriever == "hybrid":
            from dense_retrieval import BgeM3Encoder, DenseRetriever, HybridRetriever

            encoder = BgeM3Encoder()
            retriever = HybridRetriever([retriever, DenseRetriever(chunks, encoder)])
            name = f"RRF[{name}, {encoder.name}]"
    result = evaluate(questions, retriever)
    if expander:
        expander.lexicon.save()

    rows = result.pop("rows")
    print(f"retriever: {name} | split: {args.questions or args.split} | chunks: {len(chunks)} | 기준일 {today}")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if args.show_misses:
        print("\n[top-3 밖]")
        for r in rows:
            if not r["rank"] or r["rank"] > 3:
                top = ", ".join(f"{law} {art}" for law, art in r["top3"])
                gold = ", ".join(f"{law} {art}" for law, art in r["gold"])
                print(f"  {r['id']} rank={r['rank']} | {r['question']}\n      정답: {gold}\n      top3: {top}")
                if expander:
                    print(f"      확장: {', '.join(expander.expand(r['question'])) or '-'}")


if __name__ == "__main__":
    main()
