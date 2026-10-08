"""답변 평가: 평가셋 전체(범위 안 + 범위 밖)를 QA 파이프라인에 넣고 거절/인용 지표를 잰다.

사용:
    python eval_answers.py --today 2026-10-07
    -> 콘솔 요약 + eval/results/answers_<llm>.jsonl (질문별 결과, 사람이 검토용)

지표
- 범위 밖(expected=refuse): refusal_rate          거절해야 할 질문을 거절한 비율 (높을수록 좋음)
- 범위 안:                  false_refusal_rate    답해야 할 질문을 거절한 비율 (낮을수록 좋음)
                            gold_cited_rate       정답 조문 중 하나라도 인용한 비율 (분모 = 범위 안 전체)
                            gold_in_context_rate  정답 조문이 LLM에 준 근거 안에 있었던 비율 (검색 상한)
- 전체:                     invalid_citation_rate 검색 결과에 없는 조문을 인용한 질문 비율 (낮을수록 좋음)
                            parse_error_rate      JSON 형식 오류 비율
                            downgraded            답한다고 했지만 유효 인용이 없어 거절로 바꾼 수

gold_cited_rate는 "정답 조문을 근거로 댔는가"만 본다. 답변 문장이 조문 내용과 맞는지(정확성)는
자동으로 판정하지 않으므로 결과 파일을 보고 사람이 검토한다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import date
from pathlib import Path

from answer import Answer, load_default_pipeline
from eval_retrieval import SPLITS, gold_refs, load_questions


def answer_row(q: dict, a: Answer, seconds: float) -> dict:
    return {
        "id": q["id"],
        "category": q["category"],
        "expected": q.get("expected", "answer"),
        "question": q["question"],
        "gold": sorted(gold_refs(q)),
        "answerable": a.answerable,
        "answer": a.answer,
        "reason": a.reason,
        "citations": a.citations,
        "invalid_citations": a.invalid_citations,
        "retrieved": [(c["law_name"], c["article_label"]) for c in a.retrieved],
        "parse_error": a.parse_error,
        "downgraded": a.downgraded,
        "seconds": round(seconds, 1),
        "raw": a.raw,
    }


def score_answers(rows: list[dict]) -> dict:
    scoped = [r for r in rows if r["expected"] != "refuse"]
    out_of_scope = [r for r in rows if r["expected"] == "refuse"]

    def rate(xs, pred):
        return round(sum(1 for x in xs if pred(x)) / len(xs), 3) if xs else None

    def as_set(refs):
        return {tuple(r) for r in refs}

    return {
        "in_scope": len(scoped),
        "out_of_scope": len(out_of_scope),
        "refusal_rate": rate(out_of_scope, lambda r: not r["answerable"]),
        "false_refusal_rate": rate(scoped, lambda r: not r["answerable"]),
        "gold_cited_rate": rate(scoped, lambda r: r["answerable"] and as_set(r["citations"]) & as_set(r["gold"])),
        "gold_in_context_rate": rate(scoped, lambda r: as_set(r["retrieved"]) & as_set(r["gold"])),
        "invalid_citation_rate": rate(rows, lambda r: bool(r["invalid_citations"])),
        "parse_error_rate": rate(rows, lambda r: r["parse_error"]),
        "downgraded": sum(1 for r in rows if r["downgraded"]),
    }


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--split", choices=list(SPLITS), default="dev", help="평가셋 (기본 dev)")
    ap.add_argument("--questions", help="평가셋 파일 직접 지정 (--split보다 우선)")
    ap.add_argument("--today", help="만료 판정 기준일 (YYYY-MM-DD, 기본: 오늘)")
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--embed-device")
    ap.add_argument("--out-dir", default="eval/results")
    ap.add_argument("--no-lookup", action="store_true", help="조문 번호 직접 조회 끄기 (변경 전 비교용)")
    ap.add_argument("--tag", help="결과 파일명 뒤에 붙일 이름 (기존 결과 파일 보존용)")
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    qa = load_default_pipeline(today, k=args.k, embed_device=args.embed_device, lookup=not args.no_lookup)
    questions = load_questions(Path(args.questions or SPLITS[args.split]))

    rows = []
    for i, q in enumerate(questions, start=1):
        t = time.perf_counter()
        a = qa.ask(q["question"])
        rows.append(answer_row(q, a, time.perf_counter() - t))
        mark = "답변" if a.answerable else "거절"
        print(f"[{i}/{len(questions)}] {q['id']} {mark} ({rows[-1]['seconds']}s)", flush=True)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    suffix = "" if args.split == "dev" and not args.questions else f"_{Path(args.questions).stem if args.questions else args.split}"
    suffix += f"_{args.tag}" if args.tag else ""
    path = out / f"answers_{qa.llm.name.split('@')[0]}_k{args.k}{suffix}.jsonl"  # dev는 기존 파일명 유지
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8")

    summary = score_answers(rows)
    print(f"\nllm: {qa.llm.name} | retriever: dense bge-m3{'' if args.no_lookup else ' + 조문 번호 조회'} | split: {args.questions or args.split} | k={args.k} | 기준일 {today}")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"질문별 결과: {path}")


if __name__ == "__main__":
    main()
