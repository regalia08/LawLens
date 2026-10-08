"""근로기준법 조문 Q&A (로컬).

사용:
    python ask.py "입사 1년 안 됐는데 연차 있나요?"
    python ask.py                      # 대화형: 질문을 계속 입력 (빈 줄이면 종료)
    python ask.py "..." --show-context # 근거로 넘긴 조문과 LLM 원본 응답도 출력

첫 실행 시 bge-m3(~2.2GB)와 Qwen3-4B-Instruct(~8GB)를 hf_cache/로 내려받는다.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date

from answer import format_answer, load_default_pipeline


def main(argv=None) -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("question", nargs="?", help="질문 (없으면 대화형)")
    ap.add_argument("--today", help="만료 판정 기준일 (YYYY-MM-DD, 기본: 오늘)")
    ap.add_argument("--k", type=int, default=5, help="근거로 넘길 청크 수")
    ap.add_argument("--embed-device", help="임베딩 장치 (GPU 메모리가 부족하면 cpu)")
    ap.add_argument("--show-context", action="store_true")
    args = ap.parse_args(argv)

    today = date.fromisoformat(args.today) if args.today else date.today()
    qa = load_default_pipeline(today, k=args.k, embed_device=args.embed_device)

    def run(q: str) -> None:
        a = qa.ask(q)
        if args.show_context:
            print("[검색된 조문]")
            for c in a.retrieved:
                how = "조문 번호 지정" if c.get("pinned") else f"score {c['score']:.3f}"
                print(f"  - {c['citation']} ({how})")
            print(f"[LLM 원본]\n{a.raw}\n")
        print(format_answer(a))

    if args.question:
        run(args.question)
        return
    while True:
        try:
            q = input("\n질문> ").strip()
        except EOFError:
            break
        if not q:
            break
        run(q)


if __name__ == "__main__":
    main()
