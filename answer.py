"""검색된 조문을 근거로 답변 생성 + 코드로 인용 검증 + 면책 문구.

흐름: 질문 -> 검색(상위 k 청크) -> LLM(JSON 응답) -> 파싱 -> 인용 검증 -> 출력 포맷(면책 문구 포함)

설계 메모
- 근거 강제: 프롬프트에 제공한 조문 안에서만 답하게 하고, 인용 목록을 JSON으로 받는다.
- 인용 검증은 LLM이 아니라 코드가 한다. 검색 결과에 없는 조문을 인용하면 invalid_citations로 분리한다.
- 답할 수 있다고 했는데 유효한 인용이 하나도 없으면 거절로 바꾼다 ("근거 없는 답변"을 내보내지 않는다).
- JSON 파싱에 실패해도 거절로 처리한다 (로컬 모델은 형식을 깨뜨릴 수 있음). 실패 여부는 평가 지표로 남긴다.
- 면책 문구는 LLM 출력이 아니라 코드가 항상 붙인다.
- LLM은 generate(system, user) -> str 인터페이스만 맞추면 교체 가능 (로컬 Qwen <-> Claude API).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

DISCLAIMER = (
    "※ 이 답변은 법령 조문 안내이며 법률 자문이 아닙니다. 개별 사안은 노무사·변호사 또는 "
    "고용노동부 고객상담센터(국번 없이 1350)에 문의하세요."
)

CITATION_RE = re.compile(r"(근로기준법(?:\s*시행령|\s*시행규칙)?)\s*(제\d+조(?:의\d+)?)")

SYSTEM_PROMPT = """너는 한국 근로기준법 조문 안내 도우미다. 아래 규칙을 반드시 지킨다.

1. 사용자 메시지에 [조문]으로 제공된 내용만 근거로 답한다. 제공되지 않은 법령, 판례, 행정해석, 일반 상식으로 답을 보태지 않는다.
2. 답변의 각 문장 끝에 근거 조문을 [근로기준법 제60조], [근로기준법 시행령 제30조]처럼 대괄호로 표시한다.
3. 다음 경우에는 답하지 않는다 (answerable=false):
   - 제공된 조문으로 질문에 답할 수 없을 때
   - 판례 해석, 소송 승패 예측, 개별 사건에 대한 법적 판단을 요구할 때
   - 고소장·계약서 등 문서 작성 대행을 요구할 때
   - 최저임금, 실업급여, 퇴직금 액수, 육아휴직처럼 제공된 조문 밖의 법령을 물을 때
   - 노동법과 관계없는 질문일 때
4. 쉬운 말로 3~6문장 이내로 답한다. 조문에 나온 숫자(일수, 시간, 비율)는 그대로 옮긴다.

반드시 아래 형식의 JSON 하나만 출력한다. 다른 글은 쓰지 않는다.
{"answerable": true 또는 false, "reason": "답하지 않는 이유 (answerable=true면 빈 문자열)", "answer": "답변 (answerable=false면 빈 문자열)", "citations": ["근로기준법 제60조", ...]}"""


@dataclass
class Answer:
    question: str
    answerable: bool
    answer: str
    reason: str
    citations: list[tuple[str, str]]  # 검증 통과한 (법령명, 조) - 검색 결과에 있는 것만
    invalid_citations: list[tuple[str, str]]  # 검색 결과에 없는 조문을 인용한 것
    retrieved: list[dict]  # 근거로 제공한 청크
    parse_error: bool = False
    downgraded: bool = False  # answerable이었지만 유효 인용이 없어 거절로 바꿈
    raw: str = ""
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- 프롬프트
FORMAT_REMINDER = "위 조문만 근거로, 시스템 지시의 JSON 형식 하나만 출력하라."


def build_user_message(question: str, chunks: list[dict]) -> str:
    # 작은 로컬 모델은 마지막 지시를 더 잘 따르므로 형식 지시를 질문 뒤에 한 번 더 둔다
    parts = [f"[조문 {i}]\n{c['text']}" for i, c in enumerate(chunks, start=1)]
    return "\n\n".join(parts) + f"\n\n[질문]\n{question}\n\n{FORMAT_REMINDER}"


# ---------------------------------------------------------------- 파싱/검증
def extract_json(raw: str) -> dict | None:
    """응답에서 첫 JSON 객체를 꺼낸다 (```json 펜스나 앞뒤 잡담이 붙어도)."""
    start = raw.find("{")
    if start < 0:
        return None
    try:
        obj, _ = json.JSONDecoder().raw_decode(raw[start:])
    except json.JSONDecodeError:
        return None
    return obj if isinstance(obj, dict) else None


def parse_citation(text: str) -> tuple[str, str] | None:
    """'근로기준법 시행령 제7조의2제1항' -> ('근로기준법 시행령', '제7조의2'). 항·호는 버리고 조 단위로."""
    m = CITATION_RE.search(text)
    if not m:
        return None
    law = re.sub(r"\s+", " ", m[1].replace("근로기준법", "근로기준법 ")).strip()
    return law, m[2]


def collect_citations(obj: dict) -> list[tuple[str, str]]:
    """citations 목록 + 답변 본문의 [..] 표기를 모두 모은다 (순서 유지, 중복 제거)."""
    texts = [str(c) for c in obj.get("citations") or [] if c]
    texts += re.findall(r"\[([^\[\]]+)\]", str(obj.get("answer") or ""))
    refs = []
    for t in texts:
        ref = parse_citation(t)
        if ref and ref not in refs:
            refs.append(ref)
    return refs


def build_answer(question: str, raw: str, chunks: list[dict]) -> Answer:
    allowed = {(c["law_name"], c["article_label"]) for c in chunks}
    obj = extract_json(raw)
    if obj is None:
        return Answer(question, False, "", "답변 형식 오류", [], [], chunks, parse_error=True, raw=raw)

    refs = collect_citations(obj)
    valid = [r for r in refs if r in allowed]
    invalid = [r for r in refs if r not in allowed]
    answerable = bool(obj.get("answerable")) and bool(str(obj.get("answer") or "").strip())
    downgraded = False
    if answerable and not valid:
        answerable, downgraded = False, True
    return Answer(
        question=question,
        answerable=answerable,
        answer=str(obj.get("answer") or "").strip() if answerable else "",
        reason="근거 조문을 확인하지 못했습니다" if downgraded else str(obj.get("reason") or "").strip(),
        citations=valid,
        invalid_citations=invalid,
        retrieved=chunks,
        downgraded=downgraded,
        raw=raw,
    )


# ---------------------------------------------------------------- 파이프라인
class QAPipeline:
    def __init__(self, retriever, llm, k: int = 5):
        self.retriever, self.llm, self.k = retriever, llm, k

    def ask(self, question: str) -> Answer:
        chunks = self.retriever(question, self.k)
        raw = self.llm.generate(SYSTEM_PROMPT, build_user_message(question, chunks))
        return build_answer(question, raw, chunks)


def format_answer(a: Answer) -> str:
    lines = []
    if a.answerable:
        lines.append(a.answer)
        # 인용은 조 단위로 검증하므로 표시도 조 단위로 한다. 항 단위 청크의 citation("... 제4항")을
        # 그대로 쓰면 실제 근거 항과 다른 항 번호가 보일 수 있다.
        titles = {}
        for c in a.retrieved:
            titles.setdefault((c["law_name"], c["article_label"]), re.sub(r"\s*제\d+항$", "", c["citation"]))
        lines.append("\n[근거 조문]")
        lines += [f"- {titles.get(r, ' '.join(r))}" for r in a.citations]
    else:
        lines.append("이 질문에는 답변드리기 어렵습니다." + (f" ({a.reason})" if a.reason else ""))
    if a.invalid_citations:
        lines.append("\n[검증 경고] 제공된 조문에 없는 인용을 제외했습니다: " + ", ".join(" ".join(r) for r in a.invalid_citations))
    lines.append("\n" + DISCLAIMER)
    return "\n".join(lines)


def load_default_pipeline(today, k: int = 5, embed_device: str | None = None, data_dir: str = "data") -> QAPipeline:
    """현재 최고 검색기(bge-m3 dense) + 로컬 Qwen. 모델은 처음 실행 시 hf_cache/로 다운로드."""
    from pathlib import Path

    from dense_retrieval import BgeM3Encoder, DenseRetriever
    from eval_retrieval import load_corpus

    chunks, _ = load_corpus(Path(data_dir), today)
    retriever = DenseRetriever(chunks, BgeM3Encoder(device=embed_device))
    return QAPipeline(retriever, QwenLLM(), k=k)


# ---------------------------------------------------------------- 로컬 LLM
QWEN_ID = "Qwen/Qwen3-4B-Instruct-2507"
QWEN_REVISION = "cdbee75f17c01a7cc42f958dc650907174af0554"  # 재현성을 위해 고정
QWEN_FILES = ["*.json", "*.safetensors", "merges.txt", "vocab.json", "LICENSE"]


class QwenLLM:
    """Qwen3-4B-Instruct-2507 (Apache-2.0, bf16 8GB -> 12GB GPU에 양자화 없이 적재).

    결정적 생성(greedy)으로 같은 입력에 같은 출력 -> 평가 재현 가능.
    응답 첫 글자를 '{'로 고정(prefill)해 JSON을 이어 쓰게 한다. 프리필 없이는 형식 지시를 무시하고
    일반 문장으로 답하는 경우가 있었다. 반환값에는 프리필 문자를 포함한다.
    """

    response_prefix = "{"

    name = f"qwen3-4b-instruct-2507@{QWEN_REVISION[:8]}"

    def __init__(self, cache_dir=None, device: str | None = None, max_new_tokens: int = 768):
        import torch
        from huggingface_hub import snapshot_download
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from dense_retrieval import model_dir

        path = snapshot_download(
            QWEN_ID, revision=QWEN_REVISION, allow_patterns=QWEN_FILES, cache_dir=str(cache_dir or model_dir())
        )
        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.bfloat16 if self.device == "cuda" else torch.float32
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForCausalLM.from_pretrained(path, dtype=dtype).to(self.device).eval()
        self.max_new_tokens = max_new_tokens

    def generate(self, system: str, user: str) -> str:
        messages = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        inputs = self.tokenizer(prompt + self.response_prefix, return_tensors="pt").to(self.device)
        with self.torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None,
                top_k=None,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_text = self.tokenizer.decode(out[0, inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        return self.response_prefix + new_text
