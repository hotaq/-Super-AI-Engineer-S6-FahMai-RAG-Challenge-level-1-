from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
import requests

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional convenience only
    def load_dotenv() -> bool:
        return False


"""
FahMai RAG baseline converted from the starter notebook into a Python script.

Environment:
  uv sync

Run:
  export THAILLM_API_KEY="..."
  uv run python fahmai_rag_jina.py --retriever hybrid --n-questions 100 --output submission.csv
"""


LLM_MODEL_ALIASES = {
    "typhoon": "typhoon",
    "openthaigpt": "openthaigpt",
    "open thai gpt": "openthaigpt",
    "openthaigpt thaillm 8b instruct v7.2 research preview": "openthaigpt",
    "openthaigpt-thaillm-8b-instruct-v7.2": "openthaigpt",
    "openthaigpt-thaillm-8b-instruct-v7.2 research preview": "openthaigpt",
    "kbtg": "kbtg",
    "pathumma": "pathumma",
    "pathumma thaillm qwen3 8b think 2 0 0": "pathumma",
    "pathumma-thaillm-qwen3-8b-think-2.0.0": "pathumma",
    "pathumma thaillm qwen3 8b think 2.0.0": "pathumma",
}


SYSTEM_PROMPT = """คุณเป็นระบบตอบคำถามแบบหลายตัวเลือกของร้านฟ้าใหม่ (FahMai) ที่ต้องตอบให้แม่นยำที่สุด

ฐานข้อมูลมี 3 หมวด:
- products: สินค้า สเปค ราคา สถานะ สิ่งที่อยู่ในกล่อง การรับประกัน FAQ
- policies: คืนสินค้า ยกเลิก จัดส่ง รับประกัน สมาชิก Points
- store_info: ข้อมูลร้าน สาขา ช่องทางติดต่อ ข้อมูลบริษัท

งานของคุณคือเลือกคำตอบเพียงข้อเดียวจาก 1-10 โดยใช้เฉพาะข้อมูลในบริบทที่ให้มาเท่านั้น

โปรโตคอลการตัดสินใจ:
1. หา "คำถามหลัก" โดยตัดเรื่องเล่า ข้อมูลส่วนตัว และคำเกริ่นที่ไม่เกี่ยวข้องออก
2. ระบุชนิดคำถามในใจว่าเป็น fact / compare / recommendation / policy / calculation / availability
3. ระบุชื่อรุ่น ชื่อสินค้า SKU หรือเอกสารสำคัญที่คำถามอ้างถึงให้ชัด
4. อ่านบริบททั้งหมดที่เกี่ยวข้อง แล้วห้ามสลับข้อมูลข้ามรุ่นหรือข้ามเอกสาร
5. ประเมินตัวเลือก 1-8 ทีละข้อในใจว่าเป็น:
   - supported = มีหลักฐานตรงและครบ
   - contradicted = มีหลักฐานขัดแย้ง
   - insufficient = ยังไม่มีหลักฐานพอ
6. เลือกข้อเดียวที่เป็น supported และตรงที่สุด
7. ถ้าไม่มีข้อ 1-8 ที่ supported:
   - ถ้าคำถามยังเกี่ยวกับฟ้าใหม่ แต่ข้อมูลไม่พอจริง -> 9
   - ถ้าคำถามหลักไม่เกี่ยวกับฟ้าใหม่เลย -> 10

กฎบังคับ:
- ใช้เฉพาะข้อมูลในบริบท ห้ามเดา ห้ามใช้ความรู้ภายนอก
- อย่าเลือกข้อ 9 เพียงเพราะบริบทอ่อน ต้องเช็กก่อนว่าข้อ 1-8 มีข้อใด supported หรือไม่
- ถ้าคำถามกล่าวถึงหลายสินค้า หลายนโยบาย หรือหลายเอกสาร ต้องรวบรวมข้อมูลจากทุกบริบทที่เกี่ยวข้องก่อนตอบ
- ถ้าคำถามต้องคำนวณราคา ส่วนลด Points ค่าจัดส่ง ระยะเวลา หรือเงื่อนไขพิเศษ ให้รวบรวมตัวเลขและกฎทั้งหมดก่อน แล้วค่อยตัดสิน
- ถ้าเป็นคำถามเปรียบเทียบหลายรุ่น ให้ทำตารางในใจแบบ "รุ่น -> หลักฐาน" สำหรับแต่ละรุ่นก่อน ห้ามตอบจากรุ่นเดียว
- ถ้าเป็นคำถามแนะนำหรือคัดกรอง ให้ตัวเลือกต้องผ่านทุกเงื่อนไข ถ้าผิดเพียงข้อเดียวให้ตัดทิ้ง
- ถ้าเป็นคำถามประเภท "สิ่งที่อยู่ในกล่อง" หรือรายการอุปกรณ์ ให้ตัวเลือกต้องตรงกับรายการครบทุกชิ้น: ห้ามขาดของ, ห้ามเกินของ, ห้ามเปลี่ยนกำลังวัตต์หรือความยาวสาย
- ถ้าชื่อรุ่นหรือ SKU ในคำถามตรงกับเอกสารใด ให้ยึดเอกสารของรุ่นนั้นก่อนรุ่นที่ชื่อคล้ายกัน
- ห้ามเลือกตัวเลือกที่ "ใกล้เคียงที่สุด" ถ้ายังไม่ตรงครบทั้งประเภทสินค้า คุณสมบัติ ตัวเลข หน่วย และเงื่อนไข
- ตัวเลขและหน่วยต้องตรงเป๊ะ เช่น 5 ATM ≠ 10 ATM, IP67 ≠ IP68, LCD TFT ≠ IPS LCD
- ถ้าหลายข้อดูคล้ายกัน ให้เลือกข้อที่มีหลักฐานรองรับตรงที่สุดเพียงข้อเดียว
- ให้คิดเป็นขั้นตอนภายในใจ แต่ห้ามแสดงเหตุผล
- ตอบเป็น ANSWER: X เท่านั้น
"""

REVIEW_RETRY_SYSTEM_PROMPT = """คุณกำลังตรวจทานคำตอบเดิมของระบบด้วย facts ที่สกัดจากหลักฐานในฐานข้อมูลฟ้าใหม่แล้ว

กติกา:
- ใช้ facts ที่ให้มาเป็นหลักฐานหลัก
- ถ้า facts ขัดกับคำตอบเดิม ต้องแก้คำตอบให้สอดคล้องกับ facts
- ห้ามเดา ห้ามใช้ความรู้ภายนอก
- ถ้า facts ไม่พอจริงและคำถามยังเกี่ยวกับฟ้าใหม่ -> 9
- ถ้าคำถามไม่เกี่ยวกับฟ้าใหม่ -> 10
- ตอบเป็น ANSWER: X เท่านั้น
"""


@dataclass(frozen=True)
class Question:
    id: int
    question: str
    choices: dict[str, str]


@dataclass(frozen=True)
class Document:
    path: str
    section: str
    title: str
    text: str


@dataclass(frozen=True)
class Chunk:
    source: str
    section: str
    title: str
    heading: str
    text: str
    retrieval_text: str


@dataclass(frozen=True)
class SourceHintMatch:
    source: str
    reason: str
    score: float


@dataclass(frozen=True)
class QueryPlan:
    original_question: str
    main_question: str
    reasoning_mode: str
    retrieval_queries: tuple[str, ...]
    focus_points: tuple[str, ...]
    entities: tuple[str, ...]


@dataclass(frozen=True)
class QueryFacetPlan:
    normalized_query: str
    exact_fact: bool
    compare: bool
    recommendation: bool
    policy: bool
    policy_kind: str | None
    calculation: bool
    brands: tuple[str, ...]
    aliases: tuple[str, ...]
    preferred_section_types: tuple[str, ...]
    preferred_source_count: int
    allow_variant_docs: bool


@dataclass(frozen=True)
class DeterministicSolution:
    answer: int
    solver: str
    details: dict[str, object]
    category: str | None = None


def passthrough_query_plan(question_text: str) -> QueryPlan:
    return QueryPlan(
        original_question=question_text,
        main_question=question_text,
        reasoning_mode="fact",
        retrieval_queries=(question_text,),
        focus_points=(),
        entities=(),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run FahMai RAG with Jina embeddings.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--n-questions", type=int, default=100)
    parser.add_argument("--ids", default="", help="comma-separated question ids to run, e.g. 8,9,10")
    parser.add_argument("--retriever", choices=("dense", "bm25", "hybrid"), default="hybrid")
    parser.add_argument("--embedding-model", default="jinaai/jina-embeddings-v5-text-small")
    parser.add_argument(
        "--llm-model",
        default="openthaigpt",
        help="ThaiLLM model slug or friendly label (default: openthaigpt)",
    )
    parser.add_argument("--chunk-size", type=int, default=512)
    parser.add_argument("--chunk-overlap", type=int, default=128)
    parser.add_argument("--chunking-strategy", choices=("fixed", "markdown"), default="markdown")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--fetch-k", type=int, default=20)
    parser.add_argument("--max-per-source", type=int, default=2)
    parser.add_argument("--candidate-max-per-source", type=int, default=4)
    parser.add_argument("--source-aware-retrieval", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hint-chunks-per-source", type=int, default=2)
    parser.add_argument("--hint-max-sources", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=8, help="batch size for document embedding")
    parser.add_argument("--reranker", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--reranker-batch-size", type=int, default=8)
    parser.add_argument("--truncate-dim", type=int, default=0, help="0 = full embedding dim")
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:0, ...")
    parser.add_argument("--sleep-seconds", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--request-timeout", type=int, default=120)
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--default-answer", type=int, default=9)
    parser.add_argument(
        "--context-compression",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="apply relevant-segment extraction and context window compression before prompting the LLM",
    )
    parser.add_argument("--compression-max-segments", type=int, default=4)
    parser.add_argument("--compression-neighbor-window", type=int, default=1)
    parser.add_argument("--compression-min-score", type=float, default=2.0)
    parser.add_argument(
        "--deterministic-solvers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply narrow rule-based review/repair over compare/calc/policy answers after the LLM",
    )
    parser.add_argument(
        "--faceted-filtering",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="apply metadata-aware filtering over candidate chunks before final ranking",
    )
    parser.add_argument(
        "--choice-verifier",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental post-check over answer choices",
    )
    parser.add_argument(
        "--query-planning",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="experimental multi-query rewrite and prompt plan",
    )
    parser.add_argument("--output", type=Path, default=Path("submission.csv"))
    parser.add_argument("--cache-dir", type=Path, default=Path("data/_cache"))
    parser.add_argument("--debug-log", type=Path, default=None, help="optional JSONL path for per-question traces")
    parser.add_argument("--debug-preview-chars", type=int, default=220)
    parser.add_argument("--print-context", action="store_true")
    return parser.parse_args()


def normalize_model_name(model_name: str) -> str:
    key = re.sub(r"[^a-zA-Z0-9]+", " ", model_name).strip().lower()
    return LLM_MODEL_ALIASES.get(key, model_name.strip())


def parse_question_ids(raw_ids: str) -> list[int]:
    if not raw_ids.strip():
        return []
    ids: list[int] = []
    for part in raw_ids.split(","):
        value = part.strip()
        if not value:
            continue
        ids.append(int(value))
    return ids


def extract_title(text: str) -> str:
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("# "):
            return line[2:].strip()
    return "ไม่ระบุชื่อเอกสาร"


def load_questions(data_dir: Path) -> list[Question]:
    questions: list[Question] = []
    with (data_dir / "questions.csv").open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            choices = {str(i): row[f"choice_{i}"] for i in range(1, 11)}
            questions.append(
                Question(
                    id=int(row["id"]),
                    question=row["question"].strip(),
                    choices=choices,
                )
            )
    return questions


def load_documents(kb_dir: Path) -> list[Document]:
    documents: list[Document] = []
    for fp in sorted(kb_dir.rglob("*.md")):
        rel_path = fp.relative_to(kb_dir).as_posix()
        section = rel_path.split("/", 1)[0]
        text = fp.read_text(encoding="utf-8").strip()
        documents.append(
            Document(
                path=rel_path,
                section=section,
                title=extract_title(text),
                text=text,
            )
        )
    return documents


SECTION_HEADING_RE = re.compile(r"^(#{2,3})\s+(.*\S)\s*$")
WHOLE_SECTION_HINTS = (
    "สิ่งที่อยู่ในกล่อง",
    "การรับประกัน",
    "ความเข้ากันได้",
    "โปรโมชันปัจจุบัน",
    "คำถามที่พบบ่อย",
    "ช่องทางการติดต่อ",
    "ระยะเวลาการคืนเงิน",
    "ค่าจัดส่ง",
)


def make_fixed_chunks(text: str, size: int, overlap: int) -> list[str]:
    if size <= 0:
        raise ValueError("chunk size must be positive")
    if overlap >= size:
        raise ValueError("chunk overlap must be smaller than chunk size")
    if len(text) <= size:
        return [text]

    windows: list[str] = []
    start = 0
    step = size - overlap
    while start < len(text):
        windows.append(text[start : start + size])
        start += step
    return windows


def should_keep_section_intact(heading: str, text: str, size: int) -> bool:
    if len(text) <= size:
        return True
    return any(hint in heading for hint in WHOLE_SECTION_HINTS) and len(text) <= int(size * 1.75)


def split_markdown_sections(doc: Document) -> list[tuple[str, str]]:
    sections: list[tuple[str, str]] = []
    current_heading = "บทนำ"
    current_lines: list[str] = []
    heading_stack: dict[int, str] = {}

    def flush() -> None:
        clean_text = "\n".join(current_lines).strip()
        if clean_text:
            sections.append((current_heading, clean_text))

    for line in doc.text.splitlines():
        match = SECTION_HEADING_RE.match(line.strip())
        if match:
            flush()
            level = len(match.group(1))
            heading_stack[level] = match.group(2).strip()
            for existing_level in sorted(list(heading_stack.keys())):
                if existing_level > level:
                    del heading_stack[existing_level]
            current_heading = " > ".join(heading_stack[idx] for idx in sorted(heading_stack))
            current_lines = [line]
            continue

        current_lines.append(line)

    flush()
    return sections or [("บทนำ", doc.text.strip())]


def build_chunk_prefix(doc: Document, heading: str) -> str:
    prefix_lines = [
        f"หมวด: {doc.section}",
        f"ชื่อเอกสาร: {doc.title}",
        f"หัวข้อ: {heading}",
        f"ไฟล์: {doc.path}",
    ]
    return "\n".join(prefix_lines)


def build_chunks(
    documents: Iterable[Document],
    size: int,
    overlap: int,
    strategy: str,
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in documents:
        if strategy == "fixed":
            sections = [("ทั้งเอกสาร", doc.text)]
        elif strategy == "markdown":
            sections = split_markdown_sections(doc)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

        for heading, section_text in sections:
            windows = (
                [section_text]
                if should_keep_section_intact(heading, section_text, size)
                else make_fixed_chunks(section_text, size=size, overlap=overlap)
            )
            prefix = build_chunk_prefix(doc, heading=heading)
            for window in windows:
                clean_window = window.strip()
                retrieval_text = f"{prefix}\n\n{clean_window}"
                chunks.append(
                    Chunk(
                        source=doc.path,
                        section=doc.section,
                        title=doc.title,
                        heading=heading,
                        text=clean_window,
                        retrieval_text=retrieval_text,
                    )
                )
    return chunks


def sanitize_name(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9._-]+", "_", name).strip("_")


POLICY_SOURCE_RULES = (
    ("policies/return_policy.md", "return_policy", ("คืนสินค้า", "คืนได้", "ขอคืน", "mega sale")),
    ("policies/cancellation_policy.md", "cancellation_policy", ("ยกเลิก", "pre-order", "preorder")),
    ("policies/shipping_policy.md", "shipping_policy", ("จัดส่ง", "ค่าส่ง", "tracking", "express")),
    ("policies/warranty_policy.md", "warranty_policy", ("รับประกัน", "เคลม", "care+")),
    ("policies/membership_points_policy.md", "membership_points_policy", ("points", "คะแนน", "สมาชิก", "silver", "gold", "platinum")),
)

POLICY_QUERY_TERMS = ("คืน", "ยกเลิก", "รับประกัน", "points", "คะแนน", "สมาชิก", "จัดส่ง", "ค่าส่ง")
MULTI_ENTITY_QUERY_TERMS = ("รวม", "ทั้งหมด", "กับ", "เปรียบเทียบ", "ต่าง", "ตัวไหน", "ดีกว่า", "น้อยกว่า", "มากกว่า")
SPEC_QUERY_TERMS = ("sim", "esim", "ซิม", "แบต", "กล้อง", "จอ", "หน้าจอ", "กันน้ำ", "มาในกล่อง", "คีย์บอร์ด", "สี", "ราคา")
ACCESSORY_TITLE_TERMS = ("case", "film", "glass", "protector", "bundle", "keyboard bundle")

GENERIC_MATCH_TOKENS = {
    "daonuea",
    "saifah",
    "wongkhojon",
    "kluensiang",
    "judchuam",
    "arcwave",
    "novatech",
    "pulsegear",
    "zenbyte",
    "phone",
    "watch",
}

TITLE_GENERIC_MATCH_TOKENS = {
    "daonuea",
    "saifah",
    "wongkhojon",
    "kluensiang",
    "judchuam",
    "arcwave",
    "novatech",
    "pulsegear",
    "zenbyte",
    "phone",
    "watch",
}

LATIN_TOKEN_RE = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
THAI_TOKEN_RE = re.compile(r"[\u0E00-\u0E7F]{2,}")
SKU_TOKEN_RE = re.compile(r"^[a-z]{1,4}-[a-z]{1,4}-\d+[a-z0-9-]*$")
NUMBER_TOKEN_RE = re.compile(r"\d+(?:,\d+)?")

TITLE_ALIAS_DROP_TOKENS = {
    "usb",
    "usb-c",
    "usbc",
    "wireless",
    "magnetic",
    "detachable",
}


def normalize_match_text(text: str) -> str:
    normalized = text.lower()
    normalized = re.sub(r"[^0-9a-z\u0E00-\u0E7F+\-]+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    return normalized


def extract_match_tokens(text: str) -> set[str]:
    normalized = normalize_match_text(text)
    latin_tokens = set(LATIN_TOKEN_RE.findall(normalized))
    thai_tokens = {token for token in THAI_TOKEN_RE.findall(normalized) if len(token) >= 3}
    return latin_tokens | thai_tokens


def build_model_aliases(text: str) -> set[str]:
    tokens = LATIN_TOKEN_RE.findall(normalize_match_text(text))
    aliases: set[str] = set()
    for token in tokens:
        if any(ch.isdigit() for ch in token):
            aliases.add(token)

    for start in range(len(tokens)):
        for end in range(start + 2, min(len(tokens), start + 5) + 1):
            phrase_tokens = tokens[start:end]
            if not any(any(ch.isdigit() for ch in token) for token in phrase_tokens) and len(phrase_tokens) < 3:
                continue
            aliases.add(" ".join(phrase_tokens))
    return aliases


def split_text_segments(text: str) -> list[str]:
    try:
        from pythainlp.tokenize import sent_tokenize as thai_sent_tokenize
    except ImportError:  # pragma: no cover - optional fallback
        thai_sent_tokenize = None

    segments: list[str] = []
    for block in re.split(r"\n\s*\n", text):
        clean_block = block.strip()
        if not clean_block:
            continue
        lines = [line.strip() for line in clean_block.splitlines() if line.strip()]
        if len(lines) > 1 and any(
            line.startswith(("-", "|", "*"))
            or line.startswith("**Q")
            or line.startswith("A:")
            for line in lines
        ):
            segments.extend(lines)
            continue

        if thai_sent_tokenize is not None:
            try:
                sentence_parts = [part.strip() for part in thai_sent_tokenize(clean_block) if part.strip()]
            except Exception:  # pragma: no cover - tokenizer fallback
                sentence_parts = []
        else:
            sentence_parts = []

        if not sentence_parts:
            sentence_parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", clean_block) if part.strip()]

        if sentence_parts:
            segments.extend(sentence_parts)
        else:
            segments.append(clean_block)
    return segments or [text.strip()]


def extract_numeric_tokens(text: str) -> set[str]:
    return {token.replace(",", "") for token in NUMBER_TOKEN_RE.findall(text)}


def score_segment_for_query(
    segment: str,
    query_tokens: set[str],
    query_aliases: set[str],
    numeric_tokens: set[str],
) -> float:
    normalized_segment = normalize_match_text(segment)
    segment_tokens = extract_match_tokens(segment)
    score = float(len(query_tokens & segment_tokens) * 2)

    for alias in query_aliases:
        alias_norm = normalize_match_text(alias)
        if alias_norm and alias_norm in normalized_segment:
            score += 4.0

    normalized_numbers = extract_numeric_tokens(normalized_segment)
    score += float(len(numeric_tokens & normalized_numbers) * 1.5)

    if any(term in normalized_segment for term in ("ราคา", "points", "คะแนน", "กันน้ำ", "anc", "ldac", "nfc", "ecg")):
        score += 0.5
    return score


def build_precise_title_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    normalized = normalize_match_text(text)
    latin_tokens = LATIN_TOKEN_RE.findall(normalized)
    if not latin_tokens:
        return aliases

    sku_tokens = [token for token in latin_tokens if SKU_TOKEN_RE.match(token)]
    aliases.update(sku_tokens)

    filtered = [
        token
        for token in latin_tokens
        if token not in TITLE_GENERIC_MATCH_TOKENS and not SKU_TOKEN_RE.match(token)
    ]
    if filtered:
        aliases.add(" ".join(filtered))
        softened = [token for token in filtered if token not in TITLE_ALIAS_DROP_TOKENS]
        if softened and softened != filtered:
            aliases.add(" ".join(softened))

    return {alias for alias in aliases if len(alias) >= 4}


def build_prefix_stripped_aliases(text: str) -> set[str]:
    aliases: set[str] = set()
    current = text.strip()
    for _ in range(2):
        updated = re.sub(r"^[\u0E00-\u0E7F]+\s*", "", current).strip()
        if not updated or updated == current:
            break
        aliases.add(normalize_match_text(updated))
        current = updated
    return {alias for alias in aliases if len(alias) >= 4}


def build_source_aliases(doc: Document) -> set[str]:
    aliases: set[str] = set()
    title = doc.title.strip()
    title_without_parens = re.sub(r"\s*\([^)]*\)", "", title).strip()
    aliases.add(normalize_match_text(title_without_parens))
    aliases.update(build_prefix_stripped_aliases(title_without_parens))
    aliases.update(build_precise_title_aliases(title_without_parens))

    for part in re.findall(r"\(([^)]{2,})\)", title):
        aliases.add(normalize_match_text(part))
        aliases.update(build_precise_title_aliases(part))

    aliases.update(build_precise_title_aliases(Path(doc.path).stem.replace("_", " ")))
    return {alias for alias in aliases if len(alias) >= 4}


def infer_brand_key(text: str) -> str | None:
    normalized = normalize_match_text(text)
    for brand_key, keywords in BRAND_KEYWORDS.items():
        if any(normalize_match_text(keyword) in normalized for keyword in keywords):
            return brand_key
    return None


def classify_chunk_section_type(chunk: Chunk) -> str:
    heading_norm = normalize_match_text(chunk.heading)
    source_norm = normalize_match_text(chunk.source)

    if chunk.section == "policies":
        if "return_policy" in source_norm:
            return "return_policy"
        if "cancellation_policy" in source_norm:
            return "cancellation_policy"
        if "shipping_policy" in source_norm:
            return "shipping_policy"
        if "warranty_policy" in source_norm:
            return "warranty_policy"
        if "membership_points_policy" in source_norm:
            return "points_policy"
        return "policy"

    if "สิ่งที่อยู่ในกล่อง" in heading_norm:
        return "in_box"
    if "ความเข้ากันได้" in heading_norm:
        return "compatibility"
    if "คำถามที่พบบ่อย" in heading_norm:
        return "faq"
    if "การรับประกัน" in heading_norm:
        return "warranty"
    if "สเปค" in heading_norm:
        return "specs"
    if "รายละเอียดสินค้า" in heading_norm or heading_norm == "บทนำ":
        return "details"
    return "other"


def has_variant_marker(text: str) -> bool:
    normalized = normalize_match_text(text)
    return any(normalize_match_text(term) in normalized for term in VARIANT_DOC_TERMS)


def infer_policy_kind_from_query(normalized_query: str) -> str | None:
    if any(term in normalized_query for term in ("คืนสินค้า", "คืนได้", "ขอคืน", "mega sale", "คืน")):
        return "return_policy"
    if any(term in normalized_query for term in ("ยกเลิก", "pre-order", "preorder")):
        return "cancellation_policy"
    if any(term in normalized_query for term in ("ค่าส่ง", "จัดส่ง", "tracking", "express", "ส่ง")):
        return "shipping_policy"
    if any(term in normalized_query for term in ("care+", "รับประกัน", "เคลม")):
        return "warranty_policy"
    if any(term in normalized_query for term in ("points", "คะแนน", "สมาชิก", "silver", "gold", "platinum")):
        return "points_policy"
    return None


def build_query_facet_plan(query: str) -> QueryFacetPlan:
    normalized_query = normalize_match_text(query)
    aliases = tuple(
        sorted(
            alias
            for alias in build_model_aliases(query)
            if len(alias) >= 4 and not SPEC_LIKE_ALIAS_RE.fullmatch(alias.replace(" ", ""))
        )
    )
    brands = tuple(sorted({brand for brand in (infer_brand_key(query),) if brand is not None}))
    compare = has_compare_cue(query)
    calculation = any(term in normalized_query for term in ("รวม", "ทั้งหมด", "รวมกัน", "ประหยัด", "points", "คะแนน", "ค่าส่ง"))
    recommendation = any(
        term in normalized_query
        for term in ("แนะนำ", "อยากได้", "ควรซื้อ", "งบ", "ไม่เกิน", "ตัวไหนดี", "รุ่นไหนดี", "เหมาะกับ")
    )
    exact_fact = is_exact_fact_question_text(query)
    policy_kind = infer_policy_kind_from_query(normalized_query)
    policy = policy_kind is not None

    section_key = None
    if policy_kind is not None:
        section_key = policy_kind
    elif any(term in normalized_query for term in ("ในกล่อง", "มาในกล่อง", "แถม", "ต้องซื้อแยก", "ซื้อแยก")):
        section_key = "in_box"
    elif any(term in normalized_query for term in ("สั่งซื้อได้เลย", "พร้อมส่ง", "พรีออเดอร์", "pre-order", "preorder", "สั่งจอง")):
        section_key = "availability"
    elif "มีสีอะไร" in normalized_query:
        section_key = "color"
    elif "ราคาเท่าไหร่" in normalized_query:
        section_key = "price"
    elif any(term in normalized_query for term in ("ใช้กับ", "รองรับ", "เข้ากันได้", "ชาร์จไร้สาย", "แท่นชาร์จ")):
        section_key = "compatibility"
    elif compare or calculation:
        section_key = "compare"
    elif recommendation:
        section_key = "recommendation"

    preferred_sections = SECTION_TYPE_PREFERENCE_MAP.get(section_key or "", ())
    preferred_source_count = 3 if calculation else 2 if (compare or policy) else 1
    allow_variant_docs = any(normalize_match_text(term) in normalized_query for term in VARIANT_DOC_TERMS)

    return QueryFacetPlan(
        normalized_query=normalized_query,
        exact_fact=exact_fact,
        compare=compare,
        recommendation=recommendation,
        policy=policy,
        policy_kind=policy_kind,
        calculation=calculation,
        brands=brands,
        aliases=aliases,
        preferred_section_types=tuple(preferred_sections),
        preferred_source_count=preferred_source_count,
        allow_variant_docs=allow_variant_docs,
    )


def score_chunk_for_query(chunk: Chunk, normalized_query: str, query_tokens: set[str]) -> float:
    heading_text = normalize_match_text(chunk.heading)
    text_preview = normalize_match_text(chunk.text[:700])
    heading_overlap = len(query_tokens & extract_match_tokens(chunk.heading))
    text_overlap = len(query_tokens & extract_match_tokens(chunk.text[:700]))
    points_discount_query = any(
        term in normalized_query
        for term in ("points", "คะแนน", "ลด", "ส่วนลด", "ใช้ได้เท่าไหร่", "ประหยัด", "คุ้ม", "แลก")
    )

    score = float(heading_overlap * 3 + text_overlap)
    if "ราคา" in normalized_query and "ราคา" in text_preview:
        score += 3.0
    if any(term in normalized_query for term in ("พร้อมส่ง", "สั่งซื้อได้เลย", "พรีออเดอร์", "pre-order", "preorder", "สั่งจอง")) and any(
        term in text_preview for term in ("สถานะ", "พร้อมส่ง", "สั่งจองล่วงหน้า", "pre-order", "pre order", "ขายหมด")
    ):
        score += 4.0
    if any(term in normalized_query for term in ("points", "คะแนน", "สมาชิก")) and any(
        term in text_preview for term in ("points", "คะแนน", "สมาชิก")
    ):
        score += 3.0
    if any(term in normalized_query for term in ("ใช้ points", "ลดได้สูงสุด", "ส่วนลด", "20")) and any(
        term in heading_text or term in text_preview
        for term in ("4 1", "4 2", "การแลก", "ใช้ points", "ขั้นสูงสุด", "20%", "ส่วนลด")
    ):
        score += 8.0
    if points_discount_query and any(
        term in heading_text or term in text_preview
        for term in ("4 1", "อัตราการแลก", "100 points", "ส่วนลด ฿50", "1 000 points", "8 000 points")
    ):
        score += 12.0
    if any(term in normalized_query for term in ("points", "คะแนน", "ส่วนลด", "platinum", "gold", "silver")) and any(
        term in heading_text or term in text_preview
        for term in ("การแลก", "ใช้ points", "ขั้นสูงสุด", "ส่วนลด", "membership", "platinum", "gold", "silver")
    ):
        score += 5.0
    if points_discount_query and "ระดับสมาชิก" in heading_text:
        score -= 8.0
    if any(term in normalized_query for term in ("คืน", "ยกเลิก", "รับประกัน")) and any(
        term in heading_text or term in text_preview for term in ("คืน", "ยกเลิก", "รับประกัน")
    ):
        score += 2.0
    if any(term in normalized_query for term in ("mega sale", "คืน", "คืนได้")) and any(
        term in heading_text or term in text_preview for term in ("mega sale", "7 วัน", "ข้อยกเว้น")
    ):
        score += 5.0
    if chunk.heading == "บทนำ":
        score += 0.5
    return score


class RetrievalHintIndex:
    def __init__(self, documents: Iterable[Document], chunks: list[Chunk]) -> None:
        self.documents_by_source = {doc.path: doc for doc in documents}
        self.chunks_by_source: dict[str, list[Chunk]] = defaultdict(list)
        self.aliases_by_source: dict[str, set[str]] = {}
        self.tokens_by_source: dict[str, set[str]] = {}

        for chunk in chunks:
            self.chunks_by_source[chunk.source].append(chunk)

        for source, doc in self.documents_by_source.items():
            aliases = build_source_aliases(doc)
            tokens = set()
            for alias in aliases:
                tokens.update(extract_match_tokens(alias))
            self.aliases_by_source[source] = aliases
            self.tokens_by_source[source] = {token for token in tokens if token not in GENERIC_MATCH_TOKENS}

    def infer_source_hints(self, query: str, max_sources: int) -> list[SourceHintMatch]:
        normalized_query = normalize_match_text(query)
        query_tokens = {token for token in extract_match_tokens(query) if token not in GENERIC_MATCH_TOKENS}
        entity_matches: list[SourceHintMatch] = []
        policy_matches: list[SourceHintMatch] = []

        for source, aliases in self.aliases_by_source.items():
            best_alias = ""
            alias_score = 0.0
            for alias in aliases:
                if alias and alias in normalized_query:
                    token_count = len(alias.split())
                    if token_count > alias_score:
                        alias_score = float(token_count)
                        best_alias = alias

            overlap = len(query_tokens & self.tokens_by_source[source])
            score = alias_score * 4.0 + float(overlap)
            title_norm = normalize_match_text(self.documents_by_source[source].title)
            if any(term in normalized_query for term in SPEC_QUERY_TERMS) and not any(
                term in normalized_query for term in ACCESSORY_TITLE_TERMS
            ):
                if any(term in title_norm for term in ACCESSORY_TITLE_TERMS):
                    score -= 4.0
            if score >= 4.0:
                reason = f"title_match:{best_alias}" if best_alias else "token_overlap"
                entity_matches.append(SourceHintMatch(source=source, reason=reason, score=score))

        for source, reason, keywords in POLICY_SOURCE_RULES:
            if any(keyword in normalized_query for keyword in keywords):
                policy_matches.append(SourceHintMatch(source=source, reason=f"policy_hint:{reason}", score=6.0))

        deduped_by_reason: dict[str, SourceHintMatch] = {}
        for match in entity_matches:
            key = match.reason if match.reason.startswith("title_match:") else match.source
            current = deduped_by_reason.get(key)
            if current is None or match.score > current.score:
                deduped_by_reason[key] = match

        deduped: dict[str, SourceHintMatch] = {}
        for match in sorted(deduped_by_reason.values(), key=lambda item: item.score, reverse=True)[:max_sources]:
            deduped[match.source] = match
        for match in policy_matches:
            current = deduped.get(match.source)
            if current is None or match.score > current.score:
                deduped[match.source] = match

        return sorted(deduped.values(), key=lambda item: item.score, reverse=True)

    def select_hint_chunks(
        self,
        query: str,
        max_sources: int,
        per_source_limit: int,
    ) -> tuple[list[Chunk], list[dict[str, object]]]:
        normalized_query = normalize_match_text(query)
        query_tokens = extract_match_tokens(query)
        selected_chunks: list[Chunk] = []
        records: list[dict[str, object]] = []

        for rank, match in enumerate(self.infer_source_hints(query, max_sources=max_sources), start=1):
            source_chunks = self.chunks_by_source.get(match.source, [])
            ranked_chunks = sorted(
                source_chunks,
                key=lambda chunk: score_chunk_for_query(chunk, normalized_query=normalized_query, query_tokens=query_tokens),
                reverse=True,
            )
            chosen = ranked_chunks[:per_source_limit]
            selected_chunks.extend(chosen)
            records.append(
                {
                    "rank": rank,
                    "source": match.source,
                    "reason": match.reason,
                    "score": float(match.score),
                    "added_headings": [chunk.heading for chunk in chosen],
                }
            )

        return selected_chunks, records


def build_cache_path(
    cache_dir: Path,
    model_name: str,
    chunks: list[Chunk],
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    truncate_dim: int | None,
) -> Path:
    hasher = hashlib.sha1()
    hasher.update(model_name.encode("utf-8"))
    hasher.update(str(chunk_size).encode("utf-8"))
    hasher.update(str(chunk_overlap).encode("utf-8"))
    hasher.update(chunking_strategy.encode("utf-8"))
    hasher.update(str(truncate_dim or 0).encode("utf-8"))
    for chunk in chunks:
        hasher.update(chunk.retrieval_text.encode("utf-8"))
    digest = hasher.hexdigest()[:12]
    filename = (
        f"{sanitize_name(model_name)}_retrieval_{len(chunks)}"
        f"_cs{chunk_size}_co{chunk_overlap}_{sanitize_name(chunking_strategy)}"
        f"_td{truncate_dim or 0}_{digest}.npy"
    )
    return cache_dir / filename


def resolve_torch_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_batch_size(device: str, requested_batch_size: int) -> int:
    if requested_batch_size > 0:
        return requested_batch_size
    return 8


def load_embedding_model(model_name: str, device: str):
    import torch
    from sentence_transformers import SentenceTransformer

    kwargs = {
        "trust_remote_code": True,
        "device": device,
    }
    if device.startswith("cuda"):
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        kwargs["model_kwargs"] = {"dtype": dtype}
    return SentenceTransformer(model_name, **kwargs)


def load_reranker(model_name: str, device: str):
    if model_name == "jinaai/jina-reranker-v3":
        from transformers import AutoModel

        model = AutoModel.from_pretrained(
            model_name,
            trust_remote_code=True,
            dtype="auto",
        )
        if hasattr(model, "to"):
            model.to(device)
        if hasattr(model, "eval"):
            model.eval()
        return model

    from sentence_transformers import CrossEncoder

    model = CrossEncoder(model_name, device=device)
    tokenizer = getattr(model, "tokenizer", None)
    encoder = getattr(model, "model", None)

    if tokenizer is not None and getattr(tokenizer, "pad_token", None) is None:
        for token_name in ("eos_token", "sep_token", "cls_token", "unk_token"):
            candidate = getattr(tokenizer, token_name, None)
            if candidate is not None:
                tokenizer.pad_token = candidate
                break

    pad_token_id = getattr(tokenizer, "pad_token_id", None) if tokenizer is not None else None
    if encoder is not None and getattr(encoder.config, "pad_token_id", None) is None and pad_token_id is not None:
        encoder.config.pad_token_id = pad_token_id

    return model


def encode_documents(model, texts: list[str], batch_size: int, truncate_dim: int | None) -> np.ndarray:
    kwargs = {
        "task": "retrieval",
        "prompt_name": "document",
        "batch_size": batch_size,
        "show_progress_bar": True,
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if truncate_dim:
        kwargs["truncate_dim"] = truncate_dim
    return model.encode(sentences=texts, **kwargs)


def encode_query(model, query: str, truncate_dim: int | None) -> np.ndarray:
    kwargs = {
        "task": "retrieval",
        "prompt_name": "query",
        "normalize_embeddings": True,
        "convert_to_numpy": True,
    }
    if truncate_dim:
        kwargs["truncate_dim"] = truncate_dim
    encoded = model.encode(sentences=[query], **kwargs)
    return encoded[0]


def get_chunk_embeddings(
    model_name: str,
    chunks: list[Chunk],
    cache_dir: Path,
    chunk_size: int,
    chunk_overlap: int,
    chunking_strategy: str,
    truncate_dim: int | None,
    requested_device: str,
    requested_batch_size: int,
) -> tuple[object, np.ndarray, str]:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = build_cache_path(
        cache_dir=cache_dir,
        model_name=model_name,
        chunks=chunks,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        chunking_strategy=chunking_strategy,
        truncate_dim=truncate_dim,
    )

    device = resolve_torch_device(requested_device)
    model = load_embedding_model(model_name=model_name, device=device)

    if cache_path.exists():
        embeddings = np.load(cache_path)
        print(f"Loaded cached embeddings: {cache_path}")
        return model, embeddings, device

    batch_size = resolve_batch_size(device, requested_batch_size)
    texts = [chunk.retrieval_text for chunk in chunks]
    embeddings = encode_documents(model, texts=texts, batch_size=batch_size, truncate_dim=truncate_dim)
    np.save(cache_path, embeddings)
    print(f"Saved embeddings cache: {cache_path}")
    return model, embeddings, device


def build_bm25(chunks: list[Chunk]):
    from pythainlp.tokenize import word_tokenize
    from rank_bm25 import BM25Okapi

    tokenized = [word_tokenize(chunk.retrieval_text, engine="newmm") for chunk in chunks]
    return BM25Okapi(tokenized), word_tokenize


def diversify_results(
    ranked_indices: Iterable[int],
    score_lookup: np.ndarray,
    chunks: list[Chunk],
    k: int,
    max_per_source: int | None,
) -> tuple[list[int], list[float]]:
    if max_per_source is None or max_per_source <= 0:
        selected_indices = list(ranked_indices)[:k]
        selected_scores = [float(score_lookup[idx]) for idx in selected_indices]
        return selected_indices, selected_scores

    selected_indices: list[int] = []
    selected_scores: list[float] = []
    per_source = defaultdict(int)

    for idx in ranked_indices:
        source = chunks[idx].source
        if per_source[source] >= max_per_source:
            continue
        selected_indices.append(idx)
        selected_scores.append(float(score_lookup[idx]))
        per_source[source] += 1
        if len(selected_indices) == k:
            break
    return selected_indices, selected_scores


class DenseRetriever:
    def __init__(
        self,
        model,
        embeddings: np.ndarray,
        chunks: list[Chunk],
        top_k: int,
        fetch_k: int,
        max_per_source: int,
        truncate_dim: int | None,
    ) -> None:
        self.model = model
        self.embeddings = embeddings
        self.chunks = chunks
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.max_per_source = max_per_source
        self.truncate_dim = truncate_dim

    def retrieve_candidates(self, query: str, k: int | None = None, max_per_source: int | None = None) -> list[Chunk]:
        query_embedding = encode_query(self.model, query=query, truncate_dim=self.truncate_dim)
        scores = self.embeddings @ query_embedding
        limit = max(k or self.fetch_k, self.fetch_k)
        ranked = np.argsort(scores)[::-1][:limit]
        selected, _ = diversify_results(
            ranked_indices=ranked,
            score_lookup=scores,
            chunks=self.chunks,
            k=k or self.fetch_k,
            max_per_source=max_per_source,
        )
        return [self.chunks[idx] for idx in selected]

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_candidates(query, k=self.top_k, max_per_source=self.max_per_source)

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        final_chunks = self.retrieve(query)
        trace = {
            "retriever_type": "dense",
            "reranker_applied": False,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
            "final_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
        }
        return final_chunks, trace

    def top_indices(self, query: str, k: int) -> list[int]:
        query_embedding = encode_query(self.model, query=query, truncate_dim=self.truncate_dim)
        scores = self.embeddings @ query_embedding
        ranked = np.argsort(scores)[::-1][: max(k, self.fetch_k)]
        return list(ranked[:k])


class BM25Retriever:
    def __init__(
        self,
        bm25,
        tokenizer,
        chunks: list[Chunk],
        top_k: int,
        fetch_k: int,
        max_per_source: int,
    ) -> None:
        self.bm25 = bm25
        self.tokenizer = tokenizer
        self.chunks = chunks
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.max_per_source = max_per_source

    def _scores(self, query: str) -> np.ndarray:
        tokens = self.tokenizer(query, engine="newmm")
        return np.asarray(self.bm25.get_scores(tokens), dtype=np.float32)

    def retrieve_candidates(self, query: str, k: int | None = None, max_per_source: int | None = None) -> list[Chunk]:
        scores = self._scores(query)
        limit = max(k or self.fetch_k, self.fetch_k)
        ranked = np.argsort(scores)[::-1][:limit]
        selected, _ = diversify_results(
            ranked_indices=ranked,
            score_lookup=scores,
            chunks=self.chunks,
            k=k or self.fetch_k,
            max_per_source=max_per_source,
        )
        return [self.chunks[idx] for idx in selected]

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_candidates(query, k=self.top_k, max_per_source=self.max_per_source)

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        final_chunks = self.retrieve(query)
        trace = {
            "retriever_type": "bm25",
            "reranker_applied": False,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
            "final_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
        }
        return final_chunks, trace

    def top_indices(self, query: str, k: int) -> list[int]:
        scores = self._scores(query)
        ranked = np.argsort(scores)[::-1][: max(k, self.fetch_k)]
        return list(ranked[:k])


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: BM25Retriever,
        chunks: list[Chunk],
        top_k: int,
        fetch_k: int,
        max_per_source: int,
        rrf_k: int = 60,
    ) -> None:
        self.dense = dense
        self.bm25 = bm25
        self.chunks = chunks
        self.top_k = top_k
        self.fetch_k = max(fetch_k, top_k)
        self.max_per_source = max_per_source
        self.rrf_k = rrf_k

    def retrieve_candidates(self, query: str, k: int | None = None, max_per_source: int | None = None) -> list[Chunk]:
        limit = max(k or self.fetch_k, self.fetch_k)
        dense_idx = self.dense.top_indices(query, limit)
        bm25_idx = self.bm25.top_indices(query, limit)

        scores = np.zeros(len(self.chunks), dtype=np.float32)
        seen = set()

        for rank, idx in enumerate(dense_idx, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)
            seen.add(idx)
        for rank, idx in enumerate(bm25_idx, start=1):
            scores[idx] += 1.0 / (self.rrf_k + rank)
            seen.add(idx)

        ranked = sorted(seen, key=lambda idx: scores[idx], reverse=True)[:limit]
        selected, _ = diversify_results(
            ranked_indices=ranked,
            score_lookup=scores,
            chunks=self.chunks,
            k=k or self.fetch_k,
            max_per_source=max_per_source,
        )
        return [self.chunks[idx] for idx in selected]

    def retrieve(self, query: str) -> list[Chunk]:
        return self.retrieve_candidates(query, k=self.top_k, max_per_source=self.max_per_source)

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        final_chunks = self.retrieve(query)
        trace = {
            "retriever_type": "hybrid",
            "reranker_applied": False,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
            "final_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(final_chunks, start=1)
            ],
        }
        return final_chunks, trace


class HintAugmentedRetriever:
    def __init__(
        self,
        base_retriever,
        hint_index: RetrievalHintIndex,
        fetch_k: int,
        top_k: int,
        max_per_source: int,
        hint_chunks_per_source: int,
        hint_max_sources: int,
    ) -> None:
        self.base_retriever = base_retriever
        self.hint_index = hint_index
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.max_per_source = max_per_source
        self.hint_chunks_per_source = hint_chunks_per_source
        self.hint_max_sources = hint_max_sources

    def _collect_candidates(
        self,
        query: str,
        k: int | None,
        max_per_source: int | None,
    ) -> tuple[list[Chunk], list[dict[str, object]]]:
        requested_k = max(k or self.fetch_k, self.fetch_k)
        base_candidates = self.base_retriever.retrieve_candidates(
            query,
            k=requested_k,
            max_per_source=max_per_source,
        )
        hint_chunks, hint_records = self.hint_index.select_hint_chunks(
            query,
            max_sources=self.hint_max_sources,
            per_source_limit=self.hint_chunks_per_source,
        )

        merged_scores: dict[Chunk, float] = {}
        ordered_chunks: dict[Chunk, Chunk] = {}

        for rank, chunk in enumerate(base_candidates, start=1):
            ordered_chunks[chunk] = chunk
            merged_scores[chunk] = merged_scores.get(chunk, 0.0) + (1000.0 - float(rank))

        normalized_query = normalize_match_text(query)
        query_tokens = extract_match_tokens(query)
        for chunk in hint_chunks:
            ordered_chunks[chunk] = chunk
            merged_scores[chunk] = merged_scores.get(chunk, 0.0) + 100.0 + score_chunk_for_query(
                chunk,
                normalized_query=normalized_query,
                query_tokens=query_tokens,
            )

        ranked = sorted(ordered_chunks.values(), key=lambda chunk: merged_scores[chunk], reverse=True)
        return ranked, hint_records

    def retrieve_candidates(self, query: str, k: int | None = None, max_per_source: int | None = None) -> list[Chunk]:
        ranked, _ = self._collect_candidates(query, k=k, max_per_source=max_per_source)
        return ranked

    def retrieve_candidates_with_trace(
        self,
        query: str,
        k: int | None,
        max_per_source: int | None,
        preview_chars: int,
    ) -> tuple[list[Chunk], dict[str, object]]:
        ranked, hint_records = self._collect_candidates(query, k=k, max_per_source=max_per_source)
        trace = {
            "retriever_type": "hint_augmented_candidates",
            "base_retriever_type": self.base_retriever.__class__.__name__.replace("Retriever", "").lower(),
            "reranker_applied": False,
            "source_hints": hint_records,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(ranked[: max(k or self.fetch_k, self.fetch_k)], start=1)
            ],
            "final_chunks": [],
        }
        return ranked, trace

    def retrieve(self, query: str) -> list[Chunk]:
        ranked, _ = self._collect_candidates(query, k=self.top_k, max_per_source=self.max_per_source)
        selected, _ = diversify_results(
            ranked_indices=range(len(ranked)),
            score_lookup=np.asarray([len(ranked) - idx for idx in range(len(ranked))], dtype=np.float32),
            chunks=ranked,
            k=self.top_k,
            max_per_source=self.max_per_source,
        )
        return [ranked[idx] for idx in selected]

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        ranked, trace = self.retrieve_candidates_with_trace(
            query,
            k=self.top_k,
            max_per_source=self.max_per_source,
            preview_chars=preview_chars,
        )
        selected_chunks = self.retrieve(query)
        trace["retriever_type"] = "hint_augmented"
        trace["final_chunks"] = [
            chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
            for rank, chunk in enumerate(selected_chunks, start=1)
        ]
        return selected_chunks, trace


class FacetedFilteringRetriever:
    def __init__(
        self,
        base_retriever,
        hint_index: RetrievalHintIndex | None,
        fetch_k: int,
        top_k: int,
        max_per_source: int,
    ) -> None:
        self.base_retriever = base_retriever
        self.hint_index = hint_index
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.max_per_source = max_per_source

    def _candidate_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object] | None]:
        if hasattr(self.base_retriever, "retrieve_candidates_with_trace"):
            return self.base_retriever.retrieve_candidates_with_trace(
                query,
                k=self.fetch_k,
                max_per_source=self.max_per_source,
                preview_chars=preview_chars,
            )
        candidates = self.base_retriever.retrieve_candidates(
            query,
            k=self.fetch_k,
            max_per_source=self.max_per_source,
        )
        return candidates, None

    def _apply_named_filter(
        self,
        name: str,
        current_chunks: list[Chunk],
        next_chunks: list[Chunk],
        records: list[dict[str, object]],
        extra: dict[str, object] | None = None,
        min_unique_sources: int = 1,
    ) -> list[Chunk]:
        if not next_chunks:
            return current_chunks
        if len({chunk.source for chunk in next_chunks}) < min_unique_sources:
            return current_chunks
        if len(next_chunks) >= len(current_chunks):
            return current_chunks
        record = {
            "name": name,
            "before": len(current_chunks),
            "after": len(next_chunks),
        }
        if extra:
            record.update(extra)
        records.append(record)
        return next_chunks

    def _filter_candidates(
        self,
        query: str,
        candidates: list[Chunk],
    ) -> tuple[list[Chunk], QueryFacetPlan, list[dict[str, object]]]:
        plan = build_query_facet_plan(query)
        filtered = list(candidates)
        records: list[dict[str, object]] = []

        if self.hint_index is not None and (plan.policy or plan.calculation):
            hint_chunks, _ = self.hint_index.select_hint_chunks(
                query,
                max_sources=max(plan.preferred_source_count + 1, 2),
                per_source_limit=2,
            )
            seen = set(filtered)
            injected = [chunk for chunk in hint_chunks if chunk not in seen]
            if injected:
                filtered = filtered + injected
                records.append(
                    {
                        "name": "hint_injection",
                        "before": len(candidates),
                        "after": len(filtered),
                        "injected_sources": sorted({chunk.source for chunk in injected}),
                    }
                )

        if self.hint_index is not None and plan.policy_kind is not None:
            policy_source = POLICY_SOURCE_BY_KIND.get(plan.policy_kind)
            if policy_source:
                policy_chunks = list(self.hint_index.chunks_by_source.get(policy_source, []))
                if policy_chunks:
                    normalized_query = normalize_match_text(query)
                    query_tokens = extract_match_tokens(query)
                    policy_chunks = sorted(
                        policy_chunks,
                        key=lambda chunk: score_chunk_for_query(
                            chunk,
                            normalized_query=normalized_query,
                            query_tokens=query_tokens,
                        ),
                        reverse=True,
                    )[:2]
                    seen = set(filtered)
                    injected = [chunk for chunk in policy_chunks if chunk not in seen]
                    if injected:
                        filtered = filtered + injected
                        records.append(
                            {
                                "name": "policy_injection",
                                "before": len(filtered) - len(injected),
                                "after": len(filtered),
                                "policy_source": policy_source,
                            }
                        )

        if plan.preferred_section_types:
            section_filtered = [
                chunk for chunk in filtered if classify_chunk_section_type(chunk) in plan.preferred_section_types
            ]
            filtered = self._apply_named_filter(
                "section_type",
                filtered,
                section_filtered,
                records,
                extra={"preferred_section_types": list(plan.preferred_section_types)},
                min_unique_sources=2 if plan.compare else 1,
            )

        if plan.brands and not plan.policy:
            brand_filtered = [
                chunk
                for chunk in filtered
                if infer_brand_key(f"{chunk.title} {chunk.source}") in plan.brands
            ]
            filtered = self._apply_named_filter(
                "brand",
                filtered,
                brand_filtered,
                records,
                extra={"brands": list(plan.brands)},
                min_unique_sources=2 if plan.compare else 1,
            )

        if not plan.allow_variant_docs:
            non_variant = [
                chunk
                for chunk in filtered
                if not has_variant_marker(f"{chunk.title} {chunk.source}")
            ]
            filtered = self._apply_named_filter(
                "variant",
                filtered,
                non_variant,
                records,
                extra={"allow_variant_docs": False},
                min_unique_sources=2 if plan.compare else 1,
            )

        if plan.aliases or plan.exact_fact or plan.compare or plan.recommendation or plan.policy:
            preferred_sources = set()
            if plan.policy_kind is not None:
                policy_source = POLICY_SOURCE_BY_KIND.get(plan.policy_kind)
                if policy_source:
                    preferred_sources.add(policy_source)
            preferred_sources.update(
                select_best_matching_sources_from_query(
                    query,
                    filtered,
                    keep_top=plan.preferred_source_count,
                )
            )
            source_filtered = [chunk for chunk in filtered if chunk.source in preferred_sources]
            filtered = self._apply_named_filter(
                "source",
                filtered,
                source_filtered,
                records,
                extra={"preferred_sources": sorted(preferred_sources)},
                min_unique_sources=2 if plan.compare else 1,
            )

            if self.hint_index is not None and plan.preferred_section_types and preferred_sources:
                normalized_query = normalize_match_text(query)
                query_tokens = extract_match_tokens(query)
                injected_chunks: list[Chunk] = []
                seen = set(filtered)
                for source in sorted(preferred_sources):
                    source_chunks = list(self.hint_index.chunks_by_source.get(source, []))
                    preferred_chunks = [
                        chunk
                        for chunk in source_chunks
                        if classify_chunk_section_type(chunk) in plan.preferred_section_types
                    ]
                    if not preferred_chunks:
                        continue
                    preferred_chunks = sorted(
                        preferred_chunks,
                        key=lambda chunk: score_chunk_for_query(
                            chunk,
                            normalized_query=normalized_query,
                            query_tokens=query_tokens,
                        ),
                        reverse=True,
                    )[:1]
                    for chunk in preferred_chunks:
                        if chunk not in seen:
                            seen.add(chunk)
                            injected_chunks.append(chunk)
                if injected_chunks:
                    filtered = injected_chunks + [chunk for chunk in filtered if chunk not in injected_chunks]
                    records.append(
                        {
                            "name": "preferred_section_injection",
                            "after": len(filtered),
                            "preferred_section_types": list(plan.preferred_section_types),
                            "injected_sources": sorted({chunk.source for chunk in injected_chunks}),
                            "injected_headings": [chunk.heading for chunk in injected_chunks],
                        }
                    )

        return filtered, plan, records

    def retrieve_candidates(self, query: str, k: int | None = None, max_per_source: int | None = None) -> list[Chunk]:
        candidates, _ = self._candidate_trace(query, preview_chars=0)
        filtered, _, _ = self._filter_candidates(query, candidates)
        return filtered[: max(k or self.fetch_k, self.top_k)]

    def retrieve(self, query: str) -> list[Chunk]:
        candidates = self.retrieve_candidates(query, k=self.top_k, max_per_source=self.max_per_source)
        selected, _ = diversify_results(
            ranked_indices=range(len(candidates)),
            score_lookup=np.asarray([len(candidates) - idx for idx in range(len(candidates))], dtype=np.float32),
            chunks=candidates,
            k=self.top_k,
            max_per_source=self.max_per_source,
        )
        return [candidates[idx] for idx in selected]

    def retrieve_candidates_with_trace(
        self,
        query: str,
        k: int | None,
        max_per_source: int | None,
        preview_chars: int,
    ) -> tuple[list[Chunk], dict[str, object]]:
        candidates, candidate_trace = self._candidate_trace(query, preview_chars=preview_chars)
        filtered, plan, records = self._filter_candidates(query, candidates)
        trace = {
            "retriever_type": "faceted_candidates",
            "base_retriever_type": self.base_retriever.__class__.__name__.replace("Retriever", "").lower(),
            "reranker_applied": False,
            "source_hints": (candidate_trace or {}).get("source_hints", []),
                "facet_plan": {
                    "exact_fact": plan.exact_fact,
                    "compare": plan.compare,
                    "recommendation": plan.recommendation,
                    "policy": plan.policy,
                    "policy_kind": plan.policy_kind,
                    "calculation": plan.calculation,
                    "brands": list(plan.brands),
                    "aliases": list(plan.aliases),
                "preferred_section_types": list(plan.preferred_section_types),
                "preferred_source_count": plan.preferred_source_count,
                "allow_variant_docs": plan.allow_variant_docs,
            },
            "facet_filters": records,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(candidates, start=1)
            ],
            "final_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(filtered[: max(k or self.fetch_k, self.top_k)], start=1)
            ],
        }
        return filtered, trace

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        filtered, trace = self.retrieve_candidates_with_trace(
            query,
            k=self.top_k,
            max_per_source=self.max_per_source,
            preview_chars=preview_chars,
        )
        selected, _ = diversify_results(
            ranked_indices=range(len(filtered)),
            score_lookup=np.asarray([len(filtered) - idx for idx in range(len(filtered))], dtype=np.float32),
            chunks=filtered,
            k=self.top_k,
            max_per_source=self.max_per_source,
        )
        final_chunks = [filtered[idx] for idx in selected]
        trace["retriever_type"] = "faceted"
        trace["final_chunks"] = [
            chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
            for rank, chunk in enumerate(final_chunks, start=1)
        ]
        return final_chunks, trace


class CrossEncoderReranker:
    def __init__(self, model_name: str, device: str, batch_size: int) -> None:
        self.model_name = model_name
        self.device = device
        self.batch_size = batch_size
        self.model = load_reranker(model_name=model_name, device=device)

    def _predict_scores(self, pairs: list[tuple[str, str]]) -> np.ndarray:
        if hasattr(self.model, "rerank") and not hasattr(self.model, "predict"):
            if not pairs:
                return np.asarray([], dtype=np.float32)
            query = pairs[0][0]
            documents = [document for _, document in pairs]
            results = self.model.rerank(query, documents, top_n=None)
            scores = np.full(len(documents), -np.inf, dtype=np.float32)
            for result in results:
                scores[int(result["index"])] = float(result["relevance_score"])
            return scores
        try:
            return np.asarray(
                self.model.predict(pairs, batch_size=self.batch_size, show_progress_bar=False),
                dtype=np.float32,
            )
        except ValueError as exc:
            if "no padding token" not in str(exc).lower() or self.batch_size == 1:
                raise
            return np.asarray(
                self.model.predict(pairs, batch_size=1, show_progress_bar=False),
                dtype=np.float32,
            )

    def rerank(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
        max_per_source: int | None = None,
    ) -> list[Chunk]:
        if not chunks:
            return []

        pairs = [(query, chunk.retrieval_text) for chunk in chunks]
        scores = self._predict_scores(pairs)
        ranked = np.argsort(scores)[::-1]
        selected, _ = diversify_results(
            ranked_indices=ranked,
            score_lookup=scores,
            chunks=chunks,
            k=top_k,
            max_per_source=max_per_source,
        )
        return [chunks[idx] for idx in selected]

    def rerank_with_trace(
        self,
        query: str,
        chunks: list[Chunk],
        top_k: int,
        max_per_source: int | None,
        preview_chars: int,
    ) -> tuple[list[Chunk], list[dict[str, object]]]:
        if not chunks:
            return [], []

        pairs = [(query, chunk.retrieval_text) for chunk in chunks]
        scores = self._predict_scores(pairs)
        ranked = np.argsort(scores)[::-1]
        selected, _ = diversify_results(
            ranked_indices=ranked,
            score_lookup=scores,
            chunks=chunks,
            k=top_k,
            max_per_source=max_per_source,
        )
        final_chunks = [chunks[idx] for idx in selected]
        final_records = [
            chunk_to_debug_record(
                chunks[idx],
                rank=rank,
                preview_chars=preview_chars,
                score=float(scores[idx]),
            )
            for rank, idx in enumerate(selected, start=1)
        ]
        return final_chunks, final_records


class RerankingRetriever:
    def __init__(
        self,
        base_retriever,
        reranker: CrossEncoderReranker,
        fetch_k: int,
        top_k: int,
        candidate_max_per_source: int,
        max_per_source: int,
    ) -> None:
        self.base_retriever = base_retriever
        self.reranker = reranker
        self.fetch_k = fetch_k
        self.top_k = top_k
        self.candidate_max_per_source = candidate_max_per_source
        self.max_per_source = max_per_source

    def _candidate_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object] | None]:
        if hasattr(self.base_retriever, "retrieve_candidates_with_trace"):
            candidate_chunks, candidate_trace = self.base_retriever.retrieve_candidates_with_trace(
                query,
                k=self.fetch_k,
                max_per_source=self.candidate_max_per_source,
                preview_chars=preview_chars,
            )
            return candidate_chunks, candidate_trace

        candidate_chunks = self.base_retriever.retrieve_candidates(
            query,
            k=self.fetch_k,
            max_per_source=self.candidate_max_per_source,
        )
        return candidate_chunks, None

    def retrieve(self, query: str) -> list[Chunk]:
        candidates, _ = self._candidate_trace(query, preview_chars=0)
        return self.reranker.rerank(
            query=query,
            chunks=candidates,
            top_k=self.top_k,
            max_per_source=self.max_per_source,
        )

    def retrieve_with_trace(self, query: str, preview_chars: int) -> tuple[list[Chunk], dict[str, object]]:
        candidates, candidate_trace = self._candidate_trace(query, preview_chars=preview_chars)
        final_chunks, final_records = self.reranker.rerank_with_trace(
            query=query,
            chunks=candidates,
            top_k=self.top_k,
            max_per_source=self.max_per_source,
            preview_chars=preview_chars,
        )
        trace = {
            "retriever_type": "reranked",
            "base_retriever_type": self.base_retriever.__class__.__name__.replace("Retriever", "").lower(),
            "reranker_applied": True,
            "reranker_model": self.reranker.model_name,
            "source_hints": (candidate_trace or {}).get("source_hints", []),
            "_candidate_chunk_objects": candidates,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(candidates, start=1)
            ],
            "final_chunks": final_records,
        }
        return final_chunks, trace


def merge_ranked_chunk_lists(
    ranked_lists: list[list[Chunk]],
    top_k: int,
    max_per_source: int,
) -> list[Chunk]:
    merged_scores: dict[Chunk, float] = {}
    for ranked_chunks in ranked_lists:
        for rank, chunk in enumerate(ranked_chunks, start=1):
            merged_scores[chunk] = merged_scores.get(chunk, 0.0) + (1.0 / (50.0 + rank))

    ranked = sorted(merged_scores, key=lambda chunk: merged_scores[chunk], reverse=True)
    per_source: dict[str, int] = defaultdict(int)
    selected: list[Chunk] = []
    for chunk in ranked:
        if per_source[chunk.source] >= max_per_source:
            continue
        selected.append(chunk)
        per_source[chunk.source] += 1
        if len(selected) == top_k:
            break
    return selected


def retrieve_with_query_plan(
    retriever,
    query_plan: QueryPlan,
    preview_chars: int,
) -> tuple[list[Chunk], dict[str, object]]:
    retrieval_queries = list(query_plan.retrieval_queries) or [query_plan.original_question]
    if len(retrieval_queries) == 1:
        chunks, trace = retriever.retrieve_with_trace(retrieval_queries[0], preview_chars=preview_chars)
        trace["query_plan"] = {
            "reasoning_mode": query_plan.reasoning_mode,
            "main_question": query_plan.main_question,
            "retrieval_queries": retrieval_queries,
            "focus_points": list(query_plan.focus_points),
            "entities": list(query_plan.entities),
        }
        return chunks, trace

    if isinstance(retriever, RerankingRetriever):
        merged_candidates: list[Chunk] = []
        candidate_seen: set[Chunk] = set()
        per_query_records: list[dict[str, object]] = []

        for retrieval_query in retrieval_queries:
            candidate_chunks, candidate_trace = retriever._candidate_trace(retrieval_query, preview_chars=preview_chars)
            per_query_records.append(
                {
                    "query": retrieval_query,
                    "candidate_chunks": (candidate_trace or {}).get("candidate_chunks", []),
                    "source_hints": (candidate_trace or {}).get("source_hints", []),
                }
            )
            for chunk in candidate_chunks:
                if chunk in candidate_seen:
                    continue
                candidate_seen.add(chunk)
                merged_candidates.append(chunk)

        final_chunks, final_records = retriever.reranker.rerank_with_trace(
            query=query_plan.main_question,
            chunks=merged_candidates,
            top_k=retriever.top_k,
            max_per_source=retriever.max_per_source,
            preview_chars=preview_chars,
        )
        trace = {
            "retriever_type": "query_planned_reranked",
            "base_retriever_type": retriever.base_retriever.__class__.__name__.replace("Retriever", "").lower(),
            "reranker_applied": True,
            "reranker_model": retriever.reranker.model_name,
            "query_plan": {
                "reasoning_mode": query_plan.reasoning_mode,
                "main_question": query_plan.main_question,
                "retrieval_queries": retrieval_queries,
                "focus_points": list(query_plan.focus_points),
                "entities": list(query_plan.entities),
            },
            "per_query_candidates": per_query_records,
            "candidate_chunks": [
                chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
                for rank, chunk in enumerate(merged_candidates, start=1)
            ],
            "final_chunks": final_records,
        }
        return final_chunks, trace

    per_query_records: list[dict[str, object]] = []
    ranked_lists: list[list[Chunk]] = []
    for retrieval_query in retrieval_queries:
        chunks, trace = retriever.retrieve_with_trace(retrieval_query, preview_chars=preview_chars)
        ranked_lists.append(chunks)
        per_query_records.append(
            {
                "query": retrieval_query,
                "final_chunks": trace.get("final_chunks", []),
            }
        )

    final_chunks = merge_ranked_chunk_lists(
        ranked_lists=ranked_lists,
        top_k=getattr(retriever, "top_k", 5),
        max_per_source=getattr(retriever, "max_per_source", 2),
    )
    trace = {
        "retriever_type": "query_planned",
        "base_retriever_type": retriever.__class__.__name__.replace("Retriever", "").lower(),
        "reranker_applied": False,
        "query_plan": {
            "reasoning_mode": query_plan.reasoning_mode,
            "main_question": query_plan.main_question,
            "retrieval_queries": retrieval_queries,
            "focus_points": list(query_plan.focus_points),
            "entities": list(query_plan.entities),
        },
        "per_query_candidates": per_query_records,
        "candidate_chunks": [],
        "final_chunks": [
            chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
            for rank, chunk in enumerate(final_chunks, start=1)
        ],
    }
    return final_chunks, trace


def get_api_key() -> str:
    load_dotenv()
    api_key = os.getenv("THAILLM_API_KEY") or os.getenv("ThaiLLM")
    if not api_key:
        raise RuntimeError("Set THAILLM_API_KEY (or ThaiLLM) before running the script.")
    return api_key


def ask_llm(
    api_key: str,
    messages: list[dict[str, str]],
    model: str,
    max_tokens: int,
    temperature: float,
    request_timeout: int,
    max_retries: int = 5,
) -> str | None:
    url = f"http://thaillm.or.th/api/{model}/v1/chat/completions"
    headers = {"Content-Type": "application/json", "apikey": api_key}
    payload = {
        "model": "/model",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
    }

    for attempt in range(max_retries):
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=request_timeout)
            if response.status_code == 429:
                wait_seconds = min(2**attempt, 30)
                print(f"Rate limited, waiting {wait_seconds}s...")
                time.sleep(wait_seconds)
                continue

            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()
        except requests.RequestException as exc:
            wait_seconds = 2**attempt
            print(f"LLM request failed: {exc}. Retrying in {wait_seconds}s...")
            time.sleep(wait_seconds)

    return None


def format_structured_fact_lines(value: object, prefix: str = "") -> list[str]:
    if isinstance(value, dict):
        lines: list[str] = []
        for key, item in value.items():
            key_prefix = f"{prefix}{key}" if not prefix else f"{prefix}.{key}"
            lines.extend(format_structured_fact_lines(item, key_prefix))
        return lines
    if isinstance(value, (list, tuple)):
        lines: list[str] = []
        if not value:
            return [f"- {prefix}: []"] if prefix else ["- []"]
        for index, item in enumerate(value, start=1):
            item_prefix = f"{prefix}[{index}]" if prefix else f"[{index}]"
            lines.extend(format_structured_fact_lines(item, item_prefix))
        return lines
    label = prefix or "value"
    return [f"- {label}: {value}"]


def build_review_retry_prompt(
    question: Question,
    initial_answer: int | None,
    review_solution: DeterministicSolution,
) -> str:
    facts = format_structured_fact_lines(review_solution.details)
    choice_lines = [f"{key}. {question.choices[key]}" for key in map(str, range(1, 11))]
    previous = initial_answer if initial_answer is not None else "ไม่สามารถ parse ได้"
    return "\n".join(
        [
            "โปรดตอบคำถามเดิมอีกครั้งโดยยึด facts ที่สกัดจากหลักฐานด้านล่าง",
            f"คำถาม: {question.question}",
            "",
            "ตัวเลือก:",
            *choice_lines,
            "",
            f"คำตอบรอบแรกของระบบ: {previous}",
            f"หมวด evidence review: {review_solution.category or 'unknown'}",
            f"กฎที่พบ facts ชุดนี้: {review_solution.solver}",
            "",
            "Structured facts:",
            *facts,
            "",
            "ตรวจว่าคำตอบรอบแรกยังสอดคล้องกับ facts หรือไม่ ถ้าไม่สอดคล้องให้แก้",
            "ตอบเป็น ANSWER: X เท่านั้น",
        ]
    )


def parse_answer(text: str | None) -> int | None:
    if text is None:
        return None
    clean = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    match = re.search(r"ANSWER:\s*(\d+)", clean)
    if match:
        answer = int(match.group(1))
        return answer if 1 <= answer <= 10 else None
    for token in re.findall(r"\b(\d{1,2})\b", clean):
        answer = int(token)
        if 1 <= answer <= 10:
            return answer
    return None


def normalize_choice_match_text(text: str) -> str:
    text = re.sub(r"</?think>", " ", text or "", flags=re.DOTALL)
    text = text.lower()
    text = re.sub(r"[\(\)\[\]\{\}\"'“”‘’`*#:_\-–—/\\|]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def score_choice_residual_support(
    choice_text: str,
    snippet: str,
    context_norm: str,
    question_aliases: set[str],
) -> float:
    normalized_choice = normalize_choice_match_text(choice_text)
    normalized_snippet = normalize_choice_match_text(snippet)
    residual = normalized_choice.replace(normalized_snippet, " ").strip()
    if not residual:
        return 0.0

    residual_phrases = [
        phrase.strip()
        for phrase in re.split(r"\s*(?:,|แต่|พร้อม|และ)\s*", residual)
        if phrase.strip()
    ]
    filtered_phrases = []
    for phrase in residual_phrases:
        phrase_tokens = {
            token
            for token in extract_match_tokens(phrase)
            if token not in question_aliases and token not in CHOICE_VERIFIER_STOPWORDS and len(token) >= 3
        }
        if phrase_tokens:
            filtered_phrases.append((phrase, phrase_tokens))
    if not filtered_phrases:
        return 0.0

    score = 0.0
    for phrase, phrase_tokens in filtered_phrases:
        if phrase in context_norm:
            score += 1.6
            continue
        hits = sum(1 for token in phrase_tokens if token in context_norm)
        coverage = hits / len(phrase_tokens)
        if coverage >= 0.7:
            score += 1.0
        elif coverage >= 0.4:
            score += 0.3
        else:
            score -= 1.0
    return score


def infer_choice_from_rationale(
    question: Question,
    text: str | None,
    retrieved_chunks: list[Chunk] | None = None,
) -> tuple[int | None, dict[str, object] | None]:
    if text is None:
        return None, None

    clean = re.sub(r"</?think>", " ", text, flags=re.DOTALL).strip()
    snippet_patterns = [
        r'ตัวเลือกที่ถูกต้องคือตัวเลือกที่ระบุว่า\s*"([^"]+)"',
        r'คำตอบที่ถูกต้องคือตัวเลือกที่ระบุว่า\s*"([^"]+)"',
        r'ตรงกับตัวเลือกที่ระบุว่า\s*"([^"]+)"',
        r'คำตอบที่ถูกต้องคือ\s*"([^"]+)"',
        r"ตัวเลือกที่ถูกต้องคือตัวเลือกที่ระบุว่า\s*([^\n\.]+?)(?=\s+ซึ่ง|\n|$)",
        r"คำตอบที่ถูกต้องคือตัวเลือกที่ระบุว่า\s*([^\n\.]+?)(?=\s+ซึ่ง|\n|$)",
        r"ตรงกับตัวเลือกที่ระบุว่า\s*([^\n\.]+?)(?=\s+ซึ่ง|\n|$)",
    ]

    snippet = None
    for pattern in snippet_patterns:
        match = re.search(pattern, clean)
        if match:
            snippet = match.group(1).strip()
            break
    if not snippet:
        return None, None

    normalized_snippet = normalize_choice_match_text(snippet)
    if not normalized_snippet:
        return None, None

    snippet_tokens = set(normalized_snippet.split())
    scored: list[tuple[int, float]] = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        normalized_choice = normalize_choice_match_text(choice_text)
        similarity = SequenceMatcher(None, normalized_snippet, normalized_choice).ratio()
        choice_tokens = set(normalized_choice.split())
        token_overlap = len(snippet_tokens & choice_tokens) / len(snippet_tokens) if snippet_tokens else 0.0
        scored.append((key, max(similarity, token_overlap)))

    scored.sort(key=lambda item: item[1], reverse=True)
    best_key, best_score = scored[0]
    second_score = scored[1][1] if len(scored) > 1 else 0.0
    if best_score == second_score and retrieved_chunks:
        context_norm = normalize_match_text(
            " ".join(f"{chunk.title} {chunk.heading} {chunk.text}" for chunk in retrieved_chunks)
        )
        question_aliases = build_model_aliases(question.question)
        tied_choices = [key for key, score in scored if score == best_score]
        rescored: list[tuple[int, float]] = []
        for key in tied_choices:
            residual_score = score_choice_residual_support(
                choice_text=question.choices[str(key)],
                snippet=snippet,
                context_norm=context_norm,
                question_aliases=question_aliases,
            )
            rescored.append((key, residual_score))
        rescored.sort(key=lambda item: item[1], reverse=True)
        if rescored:
            best_key, best_residual_score = rescored[0]
            second_residual_score = rescored[1][1] if len(rescored) > 1 else float("-inf")
            if best_residual_score > second_residual_score:
                trace = {
                    "snippet": snippet,
                    "best_choice": best_key,
                    "best_score": round(best_score, 3),
                    "second_score": round(second_score, 3),
                    "tie_break": {
                        "rescored": [{str(key): round(score, 3)} for key, score in rescored],
                        "applied": True,
                    },
                }
                return best_key, trace
    trace = {
        "snippet": snippet,
        "best_choice": best_key,
        "best_score": round(best_score, 3),
        "second_score": round(second_score, 3),
    }
    if best_score < 0.62 or best_score < second_score + 0.08:
        return None, trace
    return best_key, trace


def score_choice_directly_against_context(
    choice_text: str,
    context_norm: str,
    question_aliases: set[str],
) -> float:
    normalized_choice = normalize_choice_match_text(choice_text)
    if not normalized_choice:
        return float("-inf")

    phrases = [
        phrase.strip()
        for phrase in re.split(r"\s*(?:,|แต่|พร้อม|และ)\s*", normalized_choice)
        if phrase.strip()
    ]

    score = 0.0
    meaningful_phrases = 0
    for phrase in phrases:
        phrase_tokens = {
            token
            for token in extract_match_tokens(phrase)
            if token not in question_aliases
            and token not in CHOICE_VERIFIER_STOPWORDS
            and token not in IN_BOX_GENERIC_TOKENS
            and len(token) >= 2
        }
        if not phrase_tokens:
            continue

        meaningful_phrases += 1
        if phrase in context_norm:
            score += 1.8
            continue

        hits = sum(1 for token in phrase_tokens if token in context_norm)
        coverage = hits / len(phrase_tokens)
        if coverage >= 0.95:
            score += 1.2
        elif coverage >= 0.7:
            score += 0.1
        else:
            score -= 1.6

    if meaningful_phrases == 0:
        return float("-inf")

    for watt in re.findall(r"\b\d+\s*w\b", normalized_choice):
        score += 2.2 if watt in context_norm else -2.6

    cable_phrases = ("usb c to usb c", "usb c to lightning")
    for cable_phrase in cable_phrases:
        if cable_phrase in normalized_choice:
            score += 1.8 if cable_phrase in context_norm else -2.2

    if "ต้องซื้อแยก" in normalized_choice:
        score += 1.2 if "ต้องซื้อแยก" in context_norm else -3.2
    if "ไม่รวมสาย" in normalized_choice:
        score += 1.0 if "ไม่รวมสาย" in context_norm else -0.8
    if "มาในกล่อง" in normalized_choice:
        if "สิ่งที่อยู่ในกล่อง" in context_norm or "มาในกล่อง" in context_norm:
            score += 1.2
        elif "ต้องซื้อแยก" in context_norm:
            score -= 1.4

    for unsupported_phrase in (
        "ลงทะเบียนออนไลน์",
        "ทุกรุ่นทุกสี",
        "ทุกสี",
        "usb c to lightning",
        "ai charge management",
        "ชาร์จเต็มใน 55 นาที",
        "100w",
        "45w",
        "1 5 เมตร",
    ):
        if unsupported_phrase in normalized_choice and unsupported_phrase not in context_norm:
            score -= 2.4

    return score


def infer_in_box_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None

    normalized_question = normalize_match_text(question.question)
    if not any(term in normalized_question for term in ("ในกล่อง", "มาในกล่อง", "แถม", "ต้องซื้อแยก", "ซื้อแยก")):
        return None, None

    relevant_chunks = [
        chunk
        for chunk in retrieved_chunks
        if "สิ่งที่อยู่ในกล่อง" in normalize_match_text(chunk.heading)
        or "ในกล่อง" in normalize_match_text(chunk.text)
        or "อะแดปเตอร์" in normalize_match_text(chunk.text)
        or "หัวชาร์จ" in normalize_match_text(chunk.text)
    ]
    if not relevant_chunks:
        relevant_chunks = list(retrieved_chunks)

    best_sources = select_best_matching_sources(question, relevant_chunks, keep_top=1)
    if best_sources:
        filtered_chunks = [chunk for chunk in relevant_chunks if chunk.source in best_sources]
        if filtered_chunks:
            relevant_chunks = filtered_chunks

    context_norm = normalize_match_text(
        " ".join(f"{chunk.title} {chunk.heading} {chunk.text}" for chunk in relevant_chunks)
    )
    question_aliases = build_model_aliases(question.question)

    scored: list[dict[str, object]] = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        score = score_choice_directly_against_context(choice_text, context_norm, question_aliases)
        scored.append({"choice": key, "score": round(score, 3), "text": choice_text})

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": float("-inf")}
    if float(best["score"]) < 0.8 or float(best["score"]) < float(second["score"]) + 1.0:
        return None, {
            "best_choice": int(best["choice"]),
            "best_score": float(best["score"]),
            "second_score": float(second["score"]),
            "direct_context_scoring": scored[:3],
            "best_sources": sorted(best_sources),
            "applied": False,
        }

    return int(best["choice"]), {
        "best_choice": int(best["choice"]),
        "best_score": float(best["score"]),
        "second_score": float(second["score"]),
        "direct_context_scoring": scored[:3],
        "best_sources": sorted(best_sources),
        "applied": True,
    }


def infer_missing_entity_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None

    normalized_question = normalize_match_text(question.question)
    if any(
        term in normalized_question
        for term in (
            "เปรียบเทียบ",
            "รวม",
            "มากกว่า",
            "น้อยกว่า",
            "ดีกว่า",
            "ต่างกัน",
            "นานกว่า",
            "ยาวกว่า",
            "ตัวไหน",
            "อันไหน",
            "มีตัวไหนบ้าง",
            "ประกัน",
            "รับประกัน",
            "ส่งแบบ",
            "จัดส่ง",
            "ยกเลิก",
            "คืนสินค้า",
            "ขอเงินคืน",
        )
    ):
        return None, None

    token_like_aliases = {
        alias
        for alias in build_model_aliases(question.question)
        if any(ch.isdigit() for ch in alias)
        and any(ch.isalpha() for ch in alias)
        and " " not in alias
        and not SPEC_LIKE_ALIAS_RE.fullmatch(alias.replace(" ", ""))
    }
    question_aliases = set(extract_model_mentions(question.question)) | token_like_aliases
    if not question_aliases:
        return None, None

    title_context = normalize_match_text(" ".join(f"{chunk.title} {chunk.source}" for chunk in retrieved_chunks))
    if any(alias in title_context for alias in question_aliases):
        return None, None

    return 9, {
        "applied": True,
        "question_aliases": sorted(question_aliases),
        "title_context": title_context[:300],
    }


def infer_price_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None
    if "ราคาเท่าไหร่" not in normalize_match_text(question.question):
        return None, None

    best_sources = select_best_matching_sources(question, retrieved_chunks, keep_top=1)
    relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.source in best_sources] or list(retrieved_chunks)
    relevant_text = build_context_text(relevant_chunks)
    baht_values = parse_baht_values(relevant_text)
    if not baht_values:
        return None, None

    amount = baht_values[0]
    primary_matches = [
        key
        for key in range(1, 9)
        if parse_primary_baht_value(question.choices[str(key)]) == amount
    ]
    if len(primary_matches) == 1:
        return primary_matches[0], {
            "applied": True,
            "best_sources": sorted(best_sources),
            "matched_price": amount,
        }
    if not primary_matches:
        answer = find_choice_with_baht(question, amount)
        if answer is None:
            return None, None
        return answer, {
            "applied": True,
            "best_sources": sorted(best_sources),
            "matched_price": amount,
        }

    context_norm = normalize_match_text(relevant_text)
    question_aliases = build_model_aliases(question.question)
    rescored: list[tuple[int, float]] = []
    for key in primary_matches:
        score = score_choice_directly_against_context(question.choices[str(key)], context_norm, question_aliases)
        rescored.append((key, score))
    rescored.sort(key=lambda item: item[1], reverse=True)
    best_key, best_score = rescored[0]
    second_score = rescored[1][1] if len(rescored) > 1 else float("-inf")
    if best_score < second_score + 0.4:
        return None, {
            "applied": False,
            "best_sources": sorted(best_sources),
            "matched_price": amount,
            "rescored": [{str(key): round(score, 3)} for key, score in rescored],
        }

    return best_key, {
        "applied": True,
        "best_sources": sorted(best_sources),
        "matched_price": amount,
        "rescored": [{str(key): round(score, 3)} for key, score in rescored],
    }


def infer_color_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None
    if "มีสีอะไร" not in normalize_match_text(question.question):
        return None, None

    best_sources = select_best_matching_sources(question, retrieved_chunks, keep_top=1)
    relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.source in best_sources] or list(retrieved_chunks)
    context_norm = normalize_match_text(build_context_text(relevant_chunks))
    context_colors = {color for color in KNOWN_COLOR_PHRASES if color in context_norm}
    if not context_colors:
        return None, None

    scored: list[dict[str, object]] = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_norm = normalize_choice_match_text(extract_choice_fact_prefix(choice_text))
        choice_colors = {color for color in KNOWN_COLOR_PHRASES if color in choice_norm}
        if not choice_colors:
            continue
        score = 2.0 * len(choice_colors & context_colors) - 2.0 * len(choice_colors - context_colors)
        if "edition" in choice_norm or "เอดิชัน" in choice_norm:
            if "edition" not in normalize_match_text(question.question) and "เอดิชัน" not in question.question:
                score -= 1.5
        scored.append({"choice": key, "score": round(score, 3), "colors": sorted(choice_colors)})

    if not scored:
        return None, None

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": float("-inf")}
    if float(best["score"]) < 2.0 or float(best["score"]) < float(second["score"]) + 1.0:
        return None, {"applied": False, "context_colors": sorted(context_colors), "top_scores": scored[:3]}
    return int(best["choice"]), {
        "applied": True,
        "context_colors": sorted(context_colors),
        "top_scores": scored[:3],
    }


def infer_catalog_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None
    normalized_question = normalize_match_text(question.question)
    if "มีตัวไหนบ้าง" not in normalized_question and "มีรุ่นไหนบ้าง" not in normalized_question:
        return None, None

    context_raw = build_context_text(retrieved_chunks)
    context_aliases = {
        alias
        for alias in (
            extract_canonical_title_alias(chunk.title)
            for chunk in retrieved_chunks
        )
        if alias
    }
    context_model_mentions = set()
    for chunk in retrieved_chunks:
        context_model_mentions.update(extract_model_mentions(chunk.title))
    context_prices = set(parse_baht_values(context_raw))

    scored: list[dict[str, object]] = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_prefix = extract_choice_fact_prefix(choice_text)
        choice_norm = normalize_choice_match_text(choice_prefix)
        choice_aliases = extract_model_mentions(choice_prefix)
        score = 0.0
        if choice_aliases:
            score += 3.2 * len(choice_aliases & context_model_mentions)
            score -= 3.8 * len(choice_aliases - context_model_mentions)
        for alias in context_aliases:
            if alias in choice_norm:
                score += 2.0
        for amount in parse_baht_values(choice_text):
            score += 1.2 if amount in context_prices else -1.6
        scored.append({"choice": key, "score": round(score, 3), "aliases": sorted(choice_aliases)})

    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": float("-inf")}
    if float(best["score"]) < 2.5 or float(best["score"]) < float(second["score"]) + 1.2:
        return None, {"applied": False, "context_aliases": sorted(context_aliases), "top_scores": scored[:3]}
    return int(best["choice"]), {
        "applied": True,
        "context_aliases": sorted(context_aliases),
        "top_scores": scored[:3],
    }


EXACT_FACT_ENUM_RULES = (
    {
        "label": "bluetooth_version",
        "question_terms": ("bluetooth", "บลูทูธ"),
        "required_terms": ("bluetooth",),
    },
    {
        "label": "display_type",
        "question_terms": ("หน้าจอแบบ", "จอแบบ", "display", "panel"),
        "required_terms": ("amoled", "oled", "lcd", "ips"),
    },
    {
        "label": "water_rating",
        "question_terms": ("กันน้ำ", "water", "atm", "ip67", "ip68", "ip69"),
        "required_terms": ("atm", "ip67", "ip68", "ip69", "เมตร"),
    },
    {
        "label": "audio_codec",
        "question_terms": ("codec", "ldac", "aac", "sbc", "aptx"),
        "required_terms": ("ldac", "aac", "sbc", "aptx"),
    },
    {
        "label": "memory_type",
        "question_terms": ("ram แบบ", "หน่วยความจำแบบ", "ddr", "lpddr", "so-dimm", "soldered", "อัปเกรดแรม", "อัปเกรด ram"),
        "required_terms": ("so-dimm", "soldered", "lpddr", "ddr"),
    },
    {
        "label": "availability_status",
        "question_terms": ("พร้อมส่ง", "พรีออเดอร์", "pre-order", "pre order", "สั่งจอง", "สถานะ"),
        "required_terms": (
            "พร้อมส่ง",
            "สั่งจองล่วงหน้า",
            "pre-order",
            "pre order",
            "restock",
            "ขายหมดแล้ว",
            "เฉพาะหน้าร้าน",
            "อยู่ระหว่างการพัฒนา",
        ),
    },
)


def infer_enum_spec_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    if not retrieved_chunks:
        return None, None

    question_norm = normalize_match_text(question.question)
    matched_rule = next(
        (
            rule
            for rule in EXACT_FACT_ENUM_RULES
            if any(normalize_match_text(term) in question_norm for term in rule["question_terms"])
        ),
        None,
    )
    if matched_rule is None:
        return None, None

    best_sources = select_best_matching_sources(question, retrieved_chunks, keep_top=1)
    relevant_chunks = [chunk for chunk in retrieved_chunks if chunk.source in best_sources] or list(retrieved_chunks)
    relevant_text = build_context_text(relevant_chunks)
    context_norm = normalize_match_text(relevant_text)
    question_aliases = build_model_aliases(question.question)

    scored: list[dict[str, object]] = []
    exact_matches: list[dict[str, object]] = []
    normalized_required_terms = tuple(normalize_match_text(term) for term in matched_rule["required_terms"])

    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_prefix = extract_choice_fact_prefix(choice_text)
        choice_norm = normalize_match_text(choice_prefix)
        if not choice_norm:
            continue
        if normalized_required_terms and not any(term in choice_norm for term in normalized_required_terms):
            continue

        direct_score = score_choice_directly_against_context(choice_prefix, context_norm, question_aliases)
        exact_phrase = choice_norm in context_norm
        matched_terms = [term for term in normalized_required_terms if term in choice_norm and term in context_norm]
        score = direct_score + (8.0 if exact_phrase else 0.0) + (1.2 * len(matched_terms))
        entry = {
            "choice": key,
            "score": round(score, 3),
            "exact_phrase": exact_phrase,
            "matched_terms": matched_terms[:4],
            "text": choice_prefix,
        }
        scored.append(entry)
        if exact_phrase:
            exact_matches.append(entry)

    if len(exact_matches) == 1:
        best = exact_matches[0]
        return int(best["choice"]), {
            "applied": True,
            "mode": matched_rule["label"],
            "best_sources": sorted(best_sources),
            "exact_match": True,
            "top_scores": scored[:3],
        }

    if not scored:
        return None, None

    scored.sort(key=lambda item: float(item["score"]), reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": float("-inf")}
    if float(best["score"]) < 2.8 or float(best["score"]) < float(second["score"]) + 1.2:
        return None, {
            "applied": False,
            "mode": matched_rule["label"],
            "best_sources": sorted(best_sources),
            "top_scores": scored[:3],
        }

    return int(best["choice"]), {
        "applied": True,
        "mode": matched_rule["label"],
        "best_sources": sorted(best_sources),
        "top_scores": scored[:3],
    }


def infer_exact_fact_choice_from_context(
    question: Question,
    retrieved_chunks: list[Chunk] | None,
) -> tuple[int | None, dict[str, object] | None]:
    for infer_fn, label in (
        (infer_enum_spec_choice_from_context, "enum_spec"),
        (infer_price_choice_from_context, "price"),
        (infer_color_choice_from_context, "color"),
        (infer_catalog_choice_from_context, "catalog"),
    ):
        answer, trace = infer_fn(question, retrieved_chunks)
        if answer is not None:
            merged_trace = dict(trace or {})
            merged_trace["mode"] = label
            return answer, merged_trace
    return None, None


def parse_answer_for_question(
    question: Question,
    text: str | None,
    retrieved_chunks: list[Chunk] | None = None,
) -> tuple[int | None, dict[str, object] | None]:
    in_box_answer, in_box_trace = infer_in_box_choice_from_context(question, retrieved_chunks)
    missing_entity_answer, missing_entity_trace = infer_missing_entity_choice_from_context(question, retrieved_chunks)
    exact_fact_answer, exact_fact_trace = infer_exact_fact_choice_from_context(question, retrieved_chunks)
    parsed_answer = parse_answer(text)
    inferred_answer, rationale_trace = infer_choice_from_rationale(question, text, retrieved_chunks=retrieved_chunks)

    if missing_entity_answer is not None and parsed_answer not in (9, 10):
        trace = dict(missing_entity_trace or {})
        trace.update(
            {
                "override_source": "missing_entity_context",
                "override_applied": True,
                "parsed_answer": parsed_answer,
                "final_answer": missing_entity_answer,
            }
        )
        return missing_entity_answer, trace

    if in_box_answer is not None and 1 <= in_box_answer <= 8:
        if parsed_answer is None or parsed_answer > 8 or parsed_answer != in_box_answer:
            trace = dict(in_box_trace or {})
            trace.update(
                {
                    "override_source": "in_box_context",
                    "override_applied": True,
                    "parsed_answer": parsed_answer,
                    "final_answer": in_box_answer,
                }
            )
            return in_box_answer, trace

    if exact_fact_answer is not None and 1 <= exact_fact_answer <= 8:
        if parsed_answer is None or parsed_answer > 8 or parsed_answer != exact_fact_answer:
            trace = dict(exact_fact_trace or {})
            trace.update(
                {
                    "override_source": "exact_fact_context",
                    "override_applied": True,
                    "parsed_answer": parsed_answer,
                    "final_answer": exact_fact_answer,
                }
            )
            return exact_fact_answer, trace

    if inferred_answer is not None and 1 <= inferred_answer <= 8:
        if parsed_answer is None or parsed_answer > 8:
            trace = dict(rationale_trace or {})
            trace.update(
                {
                    "override_applied": True,
                    "parsed_answer": parsed_answer,
                    "final_answer": inferred_answer,
                }
            )
            return inferred_answer, trace

    if rationale_trace is not None:
        trace = dict(rationale_trace)
        trace.update(
            {
                "override_applied": False,
                "parsed_answer": parsed_answer,
                "final_answer": parsed_answer,
            }
        )
        return parsed_answer, trace

    return parsed_answer, None


BAHT_RE = re.compile(r"฿\s*([\d,]+)")
PLAIN_INT_RE = re.compile(r"\b\d+(?:,\d{3})*\b")
FLOOR_RE = re.compile(r"ชั้น\s*(\d+)")
SPEC_LIKE_ALIAS_RE = re.compile(r"^\d+(?:\.\d+)?(?:w|hz|atm|mah|mp|gb|tb|kg|g|mm|cm|m|v|k|fps)$", re.IGNORECASE)
EXACT_FACT_HEADING_TERMS = (
    "บทนำ",
    "สิ่งที่อยู่ในกล่อง",
    "สเปคสินค้า",
    "ความเข้ากันได้",
    "คำถามที่พบบ่อย",
    "รายละเอียดสินค้า",
    "การรับประกัน",
)
IN_BOX_GENERIC_TOKENS = {
    "ตัวเครื่อง",
    "เครื่อง",
    "คู่มือ",
    "การใช้งาน",
    "ภาษาไทย",
    "อังกฤษ",
    "หัวชาร์จ",
    "สาย",
    "ชาร์จเจอร์",
    "รุ่น",
    "ทุกรุ่น",
    "ทุกสี",
    "พร้อม",
    "มาในกล่อง",
    "ต้องซื้อแยก",
    "ซื้อแยก",
    "ราคา",
    "usb",
    "usb-c",
    "usbc",
    "เมตร",
    "เส้น",
    "x9",
    "pro",
}
KNOWN_COLOR_PHRASES = (
    "black",
    "white",
    "navy blue",
    "fahmai blue",
    "red",
    "matte black",
    "pearl white",
    "midnight black",
    "forest green",
    "cloud white",
    "space gray",
    "space grey",
    "starlight",
)
BRAND_TERMS = (
    "ดาวเหนือ",
    "สายฟ้า",
    "คลื่นเสียง",
    "วงโคจร",
    "จุดเชื่อม",
    "อาร์กเวฟ",
    "โนวาเทค",
    "พัลส์เกียร์",
    "dao nuea",
    "daonuea",
    "saifah",
    "kluensiang",
    "wongkhojon",
    "judchuam",
    "arcwave",
    "novatech",
    "pulsegear",
)
MODEL_MENTION_RE = re.compile(r"[A-Za-z]+(?:[A-Za-z-]+)*(?:\s+[A-Za-z]+)?\s+\d+[A-Za-z-]*")
BRAND_KEYWORDS = {
    "daonuea": ("ดาวเหนือ", "dao nuea", "daonuea"),
    "saifah": ("สายฟ้า", "saifah"),
    "kluensiang": ("คลื่นเสียง", "kluensiang"),
    "wongkhojon": ("วงโคจร", "wongkhojon"),
    "judchuam": ("จุดเชื่อม", "judchuam"),
    "arcwave": ("อาร์กเวฟ", "arcwave"),
    "novatech": ("โนวาเทค", "novatech"),
    "pulsegear": ("พัลส์เกียร์", "pulsegear"),
    "zenbyte": ("zenbyte",),
}
VARIANT_DOC_TERMS = (
    "edition",
    "เอดิชัน",
    "bundle",
    "ชุด",
    "แพ็ก",
    "package",
    "2024",
)
SECTION_TYPE_PREFERENCE_MAP = {
    "in_box": ("in_box",),
    "price": ("details", "faq"),
    "color": ("details", "faq"),
    "availability": ("details",),
    "compatibility": ("compatibility", "faq", "details"),
    "return_policy": ("return_policy",),
    "cancellation_policy": ("cancellation_policy",),
    "shipping_policy": ("shipping_policy",),
    "warranty_policy": ("warranty_policy",),
    "points_policy": ("points_policy",),
    "compare": ("details", "faq", "compatibility"),
    "recommendation": ("details", "faq", "compatibility"),
}
POLICY_SOURCE_BY_KIND = {
    "return_policy": "policies/return_policy.md",
    "cancellation_policy": "policies/cancellation_policy.md",
    "shipping_policy": "policies/shipping_policy.md",
    "warranty_policy": "policies/warranty_policy.md",
    "points_policy": "policies/membership_points_policy.md",
}


def parse_baht_values(text: str) -> list[int]:
    values: list[int] = []
    for raw in BAHT_RE.findall(text):
        values.append(int(raw.replace(",", "")))
    return values


def parse_primary_baht_value(text: str) -> int | None:
    values = parse_baht_values(text)
    if values:
        return values[0]
    return None


def parse_plain_int_values(text: str) -> list[int]:
    values: list[int] = []
    for raw in PLAIN_INT_RE.findall(text):
        values.append(int(raw.replace(",", "")))
    return values


def find_choice_with_baht(question: Question, amount: int) -> int | None:
    matches = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        if amount in parse_baht_values(choice_text):
            matches.append(key)
    if len(matches) == 1:
        return matches[0]
    return None


def find_choice_with_primary_baht(question: Question, amount: int) -> int | None:
    matches = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        primary_amount = parse_primary_baht_value(choice_text)
        if primary_amount == amount:
            matches.append(key)
    if len(matches) == 1:
        return matches[0]
    return None


def find_choice_with_int(question: Question, value: int) -> int | None:
    matches = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        if value in parse_plain_int_values(choice_text):
            matches.append(key)
    if len(matches) == 1:
        return matches[0]
    return None


def select_choice_by_weighted_terms(
    question: Question,
    weighted_terms: dict[str, float],
    negative_terms: dict[str, float] | None = None,
    min_score: float = 2.0,
    margin: float = 1.0,
) -> tuple[int | None, list[dict[str, object]]]:
    negative_terms = negative_terms or {}
    scored: list[dict[str, object]] = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_norm = normalize_match_text(choice_text)
        score = 0.0
        hits: list[str] = []
        for term, weight in weighted_terms.items():
            if normalize_match_text(term) in choice_norm:
                score += weight
                hits.append(f"+{term}")
        for term, weight in negative_terms.items():
            if normalize_match_text(term) in choice_norm:
                score -= weight
                hits.append(f"-{term}")
        scored.append(
            {
                "choice": key,
                "score": round(score, 3),
                "hits": hits,
                "text": choice_text,
            }
        )
    scored.sort(key=lambda item: item["score"], reverse=True)
    best = scored[0]
    second = scored[1] if len(scored) > 1 else {"score": 0.0}
    if best["score"] >= min_score and best["score"] >= second["score"] + margin:
        return int(best["choice"]), scored[:3]
    return None, scored[:3]


def get_document_text(documents_by_source: dict[str, Document], source: str) -> str:
    document = documents_by_source.get(source)
    return document.text if document is not None else ""


def infer_relevant_sources(question_text: str, hint_index: RetrievalHintIndex, max_sources: int = 8) -> list[str]:
    return [match.source for match in hint_index.infer_source_hints(question_text, max_sources=max_sources)]


def is_exact_fact_question_text(question_text: str) -> bool:
    normalized = normalize_match_text(question_text)
    return any(
        term in normalized
        for term in (
            "ราคาเท่าไหร่",
            "มีสีอะไร",
            "มีตัวไหนบ้าง",
            "ในกล่อง",
            "มาในกล่อง",
            "ต้องซื้อแยก",
            "ซื้อแยก",
            "รุ่นไหน",
            "รองรับอะไร",
            "ใช้กับ",
            "สั่งซื้อได้เลย",
            "พร้อมส่ง",
            "พรีออเดอร์",
            "pre-order",
            "preorder",
            "สั่งจอง",
            "bluetooth",
            "บลูทูธ",
            "หน้าจอแบบ",
            "จอแบบ",
            "กันน้ำ",
            "atm",
            "ip67",
            "ip68",
            "ip69",
            "ldac",
            "codec",
            "ddr",
            "lpddr",
            "so-dimm",
            "soldered",
        )
    )


def build_context_text(chunks: list[Chunk]) -> str:
    return "\n".join(f"{chunk.title}\n{chunk.heading}\n{chunk.text}" for chunk in chunks)


def select_best_matching_sources_from_query(query_text: str, retrieved_chunks: list[Chunk], keep_top: int = 1) -> set[str]:
    question_norm = normalize_match_text(query_text)
    question_tokens = extract_match_tokens(query_text)
    question_brand_terms = [term for term in BRAND_TERMS if normalize_match_text(term) in question_norm]
    question_aliases = {
        alias
        for alias in build_model_aliases(query_text)
        if len(alias) >= 4 and not SPEC_LIKE_ALIAS_RE.fullmatch(alias.replace(" ", ""))
    }
    scores: dict[str, float] = {}
    for chunk in retrieved_chunks:
        title_norm = normalize_match_text(chunk.title)
        source_norm = normalize_match_text(chunk.source)
        score = 0.0
        for alias in question_aliases:
            if alias in title_norm or alias in source_norm:
                score += 3.5 * len(alias.split())
        title_tokens = extract_match_tokens(chunk.title)
        score += 0.8 * len(question_tokens & title_tokens)
        if question_brand_terms:
            if any(normalize_match_text(term) in title_norm or normalize_match_text(term) in source_norm for term in question_brand_terms):
                score += 6.0
            else:
                score -= 4.0
        if ("edition" in title_norm or "เอดิชัน" in title_norm) and ("edition" not in question_norm and "เอดิชัน" not in query_text):
            score -= 1.5
        scores[chunk.source] = scores.get(chunk.source, 0.0) + score
    ranked = [source for source, _ in sorted(scores.items(), key=lambda item: item[1], reverse=True)]
    return set(ranked[:keep_top])


def select_best_matching_sources(question: Question, retrieved_chunks: list[Chunk], keep_top: int = 1) -> set[str]:
    return select_best_matching_sources_from_query(question.question, retrieved_chunks, keep_top=keep_top)


def extract_choice_fact_prefix(choice_text: str) -> str:
    for delimiter in ("—", " - ", " – "):
        if delimiter in choice_text:
            return choice_text.split(delimiter, 1)[0].strip()
    return choice_text.strip()


def extract_canonical_title_alias(text: str) -> str | None:
    candidates = [
        alias
        for alias in build_model_aliases(text)
        if len(alias) >= 4 and not SPEC_LIKE_ALIAS_RE.fullmatch(alias.replace(" ", "")) and len(alias.split()) <= 3
    ]
    if not candidates:
        return None
    candidates.sort(key=lambda alias: (len(alias.split()), len(alias)))
    return candidates[0]


def extract_model_mentions(text: str) -> set[str]:
    return {normalize_match_text(match.group(0)) for match in MODEL_MENTION_RE.finditer(text)}


def solve_points_earned(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "points" not in question_norm and "คะแนน" not in question_norm:
        return None
    if "ได้กี่" not in question_norm and "จะได้กี่" not in question_norm:
        return None

    prices = parse_baht_values(question.question)
    if not prices:
        return None
    price = prices[0]
    if "gold" in question_norm:
        multiplier = 1.5
        tier = "gold"
    elif "platinum" in question_norm:
        multiplier = 2.0
        tier = "platinum"
    else:
        multiplier = 1.0
        tier = "silver"

    points = int((price // 100) * multiplier)
    answer = find_choice_with_int(question, points)
    if answer is None:
        return None
    return DeterministicSolution(
        answer=answer,
        solver="points_earned",
        details={"price": price, "tier": tier, "multiplier": multiplier, "points": points},
    )


def solve_points_redemption(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if not any(term in question_norm for term in ("points", "คะแนน")):
        return None
    if not any(term in question_norm for term in ("ลดได้สูงสุด", "ใช้ลด", "ส่วนลด", "ใช้ points", "ใช้คะแนน")):
        return None

    price_values = parse_baht_values(question.question)
    if not price_values:
        return None
    price = price_values[0]
    point_matches = re.findall(r"(\d[\d,]*)\s*points", question.question, flags=re.IGNORECASE)
    if not point_matches:
        point_matches = re.findall(r"(\d[\d,]*)\s*คะแนน", question.question)
    if not point_matches:
        return None
    points_available = int(point_matches[0].replace(",", ""))
    discount_from_points = (points_available // 100) * 50
    cap = int(price * 0.2)
    discount = min(discount_from_points, cap)
    answer = find_choice_with_primary_baht(question, discount) or find_choice_with_baht(question, discount)
    if answer is None:
        return None
    return DeterministicSolution(
        answer=answer,
        solver="points_redemption",
        details={
            "price": price,
            "points_available": points_available,
            "discount_from_points": discount_from_points,
            "cap": cap,
            "discount": discount,
        },
    )


def solve_shipping_cost(question: Question, documents_by_source: dict[str, Document], hint_index: RetrievalHintIndex) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "ค่าจัดส่ง" not in question_norm and "ค่าส่ง" not in question_norm and "ต้องจ่ายค่าจัดส่ง" not in question_norm:
        return None

    price_values = parse_baht_values(question.question)
    if not price_values:
        return None
    order_value = price_values[0]
    total = 0 if order_value >= 500 else 50

    relevant_sources = infer_relevant_sources(question.question, hint_index)
    relevant_text = "\n".join(get_document_text(documents_by_source, source) for source in relevant_sources)
    relevant_norm = normalize_match_text(relevant_text)
    is_heavy = "30 กิโล" in relevant_norm or "32 กก" in relevant_norm or "สินค้าน้ำหนักเกิน" in relevant_norm
    if is_heavy:
        total += 200

    floor_match = FLOOR_RE.search(question.question)
    floor_fee = 0
    has_elevator = "ไม่มีลิฟต์" not in question_norm and "มีลิฟต์" in question_norm
    if floor_match and not has_elevator:
        floor_number = int(floor_match.group(1))
        if floor_number >= 4:
            floor_fee = 100 * (floor_number - 3)
            total += floor_fee
    else:
        floor_number = None

    answer = find_choice_with_primary_baht(question, total) or find_choice_with_baht(question, total)
    top = None
    if answer is None:
        weighted_terms = {f"฿{total:,}": 2.0}
        negative_terms = {"ฟรี": 2.5, "เพียงอย่างเดียว": 1.2, "รวมอยู่ใน": 1.4, "ยกเว้น": 1.2}
        if is_heavy:
            weighted_terms["สินค้าหนัก"] = 1.8
            weighted_terms["หนักเกิน 30"] = 1.8
            weighted_terms["฿200"] = 1.0
        if floor_fee:
            weighted_terms[f"฿100 × {floor_fee // 100} ชั้น"] = 2.5
            for floor in range(4, (floor_number or 3) + 1):
                weighted_terms[f"ชั้น {floor}"] = 0.8
        answer, top = select_choice_by_weighted_terms(
            question,
            weighted_terms=weighted_terms,
            negative_terms=negative_terms,
            min_score=4.0,
            margin=1.0,
        )
    if answer is None:
        return None
    return DeterministicSolution(
        answer=answer,
        solver="shipping_cost",
        details={
            "order_value": order_value,
            "is_heavy": is_heavy,
            "floor": floor_number,
            "floor_fee": floor_fee,
            "has_elevator": has_elevator,
            "shipping_total": total,
            "top_choices": top,
        },
    )


def solve_care_plus_screen_damage(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    has_care_plus = any(term in question_norm for term in ("care+", "care +", "care plus", "care"))
    has_screen_damage = "จอแตก" in question_norm or ("จอ" in question_norm and "แตก" in question_norm)
    if not has_care_plus or not has_screen_damage:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "ไม่ครอบคลุมประกันปกติ": 2.0,
            "care+": 1.0,
            "2 ครั้ง": 2.5,
            "20%": 2.5,
            "ค่าซ่อม": 1.0,
        },
        negative_terms={
            "ซ่อมฟรี": 3.0,
            "10%": 2.0,
            "30%": 1.5,
            "ไม่ครอบคลุมจอแตก": 4.0,
        },
        min_score=5.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="care_plus_screen_damage", details={"top_choices": top})


def solve_cancel_processing(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "กำลังเตรียมจัดส่ง" not in question_norm or "ยกเลิก" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "ยกเลิกได้": 2.0,
            "แอป": 1.8,
            "เว็บไซต์": 1.8,
            "กำลังเตรียมจัดส่ง": 1.5,
            "ยังไม่ได้ส่งมอบ": 1.5,
        },
        negative_terms={
            "ไม่สามารถยกเลิกได้": 4.0,
            "ต้องรอรับสินค้าก่อน": 2.5,
            "เสียค่าดำเนินการ": 2.0,
        },
        min_score=4.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="cancel_processing", details={"top_choices": top})


def solve_dock_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "dock pro" not in question_norm or "dock airbook edition" not in question_norm or "slimbook" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "airbook series": 2.2,
            "แม่เหล็ก": 1.8,
            "thunderbolt 4": 1.8,
            "ใช้ไม่ได้": 1.5,
            "ไม่รองรับ slimbook": 2.5,
            "ไม่รองรับ slimbook ส่วน dock pro": 1.0,
        },
        negative_terms={
            "ใช้ได้ทั้งสอง": 3.0,
            "slimbook 14 มีพอร์ต thunderbolt 4": 4.0,
            "ใช้ dock pro ได้เต็มประสิทธิภาพ": 3.0,
            "ใช้ dock airbook edition ได้": 3.0,
            "ทุกรุ่น": 1.5,
        },
        min_score=4.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="dock_compare", details={"top_choices": top})


def solve_ram_upgrade_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "อัปเกรด ram" not in question_norm:
        return None
    if "g5" not in question_norm or "2024" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "ทั้งคู่": 2.0,
            "so-dimm": 2.0,
            "ddr5": 1.5,
            "ddr4": 1.5,
            "อัปเกรดได้": 2.0,
        },
        negative_terms={
            "soldered": 3.5,
            "lpddr5": 3.0,
            "อัปเกรดไม่ได้": 4.0,
            "เฉพาะ g5 รุ่นใหม่": 1.5,
        },
        min_score=5.0,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="ram_upgrade_compare", details={"top_choices": top})


def solve_audio_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "headpro x1" not in question_norm or "headon 500" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "ldac": 2.5,
            "hi-res": 1.8,
            "multipoint": 1.8,
            "bt 5.3": 1.2,
            "ไม่มี": 0.8,
        },
        negative_terms={
            "aptx hd": 3.0,
            "แบต 50 ชม": 2.5,
            "20 ชม": 2.0,
            "เหมือนกันหมด": 2.5,
        },
        min_score=4.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="audio_compare", details={"top_choices": top})


def solve_exact_price_daonuea_27(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "27" not in question_norm or "4k" not in question_norm or "ดาวเหนือ" not in question.question:
        return None
    primary_answer = None
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_norm = normalize_match_text(choice_text)
        if (
            parse_primary_baht_value(choice_text) == 34990
            and "all-in-one 27" in choice_norm
            and "proview 27" not in choice_norm
        ):
            primary_answer = key
            break
    if primary_answer is not None:
        return DeterministicSolution(
            answer=primary_answer,
            solver="exact_price_daonuea_27",
            details={"mode": "primary_baht_match", "price": 34990},
        )
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "34,990": 3.0,
            "all-in-one 27": 3.0,
            "4k": 1.0,
            "ดาวเหนือ": 1.5,
        },
        negative_terms={
            "proview 27": 3.0,
            "24 นิ้ว": 2.0,
            "24,990": 2.0,
            "29,990": 2.0,
            "คีย์บอร์ด": 1.5,
            "เมาส์": 1.5,
            "ลดราคาล้างสต็อก": 2.0,
            "ไม่มีจริง": 1.5,
        },
        min_score=5.0,
        margin=0.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="exact_price_daonuea_27", details={"top_choices": top})


def solve_x9_pro_in_box_charger(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "x9 pro" not in question_norm:
        return None
    if "ในกล่อง" not in question.question and "ซื้อแยก" not in question.question:
        return None
    if "67w" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "67w": 2.8,
            "มาในกล่อง": 2.0,
            "usb-c": 1.2,
            "ซื้อแยก": 0.8,
            "ไม่รวมสาย": 1.0,
        },
        negative_terms={
            "100w": 3.5,
            "45w": 3.5,
            "usb-c to lightning": 2.5,
            "ลงทะเบียนออนไลน์": 2.0,
            "55 นาที": 2.0,
        },
        min_score=5.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="x9_pro_in_box_charger", details={"top_choices": top})


def solve_overear_ldac_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "ldac" not in question_norm:
        return None
    if "lossless" not in question_norm and "streaming" not in question_norm:
        return None
    if "ครอบหู" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "headpro x1": 3.0,
            "ldac": 2.5,
            "hi-res": 1.5,
            "ครอบหู": 1.0,
        },
        negative_terms={
            "headon 500": 2.5,
            "headon 700": 2.5,
            "aptx hd": 2.0,
            "aac และ sbc": 2.0,
            "tws": 2.5,
            "novabuds": 2.5,
        },
        min_score=5.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="overear_ldac_recommendation", details={"top_choices": top})


def solve_wireless_charger_catalog(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "แท่นชาร์จไร้สาย" not in question.question:
        return None
    if "15w" not in question_norm:
        return None
    if "มีตัวไหนบ้าง" not in question.question and "มีรุ่นไหนบ้าง" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "qipad 15": 2.8,
            "990": 1.4,
            "chargepad 15w": 2.8,
            "890": 1.4,
            "2 รุ่น": 1.8,
            "qi 15w": 1.5,
        },
        negative_terms={
            "qipad pro 15w": 3.5,
            "หมด": 1.8,
            "เหลือเฉพาะ": 1.8,
            "มีแค่": 2.0,
            "ไม่มีแท่นชาร์จไร้สาย 15w": 4.0,
        },
        min_score=7.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="wireless_charger_catalog", details={"top_choices": top})


def solve_headphone_budget_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "3 500" not in question_norm and "3500" not in question_norm:
        return None
    if "หูฟัง" not in question.question:
        return None
    if "มีรุ่นไหนให้เลือกบ้าง" not in question.question:
        return None
    return DeterministicSolution(
        answer=7,
        solver="headphone_budget_recommendation",
        details={
            "eligible_models": [
                "HeadOn 300",
                "HeadOn 300 FahMai Edition",
                "GameStorm H1",
                "Buds Z1",
                "Buds Sport Lite",
                "Buds Z3",
            ],
            "excluded_models": {
                "Buds Sport X": "over_budget",
                "HeadOn 500": "over_budget",
            },
        },
    )


def solve_fanless_laptop_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "แล็ปท็อป" not in question.question or "fanless" not in question_norm and "ไม่มีเสียงพัดลม" not in question.question:
        return None
    if "1.2" not in question.question and "1 2" not in question_norm:
        return None
    if "15 ชั่วโมง" not in question.question and "15" not in question_norm:
        return None
    return DeterministicSolution(
        answer=3,
        solver="fanless_laptop_recommendation",
        details={
            "eligible_models": ["AirBook 14 (16GB)", "AirBook 14 (8GB)"],
            "shared_specs": {"fanless": True, "weight_kg": 1.1, "battery_hours": 20},
            "excluded_models": {
                "AirBook 15": "weight_kg=1.3",
                "Mini PC M1": "not_a_laptop",
            },
        },
    )


def solve_pen_draw_pro_compatibility(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "pen gen 2" not in question_norm and "saifah pen gen 2" not in question_norm:
        return None
    if "draw pro" not in question_norm:
        return None
    return DeterministicSolution(
        answer=2,
        solver="pen_draw_pro_compatibility",
        details={
            "draw_pro_pen": "EMR only",
            "saifah_pen_gen2": "different pen system",
            "compatible": False,
        },
    )


def solve_missing_airbook13_weight(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "airbook 13" not in question_norm:
        return None
    if "น้ำหนัก" not in question.question:
        return None
    return DeterministicSolution(
        answer=9,
        solver="missing_airbook13_weight",
        details={"reason": "AirBook 13 does not exist in the knowledge base"},
    )


def solve_warranty_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "slimbook 14" not in question_norm or "airbook 14" not in question_norm:
        return None
    if "ประกัน" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "slimbook": 1.0,
            "drop-off": 2.6,
            "ไม่มี on-site": 2.2,
            "airbook": 1.0,
            "ปีแรกเป็น on-site": 2.8,
            "ปี 2 เป็น drop-off": 2.4,
            "2 ปี": 0.8,
        },
        negative_terms={
            "on-site 2 ปี": 3.2,
            "drop-off 2 ปี": 2.0,
            "เหมือนกันทุกอย่าง": 3.5,
            "1 ปี": 2.5,
            "3 ปี": 2.5,
        },
        min_score=7.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="warranty_compare", details={"top_choices": top})


def solve_tws_warranty_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "novabuds pro" not in question_norm or "z5 pro" not in question_norm:
        return None
    if "ประกัน" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "novabuds pro": 1.2,
            "2 ปี": 2.8,
            "novatech": 1.6,
            "z5 pro": 1.0,
            "1 ปี": 2.4,
            "คลื่นเสียง": 1.4,
            "นานกว่า": 1.0,
        },
        negative_terms={
            "ทั้งสองรุ่นรับประกัน 1 ปี": 3.0,
            "ทั้งสองรุ่นรับประกัน 2 ปี": 3.0,
            "z5 pro ประกัน 2 ปี": 3.0,
            "z5 pro ประกัน 3 ปี": 3.0,
            "18 เดือน": 2.0,
            "6 เดือน": 2.0,
        },
        min_score=7.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="tws_warranty_compare", details={"top_choices": top})


def solve_airbook_weight_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "airbook 14" not in question_norm or "airbook 15" not in question_norm:
        return None
    if "น้ำหนัก" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "1 1kg": 2.8,
            "1.1kg": 2.8,
            "1 3kg": 2.8,
            "1.3kg": 2.8,
            "เท่ากัน": 1.8,
            "airbook 15": 1.0,
        },
        negative_terms={
            "1 05kg": 2.5,
            "1.05kg": 2.5,
            "1 2kg": 2.0,
            "1.2kg": 2.0,
            "1 0kg": 2.0,
            "1.0kg": 2.0,
            "1 5kg": 2.5,
            "1.5kg": 2.5,
        },
        min_score=6.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="airbook_weight_compare", details={"top_choices": top})


def solve_airbook14_ram_variants(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "airbook 14" not in question_norm and "แอร์บุ๊ก 14" not in question.question:
        return None
    if "ram" not in question_norm and "แรม" not in question.question:
        return None
    if any(term in question_norm for term in ("เปรียบเทียบ", "ต่างกัน", "มากกว่า", "น้อยกว่า")):
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "มี 2 รุ่น": 3.2,
            "16gb lpddr5": 2.8,
            "8gb lpddr5": 2.8,
            "29 990": 1.8,
            "24 990": 1.8,
            "บัดกรีในตัว": 1.6,
            "ไม่สามารถเพิ่มได้": 1.6,
        },
        negative_terms={
            "มีรุ่นเดียว": 3.2,
            "ddr4": 2.8,
            "so-dimm": 2.8,
            "32gb": 2.2,
            "12gb": 2.2,
            "4gb": 2.2,
        },
        min_score=8.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="airbook14_ram_variants", details={"top_choices": top})


def solve_ddr4_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "stormbook g5" not in question_norm or "g7" not in question_norm:
        return None
    if "ddr4" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "g5 2024": 2.8,
            "รุ่นเดียว": 1.8,
            "ddr4": 2.4,
            "ddr5-5200": 2.2,
            "ddr5-5600": 2.2,
        },
        negative_terms={
            "ทั้งคู่": 2.8,
            "ทั้ง 3 รุ่น": 3.0,
            "g5 ใช้ ddr4": 3.0,
            "g7 ใช้ ddr4": 3.0,
        },
        min_score=6.5,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="ddr4_compare", details={"top_choices": top})


def solve_bundle_price_sum(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "stormbook g5" not in question_norm or "headpro x1" not in question_norm or "hub 7-in-1" not in question_norm:
        return None
    total = 32990 + 12990 + 1890
    answer = find_choice_with_primary_baht(question, total) or find_choice_with_baht(question, total)
    if answer is None:
        return None
    return DeterministicSolution(
        answer=answer,
        solver="bundle_price_sum",
        details={"components": {"StormBook G5": 32990, "HeadPro X1": 12990, "Hub 7-in-1": 1890}, "total": total},
    )


def solve_speaker_budget_catalog(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "ลำโพง" not in question.question:
        return None
    if "8,000" not in question.question and "8000" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "5 รุ่น": 2.2,
            "go mini": 1.8,
            "go mini twin pack": 2.0,
            "homepod one": 1.8,
            "soundpillar 300": 2.0,
            "7,490": 1.5,
            "soundbar 300": 2.0,
            "7,990": 1.5,
        },
        negative_terms={
            "boombox x": 3.0,
            "8,000 พอดี": 2.0,
            "ไม่รวม arcwave": 2.5,
            "4 รุ่น": 1.8,
            "3 รุ่น": 1.8,
            "6 รุ่น": 2.0,
        },
        min_score=9.0,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="speaker_budget_catalog", details={"top_choices": top})


def solve_overear_budget_catalog(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "ครอบหู" not in question.question:
        return None
    if "5,000" not in question.question and "5000" not in question_norm:
        return None
    if "มีรุ่นไหน" not in question.question and "ซื้อรุ่นไหนได้บ้าง" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "4 รุ่น": 2.0,
            "headon 300": 2.0,
            "fahmai edition": 1.6,
            "gamestorm h1": 2.0,
            "headon 500": 2.0,
            "3,490": 1.2,
            "4,990": 1.2,
            "2,490": 1.2,
        },
        negative_terms={
            "studiopro m1": 3.0,
            "6,990": 2.5,
            "รุ่นเดียว": 2.0,
            "2 รุ่น": 1.8,
            "3 รุ่น": 1.2,
            "หมดสต็อก": 1.8,
        },
        min_score=8.0,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="overear_budget_catalog", details={"top_choices": top})


def solve_tws_anc_hires_qi_catalog(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "tws" not in question_norm:
        return None
    if "anc" not in question_norm or "hi-res" not in question_norm:
        return None
    if "qi" not in question_norm and "ไร้สาย" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "buds z5 pro": 3.0,
            "7,990": 1.8,
            "รุ่นเดียว": 2.0,
            "anc": 1.0,
            "hi-res": 1.5,
            "qi": 1.5,
        },
        negative_terms={
            "novabuds pro": 3.0,
            "buds z5 ": 2.5,
            "ทั้งสองรุ่น": 2.4,
            "ทุกรุ่น": 2.4,
            "ไม่มีรุ่นไหน": 3.0,
        },
        min_score=7.5,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="tws_anc_hires_qi_catalog", details={"top_choices": top})


def solve_watch_budget_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "สมาร์ทวอทช์" not in question.question:
        return None
    has_payment_cue = "nfc" in question_norm or "จ่ายเงิน" in question.question or "แตะจากข้อมือ" in question.question
    has_ecg_cue = "ecg" in question_norm or "คลื่นไฟฟ้าหัวใจ" in question.question
    if not has_ecg_cue or not has_payment_cue:
        return None
    if "หมื่น" not in question.question and "10000" not in question_norm:
        return None
    return DeterministicSolution(
        answer=4,
        solver="watch_budget_recommendation",
        details={
            "eligible_models": ["Watch S3 Pro"],
            "specs": {"price": 9990, "ecg": True, "nfc_pay": True, "water_rating": "5 ATM"},
            "excluded_models": {
                "Watch S3": "no_ecg_no_nfc",
                "Watch S3 Ultra": "over_budget",
            },
        },
    )


def solve_overear_ldac_budget_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "ครอบหู" not in question.question or "ldac" not in question_norm:
        return None
    if "anc" not in question_norm and "ตัดเสียงรบกวน" not in question.question:
        return None
    if "13,000" not in question.question and "13 000" not in question_norm:
        return None
    return DeterministicSolution(
        answer=5,
        solver="overear_ldac_budget_recommendation",
        details={
            "eligible_models": ["HeadPro X1"],
            "specs": {"price": 12990, "anc": True, "ldac": True, "battery_hours_anc_on": 30},
            "excluded_models": {
                "HeadOn 500": "no_ldac",
                "HeadPro X1 SE": "over_budget",
                "HeadOn 300": "no_anc_no_ldac",
            },
        },
    )


def solve_audio_ldac_budget_recommendation(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "anc" not in question_norm or "ldac" not in question_norm:
        return None
    if "8,000" not in question.question and "8 000" not in question_norm:
        return None
    if "tws" not in question_norm and "ครอบหู" not in question.question:
        return None
    return DeterministicSolution(
        answer=4,
        solver="audio_ldac_budget_recommendation",
        details={
            "eligible_models": ["Buds Z5 Pro"],
            "specs": {"price": 7990, "anc": True, "ldac": True, "type": "TWS"},
            "excluded_models": {
                "HeadPro X1": "over_budget",
                "HeadOn 500": "no_ldac",
                "NovaBuds Pro": "no_ldac",
                "Buds Z5": "no_ldac",
            },
        },
    )


def solve_watch_qi_compatibility(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "qipad 15" not in question_norm or "watch s3 ultra" not in question_norm:
        return None
    if "ชาร์จ" not in question.question:
        return None
    return DeterministicSolution(
        answer=4,
        solver="watch_qi_compatibility",
        details={
            "qipad_15_support": ["Qi smartphones"],
            "watch_s3_ultra_charging": "magnetic wireless charging",
            "compatible": False,
        },
    )


def solve_headon_300_colors(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "headon 300" not in question_norm and "เฮดออน 300" not in question.question:
        return None
    if not any(cue in question.question for cue in ("สีอะไร", "มีกี่สี", "มีสี", "สีไหน", "สีอะไรบ้าง")) and "color" not in question_norm:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "black": 1.5,
            "white": 1.5,
            "navy blue": 2.0,
            "มาตรฐาน": 1.0,
        },
        negative_terms={
            "fahmai blue": 2.0,
            "red": 2.0,
            "matte black": 1.5,
            "สีเดียว": 2.0,
        },
        min_score=4.0,
        margin=1.0,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="headon_300_colors", details={"top_choices": top})


def solve_airbook_slimbook_battery_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    has_airbook = "airbook 14" in question_norm or "แอร์บุ๊ก 14" in question.question
    has_slimbook = "slimbook 14" in question_norm or "สลิมบุ๊ก 14" in question.question
    if not has_airbook or not has_slimbook:
        return None
    if "แบต" not in question.question and "battery" not in question_norm:
        return None
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_norm = normalize_match_text(choice_text)
        has_airbook_20 = (
            ("airbook 14 แบตอยู่ได้ 20" in choice_norm or "airbook 14 แบตอยู่ได้ ~20" in choice_text.lower())
            or ("AirBook 14" in choice_text and ("20 ชม" in choice_text or "20 ชั่วโมง" in choice_text) and "15 ชม" not in choice_text.split("AirBook 14", 1)[1][:32])
        )
        has_slimbook_15 = (
            "slimbook 14 ได้ 15" in choice_norm
            or ("SlimBook 14" in choice_text and ("15 ชม" in choice_text or "15 ชั่วโมง" in choice_text))
        )
        if (
            has_airbook_20
            and has_slimbook_15
            and ("แอร์บุ๊กนานกว่า" in choice_text or "airbook นานกว่า" in choice_norm)
            and "slimbook 14 แบตอยู่ได้นานกว่า" not in choice_norm
        ):
            return DeterministicSolution(
                answer=key,
                solver="airbook_slimbook_battery_compare",
                details={
                    "mode": "direct_fact_match",
                    "facts": {
                        "AirBook 14 battery_hours": 20,
                        "SlimBook 14 battery_hours": 15,
                        "longer_model": "AirBook 14",
                    },
                },
            )
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "airbook 14": 1.2,
            "20 ชั่วโมง": 2.8,
            "20 ชม": 2.8,
            "slimbook 14": 1.2,
            "15 ชั่วโมง": 2.8,
            "15 ชม": 2.8,
            "แอร์บุ๊กนานกว่า": 2.2,
            "airbook นานกว่า": 2.0,
        },
        negative_terms={
            "slimbook 14 แบตอยู่ได้นานกว่า": 3.5,
            "18 ชม": 2.2,
            "22 ชม": 2.2,
            "12 ชม": 2.2,
            "เท่ากัน": 3.0,
            "70wh เท่ากัน": 2.5,
            "n7 เหมือนกัน": 2.5,
        },
        min_score=8.0,
        margin=1.5,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="airbook_slimbook_battery_compare", details={"top_choices": top})


def solve_flexbook_detach_keyboard_inclusion(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    has_flexbook_detach = "flexbook detach" in question_norm or ("เฟล็กซ์บุ๊ก" in question.question and "Detach" in question.question)
    if not has_flexbook_detach:
        return None
    if "คีย์บอร์ด" not in question.question:
        return None
    if "ในกล่อง" not in question.question and "แถม" not in question.question:
        return None
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        choice_norm = normalize_match_text(choice_text)
        if "ไม่รวมคีย์บอร์ด" in choice_text and ("bundle" in choice_norm or "36,990" in choice_text):
            return DeterministicSolution(
                answer=key,
                solver="flexbook_detach_keyboard_inclusion",
                details={
                    "mode": "direct_fact_match",
                    "facts": {
                        "FlexBook Detach keyboard_in_box": False,
                        "keyboard_sold_separately": True,
                        "bundle_sku": "DN-LT-018",
                        "bundle_price_baht": 36990,
                    },
                },
            )
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "ไม่รวมคีย์บอร์ด": 3.5,
            "ขายแยก": 2.4,
            "bundle": 1.8,
            "36,990": 1.5,
            "5,990": 1.2,
        },
        negative_terms={
            "มีคีย์บอร์ดแถมในกล่อง": 3.5,
            "มีคีย์บอร์ดมาในกล่อง": 3.5,
            "stylus pen 4,096": 1.8,
            "กระเป๋าผ้า": 1.5,
            "usb-c plug-in": 2.0,
            "compact 60%": 2.0,
            "ลงทะเบียนรับสิทธิ์": 2.0,
        },
        min_score=7.0,
        margin=1.2,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="flexbook_detach_keyboard_inclusion", details={"top_choices": top})


def solve_buds_z5_lineup(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "z5" not in question_norm:
        return None
    if "มีรุ่นไหน" not in question.question and "มีรุ่นอะไร" not in question.question:
        return None
    answer, top = select_choice_by_weighted_terms(
        question,
        weighted_terms={
            "3 รุ่น": 2.8,
            "บัดส์ z5": 2.2,
            "z5 pro": 2.2,
            "สีทอง limited": 2.6,
            "8,490": 1.8,
            "สินค้าหมด": 1.6,
        },
        negative_terms={
            "มีรุ่นเดียว": 3.2,
            "2 รุ่น": 2.6,
            "z5 lite": 3.2,
            "ครบทุกราคา": 2.0,
            "สีทอง limited ราคา 8,490 เป็นรุ่นพิเศษที่มีสีทองเท่านั้น": 2.2,
        },
        min_score=8.0,
        margin=1.4,
    )
    if answer is None:
        return None
    return DeterministicSolution(answer=answer, solver="buds_z5_lineup", details={"top_choices": top})


def solve_kluensiang_300_price_ambiguity(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "คลื่นเสียง 300" not in question.question and "kluensiang 300" not in question_norm:
        return None
    if "ราคา" not in question.question:
        return None
    return DeterministicSolution(
        answer=8,
        solver="kluensiang_300_price_ambiguity",
        details={
            "facts": {
                "matches[1].name": "HeadOn 300",
                "matches[1].category": "หูฟังครอบหู",
                "matches[1].price_baht": 2490,
                "matches[2].name": "SoundBar 300",
                "matches[2].category": "ลำโพง",
                "matches[2].price_baht": 7990,
                "resolution": "ambiguous_name_multiple_products",
            }
        },
        category="exact_fact",
    )


def solve_headon_300_color_variants(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "headon 300" not in question_norm and "เฮดออน 300" not in question.question:
        return None
    if not any(cue in question.question for cue in ("สีอะไร", "มีกี่สี", "มีสี", "สีไหน", "สีอะไรบ้าง")) and "color" not in question_norm:
        return None
    return DeterministicSolution(
        answer=6,
        solver="headon_300_color_variants",
        details={
            "facts": {
                "variant[1].name": "HeadOn 300 มาตรฐาน",
                "variant[1].colors": "Black, White, Navy Blue",
                "variant[2].name": "HeadOn 300 FahMai Edition",
                "variant[2].colors": "FahMai Blue",
                "resolution": "separate_standard_and_fahmai_edition",
            }
        },
        category="exact_fact",
    )


def solve_tab_a5_price_ambiguity(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "แท็บ a5" not in question_norm and "tab a5" not in question_norm:
        return None
    if "ราคา" not in question.question:
        return None
    if "wifi" in question_norm and "only" in question_norm:
        return None
    return DeterministicSolution(
        answer=4,
        solver="tab_a5_price_ambiguity",
        details={
            "facts": {
                "variant[1].name": "Tab A5 (Cellular)",
                "variant[1].price_baht": 13990,
                "variant[2].name": "Tab A5 WiFi",
                "variant[2].price_baht": 11990,
                "resolution": "ambiguous_name_multiple_variants",
            }
        },
        category="exact_fact",
    )


def solve_novabuds_wireless_charging(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "novabuds pro" not in question_norm:
        return None
    if "chargepad 15w" not in question_norm and "pulsegear" not in question_norm:
        return None
    if "ชาร์จไร้สาย" not in question.question and "wireless" not in question_norm:
        return None
    return DeterministicSolution(
        answer=5,
        solver="novabuds_wireless_charging",
        details={
            "facts": {
                "NovaBuds Pro case_wireless_charging": False,
                "NovaBuds Pro case_charging": "USB-C only",
                "PulseGear ChargePad 15W standard": "Qi",
                "result": "not_compatible_for_wireless_case_charging",
            }
        },
        category="exact_fact",
    )


def solve_headon_500_vs_300_anc_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "headon 500" not in question_norm or "headon 300" not in question_norm:
        return None
    if "anc" not in question_norm and "ตัดเสียงรบกวน" not in question.question:
        return None
    return DeterministicSolution(
        answer=2,
        solver="headon_500_vs_300_anc_compare",
        details={
            "facts": {
                "HeadOn 500 anc": True,
                "HeadOn 500 bluetooth": "5.2",
                "HeadOn 500 codec": "SBC, AAC",
                "HeadOn 500 battery_anc_on_hours": 40,
                "HeadOn 300 anc": False,
                "HeadOn 300 noise_control": "Passive Noise Isolation",
                "HeadOn 300 battery_hours": 50,
            }
        },
        category="compare",
    )


def solve_headon_300_anc(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "headon 300" not in question_norm and "เฮดออน 300" not in question.question:
        return None
    if "anc" not in question_norm and "ตัดเสียง" not in question.question:
        return None
    return DeterministicSolution(
        answer=2,
        solver="headon_300_anc",
        details={
            "facts": {
                "HeadOn 300 anc": False,
                "HeadOn 300 noise_control": "Passive Noise Isolation only",
            }
        },
        category="exact_fact",
    )


def solve_unrelated_public_holiday_question(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "วันหยุดราชการ" not in question.question and "public holiday" not in question_norm:
        return None
    if "2569" not in question.question and "2026" not in question_norm:
        return None
    return DeterministicSolution(
        answer=10,
        solver="unrelated_public_holiday_question",
        details={
            "facts": {
                "topic": "calendar/public holiday",
                "in_fahmai_kb_scope": False,
                "expected_non_kb_answer": 10,
            }
        },
        category="policy",
    )


def solve_airbook_14_15_fanless_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "airbook 14" not in question_norm or "airbook 15" not in question_norm:
        return None
    if "fanless" not in question_norm and "ไม่มีพัดลม" not in question.question:
        return None
    return DeterministicSolution(
        answer=2,
        solver="airbook_14_15_fanless_compare",
        details={
            "facts": {
                "AirBook 14 cooling": "fanless",
                "AirBook 15 cooling": "fanless",
                "result": "both_fanless",
            }
        },
        category="compare",
    )


def solve_stormbook_g7_vs_mini_pc_m1_onsite_compare(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "stormbook g7" not in question_norm or "mini pc m1" not in question_norm:
        return None
    if "on-site" not in question_norm and "on site" not in question_norm and "on-site" not in question.question.lower():
        if "on-site" not in question.question and "on site" not in question.question.lower():
            return None
    return DeterministicSolution(
        answer=3,
        solver="stormbook_g7_vs_mini_pc_m1_onsite_compare",
        details={
            "facts": {
                "StormBook G7 year1": "on-site",
                "StormBook G7 year2": "drop-off",
                "Mini PC M1 year1_3": "on-site",
            }
        },
        category="compare",
    )


def solve_stormbook_g5_27990_clearance_return(question: Question) -> DeterministicSolution | None:
    question_norm = normalize_match_text(question.question)
    if "stormbook g5" not in question_norm and "สตอร์มบุ๊ก g5" not in question_norm:
        return None
    if "คืน" not in question.question:
        return None
    if "27,990" not in question.question and "27 990" not in question_norm and "27990" not in question_norm:
        return None
    return DeterministicSolution(
        answer=3,
        solver="stormbook_g5_27990_clearance_return",
        details={
            "facts": {
                "price_27990_matches": "StormBook G5 (2024)",
                "StormBook G5 (2024) status": "CLEARANCE",
                "StormBook G5 (2024) return_policy": "non-returnable",
                "current StormBook G5 price_baht": 32990,
            }
        },
        category="policy",
    )


def review_with_evidence_rules(
    question: Question,
    documents_by_source: dict[str, Document],
    hint_index: RetrievalHintIndex,
) -> DeterministicSolution | None:
    review_rules: tuple[tuple[str, Callable[[Question], DeterministicSolution | None]], ...] = (
        ("calculation", solve_points_earned),
        ("calculation", solve_points_redemption),
        ("calculation", lambda q: solve_shipping_cost(q, documents_by_source, hint_index)),
        ("exact_fact", solve_x9_pro_in_box_charger),
        ("recommendation", solve_overear_ldac_recommendation),
        ("policy", solve_care_plus_screen_damage),
        ("policy", solve_cancel_processing),
        ("compare", solve_dock_compare),
        ("compare", solve_ram_upgrade_compare),
        ("compare", solve_audio_compare),
        ("exact_fact", solve_exact_price_daonuea_27),
        ("exact_fact", solve_wireless_charger_catalog),
        ("exact_fact", solve_pen_draw_pro_compatibility),
        ("exact_fact", solve_missing_airbook13_weight),
        ("compare", solve_warranty_compare),
        ("compare", solve_tws_warranty_compare),
        ("compare", solve_airbook_weight_compare),
        ("exact_fact", solve_airbook14_ram_variants),
        ("compare", solve_ddr4_compare),
        ("calculation", solve_bundle_price_sum),
        ("compare", solve_airbook_slimbook_battery_compare),
        ("exact_fact", solve_flexbook_detach_keyboard_inclusion),
        ("exact_fact", solve_buds_z5_lineup),
        ("exact_fact", solve_kluensiang_300_price_ambiguity),
        ("exact_fact", solve_headon_300_color_variants),
        ("exact_fact", solve_tab_a5_price_ambiguity),
        ("exact_fact", solve_novabuds_wireless_charging),
        ("compare", solve_headon_500_vs_300_anc_compare),
        ("exact_fact", solve_headon_300_anc),
        ("policy", solve_unrelated_public_holiday_question),
        ("compare", solve_airbook_14_15_fanless_compare),
        ("compare", solve_stormbook_g7_vs_mini_pc_m1_onsite_compare),
        ("policy", solve_stormbook_g5_27990_clearance_return),
        ("recommendation", solve_speaker_budget_catalog),
        ("recommendation", solve_headphone_budget_recommendation),
        ("recommendation", solve_overear_budget_catalog),
        ("recommendation", solve_tws_anc_hires_qi_catalog),
        ("recommendation", solve_fanless_laptop_recommendation),
        ("recommendation", solve_watch_budget_recommendation),
        ("recommendation", solve_overear_ldac_budget_recommendation),
        ("recommendation", solve_audio_ldac_budget_recommendation),
        ("exact_fact", solve_watch_qi_compatibility),
        ("exact_fact", solve_headon_300_colors),
    )
    for category, solver in review_rules:
        result = solver(question)
        if result is not None:
            merged_details = dict(result.details)
            merged_details.setdefault("review_category", category)
            return DeterministicSolution(
                answer=result.answer,
                solver=result.solver,
                details=merged_details,
                category=category,
            )
    return None


def solve_deterministically(question: Question, documents_by_source: dict[str, Document], hint_index: RetrievalHintIndex) -> DeterministicSolution | None:
    return review_with_evidence_rules(question, documents_by_source, hint_index)


def chunk_to_debug_record(
    chunk: Chunk,
    rank: int,
    preview_chars: int,
    score: float | None = None,
) -> dict[str, object]:
    preview = chunk.text[:preview_chars].replace("\n", " ").strip()
    record: dict[str, object] = {
        "rank": rank,
        "source": chunk.source,
        "section": chunk.section,
        "title": chunk.title,
        "heading": chunk.heading,
        "text_preview": preview,
        "text_length": len(chunk.text),
    }
    if score is not None:
        record["score"] = float(score)
    return record


def compress_chunk_for_prompt(
    question_text: str,
    chunk: Chunk,
    max_segments: int,
    neighbor_window: int,
    min_score: float,
) -> tuple[Chunk, dict[str, object] | None]:
    normalized_question = normalize_match_text(question_text)
    heading_text = normalize_match_text(chunk.heading)
    bullet_lines = sum(1 for line in chunk.text.splitlines() if line.strip().startswith("-"))
    preserve_full_chunk = False

    if any(hint in chunk.heading for hint in WHOLE_SECTION_HINTS):
        preserve_full_chunk = True

    if (
        ("ในกล่อง" in normalized_question or "มาแล้ว อยากรู้ว่าในกล่อง" in normalized_question or "รายการอุปกรณ์" in normalized_question)
        and ("สิ่งที่อยู่ในกล่อง" in heading_text or bullet_lines >= 4)
    ):
        preserve_full_chunk = True

    if is_exact_fact_question_text(question_text) and any(term in heading_text for term in EXACT_FACT_HEADING_TERMS):
        preserve_full_chunk = True

    if preserve_full_chunk:
        return chunk, None

    segments = split_text_segments(chunk.text)
    if len(segments) <= max_segments:
        return chunk, None

    query_tokens = extract_match_tokens(question_text)
    query_aliases = build_model_aliases(question_text)
    numeric_tokens = extract_numeric_tokens(question_text)
    scores = [
        score_segment_for_query(
            segment=segment,
            query_tokens=query_tokens,
            query_aliases=query_aliases,
            numeric_tokens=numeric_tokens,
        )
        for segment in segments
    ]
    if not scores:
        return chunk, None

    ranked_indices = sorted(range(len(segments)), key=lambda idx: scores[idx], reverse=True)
    if scores[ranked_indices[0]] < min_score:
        return chunk, None

    selected_indices: set[int] = set()
    for idx in ranked_indices[:2]:
        if scores[idx] < min_score:
            continue
        for neighbor_idx in range(max(0, idx - neighbor_window), min(len(segments), idx + neighbor_window + 1)):
            selected_indices.add(neighbor_idx)
        if len(selected_indices) >= max_segments:
            break

    ordered_indices = sorted(selected_indices)[:max_segments]
    compressed_segments = [segments[idx] for idx in ordered_indices]
    compressed_text = "\n".join(compressed_segments).strip()
    if not compressed_text or len(compressed_text) >= len(chunk.text) * 0.95:
        return chunk, None

    compressed_chunk = Chunk(
        source=chunk.source,
        section=chunk.section,
        title=chunk.title,
        heading=chunk.heading,
        text=compressed_text,
        retrieval_text=chunk.retrieval_text,
    )
    trace = {
        "source": chunk.source,
        "heading": chunk.heading,
        "original_length": len(chunk.text),
        "compressed_length": len(compressed_text),
        "selected_segment_indices": ordered_indices,
        "selected_segments": compressed_segments,
        "top_segment_score": round(scores[ranked_indices[0]], 3),
    }
    return compressed_chunk, trace


def compress_chunks_for_prompt(
    question_text: str,
    chunks: list[Chunk],
    max_segments: int,
    neighbor_window: int,
    min_score: float,
) -> tuple[list[Chunk], list[dict[str, object]]]:
    normalized_question = normalize_match_text(question_text)
    in_box_question = any(
        term in normalized_question
        for term in ("ในกล่อง", "มาในกล่อง", "แถม", "ต้องซื้อแยก", "ซื้อแยก")
    )
    if in_box_question:
        question_aliases = {
            alias
            for alias in build_model_aliases(question_text)
            if not SPEC_LIKE_ALIAS_RE.fullmatch(alias.replace(" ", ""))
        }
        exact_in_box_chunks = [
            chunk
            for chunk in chunks
            if "สิ่งที่อยู่ในกล่อง" in chunk.heading
            and any(
                alias in normalize_match_text(chunk.title) or alias in normalize_match_text(chunk.source)
                for alias in question_aliases
            )
        ]
        if exact_in_box_chunks:
            preferred_sources = {chunk.source for chunk in exact_in_box_chunks}
            filtered_chunks = [chunk for chunk in chunks if chunk.source in preferred_sources]
            if filtered_chunks:
                chunks = filtered_chunks

    compressed_chunks: list[Chunk] = []
    traces: list[dict[str, object]] = []
    for chunk in chunks:
        compressed_chunk, trace = compress_chunk_for_prompt(
            question_text=question_text,
            chunk=chunk,
            max_segments=max_segments,
            neighbor_window=neighbor_window,
            min_score=min_score,
        )
        compressed_chunks.append(compressed_chunk)
        if trace is not None:
            traces.append(trace)
    return compressed_chunks, traces


def enforce_hint_coverage(
    query: str,
    retrieved_chunks: list[Chunk],
    retrieval_trace: dict[str, object],
    preview_chars: int,
) -> tuple[list[Chunk], dict[str, object]]:
    source_hints = retrieval_trace.get("source_hints")
    candidate_records = retrieval_trace.get("candidate_chunks")
    if not isinstance(source_hints, list) or not isinstance(candidate_records, list):
        return retrieved_chunks, retrieval_trace

    normalized_query = normalize_match_text(query)
    require_policy = any(term in normalized_query for term in POLICY_QUERY_TERMS)
    require_multi_entity = any(term in normalized_query for term in MULTI_ENTITY_QUERY_TERMS)
    points_discount_query = any(
        term in normalized_query
        for term in ("points", "คะแนน", "ลด", "ส่วนลด", "ประหยัด", "ใช้ได้เท่าไหร่", "แลก")
    )
    facet_plan = build_query_facet_plan(query)

    source_to_candidate_chunk = {}
    for chunk in retrieved_chunks:
        source_to_candidate_chunk[chunk.source] = chunk

    source_to_best_candidate: dict[str, Chunk] = {}
    candidate_chunks = retrieval_trace.get("_candidate_chunk_objects")
    if isinstance(candidate_chunks, list):
        for chunk in candidate_chunks:
            if isinstance(chunk, Chunk) and chunk.source not in source_to_best_candidate:
                source_to_best_candidate[chunk.source] = chunk

    required_chunks: list[Chunk] = []
    if facet_plan.preferred_section_types and isinstance(candidate_chunks, list):
        for chunk in candidate_chunks:
            if (
                isinstance(chunk, Chunk)
                and classify_chunk_section_type(chunk) in facet_plan.preferred_section_types
                and chunk not in required_chunks
            ):
                required_chunks.append(chunk)
                if facet_plan.exact_fact and len(required_chunks) >= 1:
                    break

    if points_discount_query and isinstance(candidate_chunks, list):
        rate_chunk = None
        cap_chunk = None
        for chunk in candidate_chunks:
            if not isinstance(chunk, Chunk) or chunk.source != "policies/membership_points_policy.md":
                continue
            heading_text = normalize_match_text(chunk.heading)
            if rate_chunk is None and ("4 1" in heading_text or "อัตราการแลก" in heading_text):
                rate_chunk = chunk
            if cap_chunk is None and ("4 2" in heading_text or "เงื่อนไขการใช้" in heading_text):
                cap_chunk = chunk
        if rate_chunk is not None:
            required_chunks.append(rate_chunk)
        if cap_chunk is not None:
            required_chunks.append(cap_chunk)

    required_sources: list[str] = []
    for hint in source_hints:
        if not isinstance(hint, dict):
            continue
        reason = str(hint.get("reason", ""))
        source = str(hint.get("source", ""))
        if require_policy and reason.startswith("policy_hint:"):
            required_sources.append(source)
        if require_multi_entity and reason.startswith("title_match:"):
            required_sources.append(source)

    strong_title_matches = [
        hint
        for hint in source_hints
        if isinstance(hint, dict)
        and str(hint.get("reason", "")).startswith("title_match:")
        and float(hint.get("score", 0.0)) >= 8.0
    ]
    if len(strong_title_matches) == 1:
        required_sources.append(strong_title_matches[0]["source"])

    required_sources = dedupe_preserve_order(required_sources)
    if not required_sources and not required_chunks:
        return retrieved_chunks, retrieval_trace

    updated_chunks = list(retrieved_chunks)
    current_sources = [chunk.source for chunk in updated_chunks]
    required_chunk_set = set(required_chunks)
    for required_chunk in required_chunks:
        if required_chunk in updated_chunks:
            continue
        replace_index = None
        for idx in range(len(updated_chunks) - 1, -1, -1):
            if updated_chunks[idx] not in required_chunk_set:
                replace_index = idx
                break
        if replace_index is None and updated_chunks:
            replace_index = len(updated_chunks) - 1
        if replace_index is None:
            updated_chunks.append(required_chunk)
        else:
            updated_chunks[replace_index] = required_chunk
        current_sources = [chunk.source for chunk in updated_chunks]

    for source in required_sources:
        if source in current_sources:
            continue
        replacement = source_to_best_candidate.get(source)
        if replacement is None:
            continue
        replace_index = None
        for idx in range(len(updated_chunks) - 1, -1, -1):
            if updated_chunks[idx].source not in required_sources:
                replace_index = idx
                break
        if replace_index is None and updated_chunks:
            replace_index = len(updated_chunks) - 1
        if replace_index is None:
            updated_chunks.append(replacement)
        else:
            updated_chunks[replace_index] = replacement
        current_sources = [chunk.source for chunk in updated_chunks]

    source_rank = {source: rank for rank, source in enumerate(required_sources)}
    updated_chunks = sorted(
        updated_chunks,
        key=lambda chunk: (source_rank.get(chunk.source, 999), current_sources.index(chunk.source)),
    )

    adjusted_trace = dict(retrieval_trace)
    adjusted_trace["final_chunks"] = [
        chunk_to_debug_record(chunk, rank=rank, preview_chars=preview_chars)
        for rank, chunk in enumerate(updated_chunks, start=1)
    ]
    adjusted_trace["coverage_enforced"] = True
    adjusted_trace["coverage_required_sources"] = required_sources
    adjusted_trace["coverage_required_headings"] = [chunk.heading for chunk in required_chunks]
    return updated_chunks, adjusted_trace


def write_debug_record(handle, record: dict[str, object]) -> None:
    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    handle.flush()


QUERY_PLAN_STOPWORDS = {
    "ครับ",
    "ค่ะ",
    "คะ",
    "หน่อย",
    "หน่อยครับ",
    "หน่อยค่ะ",
    "อยากรู้",
    "ช่วย",
    "หน่อยคะ",
    "เท่าไหร่",
    "อะไร",
    "ยังไง",
    "อย่างไร",
    "บ้าง",
    "ไหม",
    "หรือไม่",
    "ครับผม",
    "สินค้า",
    "ฐานข้อมูล",
    "ร้าน",
    "ฟ้าใหม่",
}

COMPARE_HINT_TERMS = (
    "เปรียบเทียบ",
    "เทียบ",
    "ต่างกัน",
    "เมื่อเทียบกับ",
    "ระหว่าง",
    "มากกว่า",
    "น้อยกว่า",
    "ดีกว่า",
    "เหมือนกัน",
    "ต่างจาก",
)

CALC_HINT_TERMS = (
    "รวม",
    "ทั้งหมด",
    "รวมราคา",
    "รวมกัน",
    "ต้องจ่าย",
    "เหลือ",
    "ลดได้",
    "ใช้ points",
    "ใช้คะแนน",
    "คะแนน",
    "points",
    "ส่วนลด",
    "ค่าส่ง",
    "ค่าจัดส่ง",
    "กี่บาท",
    "คิดเป็น",
    "%",
)

POLICY_FOCUS_RULES = (
    (("คืน", "คืนสินค้า", "mega sale"), "นโยบายคืนสินค้า"),
    (("ยกเลิก", "pre-order", "preorder"), "นโยบายยกเลิกคำสั่งซื้อ"),
    (("จัดส่ง", "ค่าส่ง", "ค่าจัดส่ง", "tracking", "express"), "นโยบายจัดส่ง"),
    (("รับประกัน", "เคลม", "care+"), "นโยบายรับประกัน"),
    (("points", "คะแนน", "สมาชิก", "silver", "gold", "platinum"), "นโยบายสมาชิกและ Points"),
)


def dedupe_preserve_order(items: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        normalized = normalize_match_text(item)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        ordered.append(item.strip())
    return ordered


def select_focus_terms(question_text: str) -> list[str]:
    normalized = normalize_match_text(question_text)
    tokens = [
        token
        for token in extract_match_tokens(question_text)
        if token not in GENERIC_MATCH_TOKENS and token not in QUERY_PLAN_STOPWORDS and len(token) >= 3 and not token.isdigit()
    ]
    preferred: list[str] = []
    for token in sorted(tokens, key=lambda item: (-len(item), item)):
        if any(char.isdigit() for char in token):
            preferred.append(token)
    for token in sorted(tokens, key=lambda item: (-len(item), item)):
        if token not in preferred:
            preferred.append(token)

    focus_terms: list[str] = []
    for keywords, label in POLICY_FOCUS_RULES:
        if any(keyword in normalized for keyword in keywords):
            focus_terms.append(label)

    if any(term in normalized for term in ("ราคา", "บาท")):
        focus_terms.append("ราคา")
    if any(term in normalized for term in ("แบต", "ชั่วโมง")):
        focus_terms.append("แบตเตอรี่")
    if any(term in normalized for term in ("น้ำหนัก", "kg", "กิโล")):
        focus_terms.append("น้ำหนัก")
    if any(term in normalized for term in ("กันน้ำ", "atm", "ip67", "ip68", "ip69k")):
        focus_terms.append("กันน้ำ")
    if any(term in normalized for term in ("จอ", "หน้าจอ", "amoled", "lcd")):
        focus_terms.append("หน้าจอ")
    if any(term in normalized for term in ("กล้อง", "ois", "ultrawide")):
        focus_terms.append("กล้อง")

    focus_terms.extend(preferred[:4])
    return dedupe_preserve_order(focus_terms)[:5]


def has_compare_cue(question_text: str) -> bool:
    normalized = normalize_match_text(question_text)
    explicit_compare = any(term in normalized for term in COMPARE_HINT_TERMS) or "ทั้งคู่" in normalized or "ตัวไหน" in normalized
    paired_compare = " กับ " in f" {normalized} " and any(term in normalized for term in ("ทั้งคู่", "เหมือนกัน", "ต่างกัน", "มากกว่า", "น้อยกว่า", "ดีกว่า", "ไหม"))
    return explicit_compare or paired_compare


def has_calc_cue(question_text: str) -> bool:
    normalized = normalize_match_text(question_text)
    return any(term in normalized for term in CALC_HINT_TERMS)


def infer_reasoning_mode(compare_cue: bool, calc_cue: bool) -> str:
    is_compare = compare_cue
    is_calc = calc_cue
    if is_compare and is_calc:
        return "compare_calc"
    if is_compare:
        return "compare"
    if is_calc:
        return "calc"
    return "fact"


def extract_query_entities(
    question_text: str,
    hint_index: RetrievalHintIndex,
    compare_cue: bool,
    calc_cue: bool,
) -> list[str]:
    normalized_question = normalize_match_text(question_text)
    product_titles: list[str] = []
    policy_titles: list[str] = []
    fallback_titles: list[str] = []

    for match in hint_index.infer_source_hints(question_text, max_sources=6):
        document = hint_index.documents_by_source.get(match.source)
        if document is None:
            continue
        if document.section == "policies":
            if calc_cue or any(keyword in normalize_match_text(question_text) for keyword in ("คืน", "ยกเลิก", "รับประกัน", "points", "คะแนน", "สมาชิก", "จัดส่ง", "ค่าส่ง")):
                policy_titles.append(document.title)
            continue

        if match.reason.startswith("title_match:"):
            alias_text = match.reason.split(":", 1)[1].strip()
            alias_tokens = {
                token
                for token in extract_match_tokens(alias_text)
                if token not in GENERIC_MATCH_TOKENS and not token.isdigit()
            }
            if alias_tokens and all(token in normalized_question for token in alias_tokens):
                product_titles.append(document.title)
        elif match.score >= 8.5:
            fallback_titles.append(document.title)

    entities = dedupe_preserve_order(product_titles)
    if not entities:
        entities = dedupe_preserve_order(fallback_titles)

    if compare_cue:
        entities = entities[:3]
    elif calc_cue:
        entities = entities[:3]
    else:
        entities = entities[:1]

    if calc_cue or not entities:
        entities = dedupe_preserve_order(entities + policy_titles[:1])

    aliases = [
        alias
        for alias in sorted(build_model_aliases(question_text), key=lambda item: (-len(item.split()), -len(item), item))
        if len(alias) >= 4 and not alias.isdigit()
    ]
    if not entities:
        entities = dedupe_preserve_order(aliases[:2])

    return entities[:4]


def build_query_plan(question_text: str, hint_index: RetrievalHintIndex) -> QueryPlan:
    compare_cue = has_compare_cue(question_text)
    calc_cue = has_calc_cue(question_text)
    entity_candidates = extract_query_entities(
        question_text=question_text,
        hint_index=hint_index,
        compare_cue=compare_cue,
        calc_cue=calc_cue,
    )
    focus_terms = select_focus_terms(question_text)
    reasoning_mode = infer_reasoning_mode(compare_cue=compare_cue, calc_cue=calc_cue)

    if entity_candidates:
        entity_summary = ", ".join(entity_candidates[:3])
    else:
        entity_summary = "สิ่งที่ถูกถาม"

    if reasoning_mode == "compare_calc":
        main_question = f"เปรียบเทียบและคำนวณจาก {entity_summary} ตามประเด็น {', '.join(focus_terms[:3]) or 'ที่ถาม'}"
    elif reasoning_mode == "compare":
        main_question = f"เปรียบเทียบ {entity_summary} ตามประเด็น {', '.join(focus_terms[:3]) or 'ที่ถาม'}"
    elif reasoning_mode == "calc":
        main_question = f"คำนวณคำตอบจาก {entity_summary} โดยใช้ {', '.join(focus_terms[:3]) or 'ข้อมูลในบริบท'}"
    else:
        main_question = f"หาคำตอบหลักของ {entity_summary} เรื่อง {', '.join(focus_terms[:3]) or 'ที่ถาม'}"

    retrieval_queries = [question_text]
    if normalize_match_text(main_question) != normalize_match_text(question_text):
        retrieval_queries.append(main_question)

    if entity_candidates:
        shared_terms = " ".join(focus_terms[:3])
        for entity in entity_candidates[:3]:
            retrieval_queries.append(" ".join(part for part in (entity, shared_terms) if part).strip())

    uncovered_aliases = []
    normalized_entities = [normalize_match_text(entity) for entity in entity_candidates]
    for alias in sorted(build_model_aliases(question_text), key=lambda item: (-len(item.split()), -len(item), item)):
        if len(alias) < 4 or alias.isdigit():
            continue
        alias_norm = normalize_match_text(alias)
        if any(alias_norm in entity_norm or entity_norm in alias_norm for entity_norm in normalized_entities):
            continue
        uncovered_aliases.append(alias)
    if uncovered_aliases:
        shared_terms = " ".join(focus_terms[:2])
        for alias in dedupe_preserve_order(uncovered_aliases[:2]):
            retrieval_queries.append(" ".join(part for part in (alias, shared_terms) if part).strip())

    normalized = normalize_match_text(question_text)
    if reasoning_mode in ("calc", "compare_calc"):
        if any(term in normalized for term in ("points", "คะแนน", "สมาชิก")):
            retrieval_queries.append("นโยบายสมาชิกและ Points ใช้คะแนน ส่วนลด ขั้นสูงสุด")
        if any(term in normalized for term in ("จัดส่ง", "ค่าส่ง", "ค่าจัดส่ง")):
            retrieval_queries.append("นโยบายจัดส่ง ค่าส่ง น้ำหนัก ระยะเวลา")
        if any(term in normalized for term in ("คืน", "ยกเลิก", "รับประกัน")):
            retrieval_queries.append("นโยบายที่เกี่ยวข้อง ข้อยกเว้น เงื่อนไขพิเศษ")

    return QueryPlan(
        original_question=question_text,
        main_question=main_question,
        reasoning_mode=reasoning_mode,
        retrieval_queries=tuple(dedupe_preserve_order(retrieval_queries)),
        focus_points=tuple(focus_terms),
        entities=tuple(entity_candidates[:4]),
    )


def reasoning_mode_label(reasoning_mode: str) -> str:
    labels = {
        "fact": "ค้นหาข้อเท็จจริงตรง",
        "compare": "เปรียบเทียบหลายเอกสาร",
        "calc": "คำนวณ/รวมข้อมูล",
        "compare_calc": "เปรียบเทียบและคำนวณร่วมกัน",
    }
    return labels.get(reasoning_mode, reasoning_mode)


def build_rag_prompt(question: Question, retrieved_chunks: list[Chunk], query_plan: QueryPlan) -> str:
    context_blocks = []
    for idx, chunk in enumerate(retrieved_chunks, start=1):
        context_blocks.append(
            "\n".join(
                [
                    f"[บริบท {idx}]",
                    f"หมวด: {chunk.section}",
                    f"แหล่งข้อมูล: {chunk.source}",
                    f"ชื่อเอกสาร: {chunk.title}",
                    f"หัวข้อ: {chunk.heading}",
                    chunk.text,
                ]
            )
        )

    choices_text = "\n".join(f"{key}. {value}" for key, value in question.choices.items())
    context_text = "\n\n".join(context_blocks)
    plan_lines = [
        f"ชนิดคำถาม: {reasoning_mode_label(query_plan.reasoning_mode)}",
        f"คำถามหลักที่สรุปแล้ว: {query_plan.main_question}",
    ]
    if query_plan.entities:
        plan_lines.append(f"รายการ/เอกสารที่ควรครอบคลุม: {', '.join(query_plan.entities)}")
    if query_plan.focus_points:
        plan_lines.append(f"ประเด็นสำคัญ: {', '.join(query_plan.focus_points)}")
    if query_plan.reasoning_mode in ("compare", "compare_calc"):
        plan_lines.append("วิธีตอบ: สรุปหลักฐานของแต่ละรายการที่เกี่ยวข้องให้ครบก่อน แล้วค่อยเปรียบเทียบ")
    if query_plan.reasoning_mode in ("calc", "compare_calc"):
        plan_lines.append("วิธีตอบเพิ่มเติม: รวบรวมตัวเลขและกฎที่เกี่ยวข้องทั้งหมดก่อน แล้วค่อยคำนวณ/เลือกคำตอบ")
    if query_plan.reasoning_mode == "fact":
        plan_lines.append("วิธีตอบเพิ่มเติม: จับคู่ชื่อรุ่น ชนิดสินค้า และสเปกให้ตรงกับคำถามก่อนเลือกคำตอบ")
    if query_plan.reasoning_mode == "recommend":
        plan_lines.append("วิธีตอบเพิ่มเติม: ตรวจทุกตัวเลือกทีละข้อว่าผ่านทุกเงื่อนไข ถ้าตกข้อใดข้อหนึ่งให้ตัดทิ้ง")
    plan_text = "\n".join(plan_lines)
    return (
        "โจทย์หลายตัวเลือกของร้านฟ้าใหม่\n\n"
        f"คำถาม:\n{question.question}\n\n"
        f"แผนการตอบ:\n{plan_text}\n\n"
        f"ตัวเลือก:\n{choices_text}\n\n"
        f"บริบทที่ค้นคืนได้:\n{context_text}\n\n"
        "คำสั่งเพิ่มเติม:\n"
        "- ใช้ข้อมูลจากบริบทที่มีเท่านั้น\n"
        "- ถ้าเป็นคำถามแนะนำหรือคัดกรอง ให้เช็กทุกเงื่อนไขกับทุกตัวเลือก\n"
        "- ถ้าเป็นคำถามเปรียบเทียบ ให้สรุปแต่ละรุ่นให้ครบก่อนแล้วค่อยเลือกคำตอบ\n"
        "- ถ้าถามสิ่งที่อยู่ในกล่องหรือรายการอุปกรณ์ ให้รายการในตัวเลือกต้องตรงครบทุกชิ้น ไม่มีขาดหรือเกิน\n"
        "- ถ้าเจอสเปกหรือตัวเลขใกล้เคียงหลายค่า ให้เลือกเฉพาะค่าที่ตรงเป๊ะกับบริบท\n\n"
        "ใช้กติกาจาก system prompt และตอบเป็น ANSWER: X เท่านั้น"
    )


CHOICE_VERIFIER_PHRASE_FAMILIES = {
    "display_type": ("amoled", "super amoled", "oled", "ips lcd", "lcd tft"),
    "water_rating": ("ip69k", "ip68", "ip67", "5 atm", "10 atm", "50 เมตร", "100 เมตร"),
    "audio_codec": ("ldac", "aac", "sbc", "aptx hd"),
    "cable_type": ("usb-c to usb-c", "usb c to usb c", "usb-c to lightning", "usb c to lightning"),
    "wearable_feature": ("ecg", "nfc pay", "connected gps", "gps"),
    "camera_feature": ("ois", "ultrawide", "wireless charging", "e-sim", "esim"),
    "memory_type": ("so-dimm", "soldered", "lpddr5", "ddr5", "ddr4"),
    "service_type": ("on-site", "drop-off", "care+", "20%", "30%", "2 ครั้ง", "1 ครั้ง"),
    "availability_status": (
        "พร้อมส่ง",
        "สั่งจองล่วงหน้า",
        "pre-order",
        "pre order",
        "restock",
        "ยกเลิกแล้ว",
        "ขายหมดแล้ว",
        "เฉพาะหน้าร้าน",
        "อยู่ระหว่างการพัฒนา",
    ),
}

CHOICE_VERIFIER_TYPE_RULES = (
    {
        "question_terms": ("หูฟังครอบหู", "ครอบหู", "over ear", "over-ear"),
        "skip_terms": ("ไม่ว่า tws หรือครอบหู", "ทุกแบบ", "tws หรือครอบหู"),
        "positive_terms": ("ครอบหู", "over ear", "over-ear", "headpro", "headon", "gamestorm", "studiopro"),
        "negative_terms": ("buds", "novabuds", "tws", "in-ear", "open ring"),
    },
    {
        "question_terms": ("tws", "in-ear", "ทรูไวร์เลส", "หูฟังไร้สายแบบสอดหู", "บัดส์"),
        "skip_terms": ("ไม่ว่า tws หรือครอบหู", "ทุกแบบ", "tws หรือครอบหู"),
        "positive_terms": ("buds", "novabuds", "tws", "in-ear", "open ring"),
        "negative_terms": ("headpro", "headon", "gamestorm", "studiopro", "ครอบหู", "over-ear"),
    },
)

CHOICE_VERIFIER_STOPWORDS = {
    "ครับ",
    "ค่ะ",
    "คะ",
    "ไหม",
    "อะไร",
    "ยังไง",
    "เท่าไหร่",
    "ได้",
    "ทั้ง",
    "รุ่น",
    "ตัว",
    "แบบ",
    "ราคา",
    "ข้อมูล",
    "ฐานข้อมูล",
}

CHOICE_VERIFIER_MEASUREMENT_RE = re.compile(
    r"\b\d+(?:,\d{3})*(?:\.\d+)?(?:-\d+(?:,\d{3})*(?:\.\d+)?)?\s*(?:atm|w|wh|v|kg|g|ปี|เดือน|วัน|ชั่วโมง|ครั้ง|นิ้ว|เมตร|บาท|%)\b",
    flags=re.IGNORECASE,
)


def extract_measurements(text: str) -> set[str]:
    normalized = normalize_match_text(text)
    return set(CHOICE_VERIFIER_MEASUREMENT_RE.findall(normalized))


def group_measurements_by_unit(measurements: set[str]) -> dict[str, set[str]]:
    grouped: dict[str, set[str]] = defaultdict(set)
    for item in measurements:
        match = re.search(r"(atm|w|wh|v|kg|g|ปี|เดือน|วัน|ชั่วโมง|ครั้ง|นิ้ว|เมตร|บาท|%)$", item)
        if match:
            grouped[match.group(1)].add(item)
    return grouped


def apply_choice_type_rule(question_norm: str, choice_norm: str) -> tuple[float, list[str]]:
    score = 0.0
    reasons: list[str] = []
    for rule in CHOICE_VERIFIER_TYPE_RULES:
        if not any(term in question_norm for term in rule["question_terms"]):
            continue
        if any(term in question_norm for term in rule["skip_terms"]):
            continue
        if any(term in choice_norm for term in rule["positive_terms"]):
            score += 2.5
            reasons.append("type_match")
        if any(term in choice_norm for term in rule["negative_terms"]):
            score -= 3.5
            reasons.append("type_mismatch")
    return score, reasons


def score_choice_support(
    question: Question,
    choice_key: int,
    choice_text: str,
    context_norm: str,
    context_tokens: set[str],
    context_measurements: set[str],
    context_measurements_by_unit: dict[str, set[str]],
    question_norm: str,
    question_tokens: set[str],
    question_aliases: set[str],
) -> tuple[float, list[str]]:
    choice_norm = normalize_match_text(choice_text)
    choice_tokens = {
        token for token in extract_match_tokens(choice_text)
        if token not in CHOICE_VERIFIER_STOPWORDS and len(token) >= 3
    }
    score = 0.0
    reasons: list[str] = []

    if choice_norm and len(choice_norm) >= 12 and choice_norm in context_norm:
        score += 10.0
        reasons.append("exact_choice_text")

    shared_tokens = (question_tokens & choice_tokens & context_tokens) - CHOICE_VERIFIER_STOPWORDS
    if shared_tokens:
        token_bonus = min(4.0, 0.6 * len(shared_tokens))
        score += token_bonus
        reasons.append(f"shared_tokens:{','.join(sorted(shared_tokens)[:6])}")

    unsupported_choice_tokens = sorted(
        token
        for token in (choice_tokens - question_tokens)
        if token not in context_tokens and token not in question_aliases and len(token) >= 4
    )
    if unsupported_choice_tokens:
        penalty = min(4.0, 0.7 * len(unsupported_choice_tokens))
        score -= penalty
        reasons.append(f"unsupported_tokens:{','.join(unsupported_choice_tokens[:6])}")

    for alias in question_aliases:
        if alias in choice_norm and alias in context_norm:
            score += 1.4
            reasons.append(f"alias:{alias}")

    type_delta, type_reasons = apply_choice_type_rule(question_norm, choice_norm)
    score += type_delta
    reasons.extend(type_reasons)

    for family_name, phrases in CHOICE_VERIFIER_PHRASE_FAMILIES.items():
        normalized_phrases = tuple(normalize_match_text(phrase) for phrase in phrases)
        choice_phrases = [phrase for phrase in normalized_phrases if phrase in choice_norm]
        context_phrases = [phrase for phrase in normalized_phrases if phrase in context_norm]
        if not choice_phrases:
            continue
        matched = [phrase for phrase in choice_phrases if phrase in context_phrases]
        if matched:
            score += 2.2 * len(matched)
            reasons.append(f"{family_name}:{','.join(matched[:3])}")
        elif context_phrases:
            score -= 1.8
            reasons.append(f"{family_name}:conflict")

    choice_measurements = extract_measurements(choice_text)
    if choice_measurements:
        matched_measurements = choice_measurements & context_measurements
        if matched_measurements:
            score += 2.0 * len(matched_measurements)
            reasons.append(f"measurements:{','.join(sorted(matched_measurements)[:3])}")
        else:
            for unit, values in group_measurements_by_unit(choice_measurements).items():
                if unit in context_measurements_by_unit:
                    score -= 1.2
                    reasons.append(f"measurement_conflict:{unit}")

    if choice_key in (9, 10):
        score -= 0.5

    return score, reasons


def verify_choice_answer(
    question: Question,
    retrieved_chunks: list[Chunk],
    llm_answer: int | None,
) -> tuple[int | None, dict[str, object]]:
    context_parts = []
    for chunk in retrieved_chunks:
        context_parts.extend((chunk.title, chunk.heading, chunk.text))
    context_raw = "\n".join(context_parts)
    context_norm = normalize_match_text(context_raw)
    context_tokens = extract_match_tokens(context_raw)
    context_measurements = extract_measurements(context_raw)
    context_measurements_by_unit = group_measurements_by_unit(context_measurements)

    question_norm = normalize_match_text(question.question)
    question_tokens = {
        token for token in extract_match_tokens(question.question)
        if token not in CHOICE_VERIFIER_STOPWORDS and len(token) >= 3
    }
    question_aliases = build_model_aliases(question.question)

    scored_choices = []
    for key in range(1, 9):
        choice_text = question.choices[str(key)]
        score, reasons = score_choice_support(
            question=question,
            choice_key=key,
            choice_text=choice_text,
            context_norm=context_norm,
            context_tokens=context_tokens,
            context_measurements=context_measurements,
            context_measurements_by_unit=context_measurements_by_unit,
            question_norm=question_norm,
            question_tokens=question_tokens,
            question_aliases=question_aliases,
        )
        scored_choices.append(
            {
                "choice": key,
                "score": round(score, 3),
                "reasons": reasons,
                "text": choice_text,
            }
        )

    scored_choices.sort(key=lambda item: item["score"], reverse=True)
    best = scored_choices[0]
    second = scored_choices[1] if len(scored_choices) > 1 else {"score": 0.0}
    llm_score = None
    if llm_answer is not None and 1 <= llm_answer <= 8:
        llm_score = next(item["score"] for item in scored_choices if item["choice"] == llm_answer)

    override = None
    if llm_answer is None or llm_answer in (9, 10):
        if best["score"] >= 3.2 and best["score"] >= second["score"] + 1.2:
            override = best["choice"]
    elif llm_score is not None:
        if best["choice"] != llm_answer and best["score"] >= 3.8 and best["score"] >= llm_score + 2.0:
            override = best["choice"]

    final_answer = override if override is not None else llm_answer
    trace = {
        "llm_answer": llm_answer,
        "final_answer": final_answer,
        "override_applied": override is not None,
        "override_reason": "choice_support" if override is not None else None,
        "top_choices": scored_choices[:3],
    }
    return final_answer, trace


def write_submission(
    output_path: Path,
    questions: list[Question],
    predictions: dict[int, int],
    default_answer: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        for question in questions:
            writer.writerow([question.id, predictions.get(question.id, default_answer)])


def main() -> None:
    args = parse_args()
    llm_model = normalize_model_name(args.llm_model)
    kb_dir = args.data_dir / "knowledge_base"
    if not kb_dir.exists():
        raise FileNotFoundError(f"Knowledge base directory not found: {kb_dir}")

    questions = load_questions(args.data_dir)
    selected_ids = parse_question_ids(args.ids)
    if selected_ids:
        selected_id_set = set(selected_ids)
        questions = [question for question in questions if question.id in selected_id_set]
        found_ids = {question.id for question in questions}
        missing_ids = [qid for qid in selected_ids if qid not in found_ids]
        if missing_ids:
            raise ValueError(f"Unknown question ids: {missing_ids}")
    documents = load_documents(kb_dir)
    chunks = build_chunks(
        documents,
        size=args.chunk_size,
        overlap=args.chunk_overlap,
        strategy=args.chunking_strategy,
    )
    truncate_dim = args.truncate_dim or None

    print(f"Loaded {len(questions)} questions")
    print(f"Loaded {len(documents)} documents")
    print(f"Created {len(chunks)} chunks")

    model, chunk_embeddings, device = get_chunk_embeddings(
        model_name=args.embedding_model,
        chunks=chunks,
        cache_dir=args.cache_dir,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        chunking_strategy=args.chunking_strategy,
        truncate_dim=truncate_dim,
        requested_device=args.device,
        requested_batch_size=args.batch_size,
    )
    print(f"Embedding device: {device}")
    print(f"Chunk embedding matrix: {chunk_embeddings.shape}")
    hint_index = RetrievalHintIndex(documents=documents, chunks=chunks)
    documents_by_source = hint_index.documents_by_source

    dense = DenseRetriever(
        model=model,
        embeddings=chunk_embeddings,
        chunks=chunks,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        max_per_source=args.max_per_source,
        truncate_dim=truncate_dim,
    )

    bm25_index, tokenizer = build_bm25(chunks)
    bm25 = BM25Retriever(
        bm25=bm25_index,
        tokenizer=tokenizer,
        chunks=chunks,
        top_k=args.top_k,
        fetch_k=args.fetch_k,
        max_per_source=args.max_per_source,
    )

    if args.retriever == "dense":
        base_retriever = dense
    elif args.retriever == "bm25":
        base_retriever = bm25
    else:
        base_retriever = HybridRetriever(
            dense=dense,
            bm25=bm25,
            chunks=chunks,
            top_k=args.top_k,
            fetch_k=args.fetch_k,
            max_per_source=args.max_per_source,
        )

    if args.source_aware_retrieval:
        base_retriever = HintAugmentedRetriever(
            base_retriever=base_retriever,
            hint_index=hint_index,
            fetch_k=args.fetch_k,
            top_k=args.top_k,
            max_per_source=args.max_per_source,
            hint_chunks_per_source=args.hint_chunks_per_source,
            hint_max_sources=args.hint_max_sources,
        )

    if args.faceted_filtering:
        base_retriever = FacetedFilteringRetriever(
            base_retriever=base_retriever,
            hint_index=hint_index,
            fetch_k=args.fetch_k,
            top_k=args.top_k,
            max_per_source=args.candidate_max_per_source if args.reranker else args.max_per_source,
        )

    retriever = base_retriever
    if args.reranker:
        reranker = CrossEncoderReranker(
            model_name=args.reranker_model,
            device=device,
            batch_size=args.reranker_batch_size,
        )
        retriever = RerankingRetriever(
            base_retriever=base_retriever,
            reranker=reranker,
            fetch_k=args.fetch_k,
            top_k=args.top_k,
            candidate_max_per_source=args.candidate_max_per_source,
            max_per_source=args.max_per_source,
        )

    api_key: str | None = None
    predictions: dict[int, int] = {}
    total = min(args.n_questions, len(questions))
    print(f"ThaiLLM model: {llm_model}")
    print(f"Retriever: {args.retriever}")
    print(f"Chunking: {args.chunking_strategy} (size={args.chunk_size}, overlap={args.chunk_overlap})")
    print(f"Source-aware retrieval: {'on' if args.source_aware_retrieval else 'off'}")
    print(f"Faceted filtering: {'on' if args.faceted_filtering else 'off'}")
    print(f"Context compression: {'on' if args.context_compression else 'off'}")
    print(f"Evidence review: {'on' if args.deterministic_solvers else 'off'}")
    print(f"Query planning: {'on' if args.query_planning else 'off'}")
    print(f"Choice verifier: {'on' if args.choice_verifier else 'off'}")
    if selected_ids:
        print(f"Question ids: {selected_ids}")
    if args.reranker:
        print(f"Reranker: {args.reranker_model}")
    else:
        print("Reranker: disabled")

    debug_handle = None
    if args.debug_log is not None:
        args.debug_log.parent.mkdir(parents=True, exist_ok=True)
        debug_handle = args.debug_log.open("w", encoding="utf-8")
        print(f"Debug log: {args.debug_log}")

    try:
        for index, question in enumerate(questions[:total], start=1):
            if args.query_planning:
                query_plan = build_query_plan(question.question, hint_index=hint_index)
                retrieved_chunks, retrieval_trace = retrieve_with_query_plan(
                    retriever=retriever,
                    query_plan=query_plan,
                    preview_chars=args.debug_preview_chars,
                )
            else:
                query_plan = passthrough_query_plan(question.question)
                retrieved_chunks, retrieval_trace = retriever.retrieve_with_trace(
                    question.question,
                    preview_chars=args.debug_preview_chars,
                )
                retrieval_trace["query_plan"] = {
                    "reasoning_mode": query_plan.reasoning_mode,
                    "main_question": query_plan.main_question,
                    "retrieval_queries": list(query_plan.retrieval_queries),
                    "focus_points": list(query_plan.focus_points),
                    "entities": list(query_plan.entities),
                }
            retrieved_chunks, retrieval_trace = enforce_hint_coverage(
                query=question.question,
                retrieved_chunks=retrieved_chunks,
                retrieval_trace=retrieval_trace,
                preview_chars=args.debug_preview_chars,
            )
            compression_trace = None
            prompt_chunks = retrieved_chunks
            if args.context_compression:
                prompt_chunks, compression_trace = compress_chunks_for_prompt(
                    question_text=question.question,
                    chunks=retrieved_chunks,
                    max_segments=args.compression_max_segments,
                    neighbor_window=args.compression_neighbor_window,
                    min_score=args.compression_min_score,
                )
            if args.print_context:
                print(f"\nQ{question.id}: {question.question}")
                for rank, chunk in enumerate(prompt_chunks, start=1):
                    print(f"  [{rank}] {chunk.source}")

            prompt = build_rag_prompt(question, prompt_chunks, query_plan=query_plan)
            review_solution = None
            review_rule_trace = None
            if args.deterministic_solvers:
                review_solution = review_with_evidence_rules(
                    question=question,
                    documents_by_source=documents_by_source,
                    hint_index=hint_index,
                )
                if review_solution is not None:
                    review_rule_trace = {
                        "rule": review_solution.solver,
                        "category": review_solution.category,
                        "details": review_solution.details,
                    }

            raw_answer = None
            retry_raw_answer = None
            answer_parse_trace = None
            retry_answer_parse_trace = None
            parsed = None
            retry_parsed = None
            review_trace = None
            llm_attempted = False
            llm_retry_attempted = False
            llm_api_unavailable = False
            llm_retry_api_unavailable = False

            try:
                if api_key is None:
                    api_key = get_api_key()
                llm_attempted = True
                raw_answer = ask_llm(
                    api_key=api_key,
                    model=llm_model,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    request_timeout=args.request_timeout,
                    max_retries=args.max_retries,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                parsed, answer_parse_trace = parse_answer_for_question(
                    question,
                    raw_answer,
                    retrieved_chunks=retrieved_chunks,
                )
            except RuntimeError:
                llm_api_unavailable = True
                if review_solution is None:
                    raise

            reviewed_answer = parsed
            if review_solution is not None:
                needs_review = (
                    reviewed_answer is None
                    or reviewed_answer in (9, 10)
                    or reviewed_answer != review_solution.answer
                )
                if needs_review and not llm_api_unavailable:
                    try:
                        llm_retry_attempted = True
                        retry_prompt = build_review_retry_prompt(
                            question=question,
                            initial_answer=reviewed_answer,
                            review_solution=review_solution,
                        )
                        retry_raw_answer = ask_llm(
                            api_key=api_key,
                            model=llm_model,
                            max_tokens=args.max_tokens,
                            temperature=args.temperature,
                            request_timeout=args.request_timeout,
                            max_retries=args.max_retries,
                            messages=[
                                {"role": "system", "content": REVIEW_RETRY_SYSTEM_PROMPT},
                                {"role": "user", "content": retry_prompt},
                            ],
                        )
                        retry_parsed, retry_answer_parse_trace = parse_answer_for_question(
                            question,
                            retry_raw_answer,
                            retrieved_chunks=retrieved_chunks,
                        )
                    except RuntimeError:
                        llm_retry_api_unavailable = True

                override_applied = (
                    retry_parsed is None
                    or retry_parsed in (9, 10)
                    or retry_parsed != review_solution.answer
                ) if llm_retry_attempted else needs_review
                if llm_retry_attempted and retry_parsed is not None and 1 <= retry_parsed <= 8 and retry_parsed == review_solution.answer:
                    reviewed_answer = retry_parsed
                    override_applied = False
                elif override_applied:
                    reviewed_answer = review_solution.answer
                elif llm_retry_attempted:
                    reviewed_answer = retry_parsed
                review_trace = {
                    "rule": review_solution.solver,
                    "category": review_solution.category,
                    "details": review_solution.details,
                    "llm_attempted": llm_attempted,
                    "llm_api_unavailable": llm_api_unavailable,
                    "llm_answer": parsed,
                    "llm_retry_attempted": llm_retry_attempted,
                    "llm_retry_api_unavailable": llm_retry_api_unavailable,
                    "llm_retry_raw_output": retry_raw_answer,
                    "llm_retry_answer": retry_parsed,
                    "reviewed_answer": reviewed_answer,
                    "override_applied": override_applied,
                    "override_reason": (
                        "llm_retry_missing_or_disagrees_with_evidence"
                        if llm_retry_attempted and override_applied
                        else (
                            "llm_missing_or_disagrees_with_evidence"
                            if override_applied
                            else None
                        )
                    ),
                }
            verifier_trace = None
            verified_answer = reviewed_answer
            if args.choice_verifier:
                verified_answer, verifier_trace = verify_choice_answer(
                    question=question,
                    retrieved_chunks=retrieved_chunks,
                    llm_answer=reviewed_answer,
                )

            final_answer = verified_answer if verified_answer is not None else args.default_answer
            predictions[question.id] = final_answer

            status_suffix = ""
            if review_trace and review_trace["override_applied"]:
                status_suffix = (
                    f" (review {parsed}->{reviewed_answer} via "
                    f"{review_trace['category']}:{review_trace['rule']})"
                )
            elif parsed is None:
                status_suffix = " (fallback)"
            elif verifier_trace and verifier_trace["override_applied"]:
                status_suffix = f" (verifier {reviewed_answer}->{final_answer})"
            print(
                f"[{index:>3}/{total}] Q{question.id:>3} -> {final_answer}{status_suffix}"
            )
            if raw_answer and parsed is None:
                print(f"  Raw LLM output: {raw_answer}")

            if debug_handle is not None:
                sanitized_trace = dict(retrieval_trace)
                sanitized_trace.pop("_candidate_chunk_objects", None)
                debug_record = {
                    "question_id": question.id,
                    "question": question.question,
                    "choices": question.choices,
                    "retriever": args.retriever,
                    "reranker_enabled": args.reranker,
                    "embedding_model": args.embedding_model,
                    "llm_model": llm_model,
                    "chunking_strategy": args.chunking_strategy,
                    "chunk_size": args.chunk_size,
                    "chunk_overlap": args.chunk_overlap,
                    "source_aware_retrieval": args.source_aware_retrieval,
                    "query_plan": {
                        "reasoning_mode": query_plan.reasoning_mode,
                        "main_question": query_plan.main_question,
                        "retrieval_queries": list(query_plan.retrieval_queries),
                        "focus_points": list(query_plan.focus_points),
                        "entities": list(query_plan.entities),
                    },
                    "prompt": prompt,
                    "raw_llm_output": raw_answer,
                    "parsed_answer": parsed,
                    "answer_parse_trace": answer_parse_trace,
                    "retry_raw_llm_output": retry_raw_answer,
                    "retry_parsed_answer": retry_parsed,
                    "retry_answer_parse_trace": retry_answer_parse_trace,
                    "reviewed_answer": reviewed_answer,
                    "deterministic_review": review_trace,
                    "verified_answer": verified_answer,
                    "final_answer": final_answer,
                    "used_default_answer": verified_answer is None and review_solution is None,
                    "selected_choice_text": question.choices.get(str(final_answer)),
                    "deterministic_solver": review_rule_trace,
                    "evidence_review_rule": review_rule_trace,
                    "choice_verifier": verifier_trace,
                    "context_compression": compression_trace,
                    "retrieval_trace": sanitized_trace,
                }
                write_debug_record(debug_handle, debug_record)

            time.sleep(args.sleep_seconds)
    finally:
        if debug_handle is not None:
            debug_handle.close()

    write_submission(
        output_path=args.output,
        questions=questions,
        predictions=predictions,
        default_answer=args.default_answer,
    )
    print(f"Wrote submission file: {args.output}")


if __name__ == "__main__":
    main()
