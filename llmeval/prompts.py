"""Prompts e JSON Schemas de geração e de julgamento.

As instruções são idênticas entre condições, exceto pela descrição do contexto
(documento completo, trechos recuperados ou nenhum), para que a única variável
manipulada seja o contexto.
"""
from __future__ import annotations

import re

from .data import UNANSWERABLE

LETTERS = "ABCD"

CONTEXT_INTRO = {
    "full": "The user provides the full text of a document.",
    "rag": "The user provides excerpts retrieved from a document. They are only part of the "
    "document and may not contain the answer.",
    "none": "No document is provided.",
}

FREEFORM_SYSTEM = """You answer questions about a scientific paper.
{context_intro}
Rules:
- {grounding}
- Be concise: a short phrase or a single sentence.
- For yes/no questions, start the answer with "Yes" or "No".
- If the information needed is not available, the answer must be exactly "UNANSWERABLE".
Respond in JSON: {{"answer": "..."}}"""

MC_SYSTEM = """You answer multiple-choice questions about a story or article.
{context_intro}
Rules:
- {grounding}
- First state the answer in your own words in one short sentence (do not copy the option text).
- Then select the letter of the best option.
Respond in JSON: {{"answer": "...", "choice": "A|B|C|D"}}"""

GROUNDING = {
    "full": "Use only the information in the document; do not rely on prior knowledge.",
    "rag": "Use only the information in the excerpts; do not rely on prior knowledge.",
    "none": "Answer from your own knowledge.",
}

FREEFORM_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
}
MC_SCHEMA = {
    "type": "object",
    "properties": {"answer": {"type": "string"}, "choice": {"type": "string", "enum": list(LETTERS)}},
    "required": ["answer", "choice"],
}


def generation_messages(question: dict, context_kind: str, context_text: str | None) -> tuple[list, dict]:
    mc = question["dataset"] == "quality"
    system = (MC_SYSTEM if mc else FREEFORM_SYSTEM).format(
        context_intro=CONTEXT_INTRO[context_kind], grounding=GROUNDING[context_kind]
    )
    parts = []
    if context_kind != "none":
        label = "Document" if context_kind == "full" else "Excerpts"
        parts.append(f"{label}:\n<<<\n{context_text}\n>>>")
    parts.append(f"Question: {question['question']}")
    if mc:
        parts.append("Options:\n" + "\n".join(f"{l}. {o}" for l, o in zip(LETTERS, question["options"])))
    messages = [{"role": "system", "content": system}, {"role": "user", "content": "\n\n".join(parts)}]
    return messages, (MC_SCHEMA if mc else FREEFORM_SCHEMA)


_UNANS = re.compile(rf"\b{UNANSWERABLE}\b|^\s*n/?a\s*\.?$", re.IGNORECASE)


def is_abstention(answer: str | None) -> bool:
    return answer is not None and bool(_UNANS.search(answer)) and len(answer.split()) <= 12


# --------------------------------------------------------------------- júri
JUDGE_SYSTEM = """You are an impartial grader. Decide whether a CANDIDATE answer to a QUESTION is correct, using the REFERENCE answers as the only source of truth (they come from the document the question is about; do not use outside knowledge, and do not judge whether the references themselves are true).

Verdicts:
- "correct": the candidate conveys the key information of at least one reference. Paraphrases, synonyms, abbreviations, different number formats and extra details that do not contradict the reference are acceptable.
- "partially_correct": the candidate captures part of the key information of a reference but omits an essential element, or mixes correct information with a claim that conflicts with the references.
- "incorrect": the candidate contradicts the references, answers a different question, is irrelevant, or says the information is unavailable.

Judge meaning, not wording or length. Write a brief justification first, then the verdict.
Respond in JSON: {"justification": "...", "verdict": "correct|partially_correct|incorrect"}"""

JUDGE_SCHEMA = {
    "type": "object",
    "properties": {
        "justification": {"type": "string"},
        "verdict": {"type": "string", "enum": ["correct", "partially_correct", "incorrect"]},
    },
    "required": ["justification", "verdict"],
}
VERDICT_SCORE = {"correct": 1.0, "partially_correct": 0.5, "incorrect": 0.0}


def judge_messages(question: str, references: list[str], candidate: str) -> list[dict]:
    refs = "\n".join(f"- {r}" for r in references)
    user = f"QUESTION: {question}\n\nREFERENCE answers:\n{refs}\n\nCANDIDATE answer: {candidate}"
    return [{"role": "system", "content": JUDGE_SYSTEM}, {"role": "user", "content": user}]
