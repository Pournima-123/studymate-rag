"""
qa.py
-----
The "generation" half of RAG. Takes a user question plus the top retrieved
chunks and asks an LLM to answer strictly using that context, citing the
source file and page number for every claim.

Supports two providers, controlled by the LLM_PROVIDER env var:
  - "ollama" (default): runs a free, local open-source model via Ollama.
  - "anthropic": uses the Claude API (requires ANTHROPIC_API_KEY).
"""

import os
from typing import List

import requests

from src.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a study assistant. You must answer the student's question
using ONLY the provided lecture note excerpts below. Follow these rules strictly:

1. If the excerpts don't contain enough information to answer, say so clearly --
   do not guess or use outside knowledge.
2. After every claim, cite the source in this format: (Source: <file>, page <n>).
3. Keep answers concise and exam-focused.
4. If excerpts conflict, point out the conflict rather than picking one silently.
"""

OLLAMA_URL = "http://localhost:11434/api/chat"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.1")


def format_context(results: List[RetrievedChunk]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[Source: {r.chunk.source_file}, page {r.chunk.page_number}]\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def _build_user_prompt(question: str, results: List[RetrievedChunk]) -> str:
    context = format_context(results)
    return f"""Lecture note excerpts:

{context}

Student's question: {question}
"""


def _answer_with_ollama(question: str, results: List[RetrievedChunk]) -> str:
    user_prompt = _build_user_prompt(question, results)

    try:
        response = requests.post(
            OLLAMA_URL,
            json={
                "model": OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
    except requests.exceptions.ConnectionError:
        return (
            "Could not reach Ollama at localhost:11434. Make sure Ollama is installed "
            "and running, and that you've pulled a model with `ollama pull llama3.1`."
        )

    return response.json()["message"]["content"]


def _answer_with_anthropic(question: str, results: List[RetrievedChunk], model: str = "claude-sonnet-4-6") -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    user_prompt = _build_user_prompt(question, results)

    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )
    return response.content[0].text


def answer_question(question: str, results: List[RetrievedChunk]) -> str:
    if not results:
        return "I couldn't find anything relevant to that question in your uploaded notes."

    provider = os.environ.get("LLM_PROVIDER", "ollama").lower()

    if provider == "anthropic":
        return _answer_with_anthropic(question, results)
    return _answer_with_ollama(question, results)