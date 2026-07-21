"""
qa.py
-----
The "generation" half of RAG. Takes a user question plus the top retrieved
chunks and asks Claude to answer strictly using that context, citing the
source file and page number for every claim. This grounding step is what
prevents hallucinated answers -- the model can only speak to what was
actually retrieved from the student's own notes.
"""

import os
from typing import List

import anthropic

from src.retriever import RetrievedChunk

SYSTEM_PROMPT = """You are a study assistant. You must answer the student's question
using ONLY the provided lecture note excerpts below. Follow these rules strictly:

1. If the excerpts don't contain enough information to answer, say so clearly --
   do not guess or use outside knowledge.
2. After every claim, cite the source in this format: (Source: <file>, page <n>).
3. Keep answers concise and exam-focused.
4. If excerpts conflict, point out the conflict rather than picking one silently.
"""


def format_context(results: List[RetrievedChunk]) -> str:
    blocks = []
    for r in results:
        blocks.append(
            f"[Source: {r.chunk.source_file}, page {r.chunk.page_number}]\n{r.chunk.text}"
        )
    return "\n\n---\n\n".join(blocks)


def answer_question(question: str, results: List[RetrievedChunk], model: str = "claude-sonnet-4-6") -> str:
    if not results:
        return "I couldn't find anything relevant to that question in your uploaded notes."

    client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    context = format_context(results)

    user_prompt = f"""Lecture note excerpts:

{context}

Student's question: {question}
"""

    response = client.messages.create(
        model=model,
        max_tokens=600,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_prompt}],
    )

    return response.content[0].text
