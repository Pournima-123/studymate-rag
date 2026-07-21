"""
ingest.py
---------
Handles the "data collection, cleaning, and preprocessing" pipeline:
1. Extracts raw text from PDF lecture notes / slides.
2. Cleans and normalizes the text.
3. Splits text into overlapping chunks (with page number metadata)
   so the retriever can later cite exactly where an answer came from.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List
import re

from pypdf import PdfReader


@dataclass
class Chunk:
    chunk_id: int
    source_file: str
    page_number: int
    text: str


def clean_text(raw_text: str) -> str:
    """Basic text cleaning: collapse whitespace, strip weird PDF artifacts."""
    text = re.sub(r"\s+", " ", raw_text)
    text = text.strip()
    return text


def extract_pages(pdf_path: Path) -> List[str]:
    """Extract raw text from each page of a PDF."""
    reader = PdfReader(str(pdf_path))
    pages = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        pages.append(clean_text(raw))
    return pages


def chunk_text(text: str, chunk_size: int = 700, overlap: int = 120) -> List[str]:
    """
    Split a page's text into overlapping word-based chunks.
    Overlap preserves context that would otherwise be cut at chunk boundaries.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk_words = words[start:end]
        chunks.append(" ".join(chunk_words))
        if end >= len(words):
            break
        start = end - overlap
    return chunks


def build_chunks_from_pdf(pdf_path: Path) -> List[Chunk]:
    """
    Full ingestion pipeline for a single PDF:
    extract -> clean -> chunk -> attach page-level metadata.
    """
    pages = extract_pages(pdf_path)
    all_chunks: List[Chunk] = []
    chunk_id = 0

    for page_number, page_text in enumerate(pages, start=1):
        for piece in chunk_text(page_text):
            all_chunks.append(
                Chunk(
                    chunk_id=chunk_id,
                    source_file=pdf_path.name,
                    page_number=page_number,
                    text=piece,
                )
            )
            chunk_id += 1

    return all_chunks


def build_chunks_from_folder(folder: Path) -> List[Chunk]:
    """Run ingestion across every PDF in a folder (e.g. all lecture slides)."""
    all_chunks: List[Chunk] = []
    for pdf_file in sorted(folder.glob("*.pdf")):
        all_chunks.extend(build_chunks_from_pdf(pdf_file))
    return all_chunks
