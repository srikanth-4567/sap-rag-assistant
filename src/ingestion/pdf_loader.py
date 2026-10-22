"""Utilities for extracting text from PDF files."""

from pathlib import Path
from typing import List, Union
from pypdf import PdfReader



def load_pdf(pdf_path: str) -> list[dict]:
    """Load a PDF as one LangChain Document per page.

    Each document contains the page text and source/page metadata. Raises
    ``FileNotFoundError`` if the requested file does not exist.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF file not found: {path}")

    reader = PdfReader(str(path))
    documents = []
    for page_number, page in enumerate(
        reader.pages,
        start=1
        ):
        text = page.extract_text() or ""
        text = text.strip()

        #if not text:
        #    continue

        documents.append(
            {
                "text": text,
                "metadata": {
                    "source": path.name, 
                    "page": page_number,
                    },
            }
        )

    return documents