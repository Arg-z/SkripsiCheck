from pathlib import Path

import pymupdf
import pytest
from docx import Document

from app.core.extractor import DocumentExtractionError, extract_text


def test_pdf_extraction(tmp_path: Path) -> None:
    path = tmp_path / "sample.pdf"
    document = pymupdf.open()
    page = document.new_page()
    page.insert_text((72, 72), "Penelitian telur puyuh")
    document.save(path)
    document.close()

    assert "Penelitian telur puyuh" in extract_text(path)


def test_docx_extraction(tmp_path: Path) -> None:
    path = tmp_path / "sample.docx"
    document = Document()
    document.add_paragraph("Metode penelitian kuantitatif")
    document.save(path)

    assert extract_text(path) == "Metode penelitian kuantitatif"


def test_txt_extraction(tmp_path: Path) -> None:
    path = tmp_path / "sample.txt"
    path.write_text("Teks Bahasa Indonesia", encoding="utf-8")
    assert extract_text(path) == "Teks Bahasa Indonesia"


def test_rejects_extension_spoofing(tmp_path: Path) -> None:
    path = tmp_path / "fake.pdf"
    path.write_text("not actually a PDF", encoding="utf-8")
    with pytest.raises(DocumentExtractionError, match="not PDF"):
        extract_text(path)
