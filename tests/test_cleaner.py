from app.core.cleaner import clean_text


def test_cleaning_page_numbers_whitespace_and_characters() -> None:
    raw = "JUDUL LAPORAN\n1\nIsi   halaman satu.\x00\fJUDUL LAPORAN\nPage 2\nIsi halaman dua."
    cleaned = clean_text(raw)
    assert "JUDUL LAPORAN" not in cleaned
    assert "Page 2" not in cleaned
    assert "\x00" not in cleaned
    assert "Isi halaman satu." in cleaned


def test_cleaning_preserves_paragraph_breaks() -> None:
    assert clean_text("Paragraf satu.\n\n\n\nParagraf dua.") == (
        "Paragraf satu.\n\nParagraf dua."
    )

