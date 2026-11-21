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


def test_roman_like_words_are_not_removed_as_page_numbers() -> None:
    raw = "did\ndim\ncivil\nlid\nKonten utama.\nXII\nKonten kedua."
    cleaned = clean_text(raw)
    lines = cleaned.split("\n")
    for word in ("did", "dim", "civil", "lid", "Konten utama.", "Konten kedua."):
        assert word in lines
    assert "XII" not in lines


def test_strict_roman_page_numbers_are_still_removed() -> None:
    raw = "BAB I\nii\nIsi bab.\nIV\nIsi lanjutan."
    cleaned = clean_text(raw)
    lines = cleaned.split("\n")
    assert "ii" not in lines
    assert "IV" not in lines
    assert "BAB I" in lines and "Isi bab." in lines
