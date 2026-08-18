from app.core.chunker import chunk_document, split_paragraphs, split_sentences


def test_paragraph_chunking() -> None:
    text = "Paragraf pertama berisi data.\n\nParagraf kedua berisi analisis."
    assert split_paragraphs(text) == [
        "Paragraf pertama berisi data.",
        "Paragraf kedua berisi analisis.",
    ]


def test_sentence_chunking() -> None:
    assert split_sentences("Data dikumpulkan. Hasil dianalisis! Apakah valid?") == [
        "Data dikumpulkan.",
        "Hasil dianalisis!",
        "Apakah valid?",
    ]


def test_document_chunks_keep_paragraph_context() -> None:
    chunks = chunk_document("Kalimat satu. Kalimat dua.\n\nParagraf lain.")
    assert len(chunks) == 2
    assert len(chunks[0][1]) == 2

