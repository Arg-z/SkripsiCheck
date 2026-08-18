# SkripsiCheck

SkripsiCheck adalah proyek open-source untuk membantu mahasiswa meninjau
kemiripan teks terhadap koleksi sumber milik sendiri. Seluruh ekstraksi,
embedding, indexing, dan pencarian dilakukan lokal tanpa API berbayar dan tanpa
mengunggah dokumen pengguna ke layanan eksternal.

> **Status:** pengembangan v0.1.0 — PHASE 1, PHASE 2, dan PHASE 3 selesai.
> FastAPI, SQLite, upload web, dan report merupakan PHASE 4–6 dan belum tersedia.

## Screenshot

_Screenshot placeholder — antarmuka web akan ditambahkan setelah PHASE 5._

## Features

- ekstraksi teks PDF, DOCX, dan TXT;
- validasi tipe/signature dan batas ukuran dokumen;
- pembersihan nomor halaman, whitespace, karakter kontrol, serta header/footer
  sederhana yang berulang;
- pemisahan paragraf dan kalimat;
- exact normalized match, TF-IDF cosine similarity, dan word-shingle overlap;
- multilingual semantic similarity untuk Bahasa Indonesia dan Inggris;
- batch embedding dengan model yang dimuat lazy dan dipakai ulang;
- source indexing lokal menggunakan normalized embeddings dan FAISS `IndexFlatIP`;
- persistence index, metadata chunk, index info, serta fingerprint SHA-256 sumber;
- candidate retrieval sebelum lexical/ngram scoring sehingga seluruh corpus tidak
  dibandingkan secara lexical;
- combined score transparan dan deduplikasi hasil sumber.

## Installation

Gunakan Python 3.11 atau lebih baru. Untuk PHASE 1–3:

```bash
python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1

python -m pip install -r requirements-phase3.txt
python -m pip install -e . --no-deps
pytest -m "not slow"
```

`requirements.txt` tetap memuat stack yang direncanakan sampai v0.1.0. Model ML
tidak disimpan di repository. Model akan diunduh otomatis ke cache lokal pada
pemakaian semantic pertama.

## Semantic Similarity

Model default dikonfigurasi satu kali di `app/config.py`:

```text
sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
```

Model ini dipilih karena multilingual, mendukung Bahasa Indonesia dan Inggris,
relatif ringan untuk CPU, gratis, dan tersedia secara open-source. Service
memuat model secara lazy, memakai satu instance kembali, mendukung batch
encoding, dan menghasilkan embedding `float32` ternormalisasi.

Semantic similarity dapat menemukan parafrase yang menggunakan kata berbeda,
tetapi hasil tersebut bukan bukti otomatis plagiarisme. Setiap passage tetap
perlu ditinjau secara manual.

## Source Indexing

Folder sumber boleh memiliki subfolder dan berisi PDF, DOCX, atau TXT. Untuk
membuat index:

```bash
skripsicheck index ./sample_documents/references
```

Untuk membangun ulang seluruh index:

```bash
skripsicheck rebuild-index ./sample_documents/references
```

Pada MVP, `index` dan `rebuild-index` sama-sama melakukan full rebuild. Struktur
metadata dan fingerprint telah dipisahkan agar incremental indexing dapat
ditambahkan kemudian.

Progress ditampilkan tanpa dependency progress-bar tambahan:

```text
Indexing sources...

[1/2] jurnal_kalsium.txt
[2/2] keamanan_jaringan.txt

Sources indexed: 2
Chunks indexed: 4
Embedding dimension: 384

Index saved:
data/index/sources.faiss
```

Dokumen rusak atau tanpa chunk dilaporkan dan dilewati. Build gagal dengan pesan
informatif jika folder tidak ditemukan, tidak memiliki file yang didukung, atau
seluruh dokumen gagal menghasilkan chunk.

## FAISS Index

Artifact lokal disimpan di lokasi berikut secara default:

```text
data/
└── index/
    ├── sources.faiss
    ├── metadata.json
    └── index_info.json
```

`metadata.json` menyimpan `chunk_id`, nama/path sumber, teks, jumlah kata, page
jika tersedia, serta fingerprint sumber. `index_info.json` menyimpan model,
dimensi embedding, jumlah vector, waktu build, build ID, checksum artifact, dan
interpretasi score. Loader memeriksa keberadaan ketiga file, model, dimensi,
checksum, serta sinkronisasi jumlah vector dengan metadata. Folder `data/`
diabaikan Git.

## How Semantic Search Works

Alur pencarian PHASE 3:

```text
query paragraph
    -> normalized multilingual embedding
    -> FAISS top-k candidate retrieval
    -> TF-IDF + n-gram hanya untuk candidate tersebut
    -> combined score dan kategori review
```

Cari satu paragraph pada index yang sudah dibuat:

```bash
skripsicheck search "Kalsium membantu memperkuat cangkang telur" --top-k 5
```

Secara default, hasil di bawah `SKRIPSICHECK_MIN_SEMANTIC_SCORE` (0.40) tidak
ditampilkan. Nilai ini dapat diganti per pencarian, misalnya
`--min-score 0.25`, tanpa mengubah index.

FAISS memakai inner product pada embedding ternormalisasi, sehingga nilainya
setara cosine similarity dengan rentang native `-1..1`. SkripsiCheck
menginterpretasikan nilai negatif sebagai tidak ada kecocokan bermakna dan
menampilkannya sebagai `0`; score user-facing berada pada rentang `0..1`.

Combined score menggunakan:

```text
0.35 × lexical similarity
+ 0.45 × semantic similarity
+ 0.20 × n-gram overlap
```

Bobot harus non-negatif dan totalnya divalidasi menjadi `1.0`. Kategori review:

- 0–39%: LOW
- 40–59%: MODERATE
- 60–79%: HIGH
- 80–100%: VERY HIGH

Kategori berasal dari combined score, bukan satu metode saja.

## Demo End-to-End

Corpus dan dokumen demo bersifat sintetis. Jalankan:

```bash
python scripts/demo_phase3.py
```

Script mengekstrak sample document, membangun index sumber, mencari candidate
untuk setiap paragraph, lalu menampilkan lexical, semantic, n-gram, combined
score, kategori, dan alasan.

Perintah pemeriksaan dokumen final berikut disiapkan untuk PHASE 6 dan **belum
tersedia** pada PHASE 3:

```bash
skripsicheck check ./sample_documents/example.docx
```

## Configuration

Salin `.env.example` atau set environment variable secara langsung:

```text
SKRIPSICHECK_SEMANTIC_MODEL=sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
SKRIPSICHECK_EMBEDDING_BATCH_SIZE=32
SKRIPSICHECK_DEVICE=cpu
SKRIPSICHECK_MIN_SEMANTIC_SCORE=0.40
SKRIPSICHECK_TOP_K_MATCHES=5
SKRIPSICHECK_INDEX_DIR=data/index
```

CPU adalah default dan GPU/CUDA tidak diwajibkan.

## Testing

Unit test menggunakan fake deterministic encoder agar tidak membutuhkan internet:

```bash
pytest -m "not slow"
```

Test model nyata diberi marker `slow`. Setelah model tersedia di cache lokal:

```powershell
$env:SKRIPSICHECK_RUN_SLOW = "1"
pytest -m slow
```

## Performance Notes

- Model hanya dimuat saat encoding pertama dan tidak dimuat ulang per fungsi.
- Source paragraphs di-encode dalam batch; default batch size adalah 32.
- FAISS mengambil candidate terdekat sebelum perhitungan lexical dan n-gram.
- `IndexFlatIP` melakukan exact vector search dan cocok untuk MVP. Corpus sangat
  besar nantinya dapat memakai approximate index tanpa mengubah scoring layer.
- Index tersimpan dapat dibuka kembali tanpa embedding ulang seluruh sumber.

## Privacy

- Dokumen, teks, embedding, metadata, dan FAISS index tetap lokal.
- SkripsiCheck tidak mengirim dokumen ke API atau layanan analisis eksternal.
- Koneksi internet hanya diperlukan untuk mengunduh paket/model pertama kali.
- Folder data, uploads, reports, indexes, database, model, dan cache diabaikan Git.
- Jangan commit dokumen mahasiswa atau sumber berlisensi tanpa izin.

## Limitations

- Hasil hanya seluas koleksi sumber lokal milik pengguna.
- Istilah teknis, definisi umum, kutipan sah, dan bibliografi dapat menghasilkan
  similarity tinggi.
- Citation metadata belum tersedia sampai PHASE 6.
- PDF hasil scan memerlukan OCR, yang belum termasuk v0.1.0.
- Angka SkripsiCheck **tidak dapat dibandingkan langsung dengan skor Turnitin**
  karena database dan algoritmanya berbeda.

## Roadmap

- v0.2 — Web source search
- v0.3 — Better citation analysis
- v0.4 — Reference list matching
- v0.5 — Batch document analysis
- v1.0 — Stable public release

## Contributing

Issue dan pull request dipersilakan. Sertakan test untuk perubahan perilaku,
jalankan suite offline, dan jangan menyertakan dokumen sensitif atau model ML
dalam commit.

## License

MIT License. Lihat [LICENSE](LICENSE).

## Disclaimer

SkripsiCheck is a text similarity analysis tool, not an automated plagiarism
verdict system. Similarity does not necessarily indicate plagiarism. Results
should be reviewed manually and interpreted according to academic citation
standards.
