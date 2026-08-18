# Vercel protected-pilot deployment

> **Status: experimental and verification required.** Adapter Vercel tersedia di
> codebase, tetapi deployment belum boleh disebut siap publik hanya karena build
> berhasil atau halaman utama terbuka. Gunakan dokumen sintetis sampai seluruh
> checklist smoke test di bawah lulus pada deployment yang sebenarnya.

Panduan ini ditujukan untuk pilot kecil bersama teman yang dipercaya. Ini bukan
arsitektur multi-user production: access control masih berupa satu shared token,
session browser bersifat anonim, dan rate limiting, kuota, akun pengguna, serta
retention otomatis belum tersedia.

## Arsitektur

```text
Browser
  -> /api/blob-upload (short-lived upload token)
  -> Vercel Private Blob (dokumen)
  -> FastAPI Function (extract, embed, retrieve, score)
       -> Private Blob (sources.faiss + metadata.json + index_info.json)
       -> Neon PostgreSQL (metadata dokumen + report)
       -> /tmp (salinan kerja sementara pada satu warm instance)
```

Model `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` diunduh
saat Vercel build dari revision yang dipin di
`scripts/prepare_vercel_model.py`, lalu dimasukkan ke bundle Function. Model
tidak di-commit dan tidak diunduh ulang pada setiap analisis. Inference tetap
berjalan di Function dengan CPU; tidak ada dokumen yang dikirim ke API model
eksternal.

## Perbedaan privasi: lokal dan hosted

Pada mode lokal, dokumen, database, embedding, dan index berada di komputer
operator. Pada mode Vercel:

- file mahasiswa berada di **Private** Vercel Blob;
- teks sumber dan metadata chunk terdapat di `metadata.json` pada Private Blob;
- metadata dokumen dan report berada di Neon PostgreSQL; report JSON mencakup
  paragraf mahasiswa serta teks sumber yang cocok;
- Function Vercel mengunduh dokumen/index sementara untuk analisis; dan
- operator akun Vercel/Neon tetap bertanggung jawab atas akses serta penghapusan.

`Private` berarti Blob tidak mempunyai URL publik bebas akses. Itu tidak berarti
data tetap berada di laptop pengguna. Mintalah persetujuan pengguna dan jelaskan
hosting serta masa simpan sebelum menerima dokumen nyata. Jangan gunakan corpus
yang lisensinya tidak mengizinkan penyimpanan tersebut.

## Resource yang diperlukan

1. Repository GitHub SkripsiCheck.
2. Vercel project dengan region Function `sin1` (Singapore, sudah dikonfigurasi
   di `vercel.json`).
3. Satu Vercel Blob store dengan access **Private**, terhubung ke project.
4. Satu database Neon PostgreSQL, sebaiknya memakai pooled connection dan region
   terdekat yang tersedia.
5. Satu shared access token acak untuk pilot.
6. Tiga artifact FAISS dari build index yang sama.

Nama resource bebas. Nama yang mudah dikenali misalnya `skripsicheck`,
`skripsicheck-private`, dan `skripsicheck-db`. Nama tidak memengaruhi nilai
environment variable.

## 1. Verifikasi source index secara lokal

Gunakan model default yang sama dengan deployment:

```powershell
.\.venv\Scripts\skripsicheck.exe rebuild-index .\sample_documents\references
.\.venv\Scripts\skripsicheck.exe search "Kalsium membantu memperkuat cangkang telur" --top-k 5
```

Pastikan tiga file ini dibuat bersama:

```text
data/index/sources.faiss
data/index/metadata.json
data/index/index_info.json
```

Jangan mencampur file dari dua build. Loader memvalidasi checksum, build ID,
model, dimensi, serta jumlah vector/metadata dan akan menolak bundle yang tidak
sinkron.

## 2. Siapkan Private Blob

Hubungkan Blob store ber-access **Private** ke Vercel project. Integrasi Vercel
akan menyediakan `BLOB_READ_WRITE_TOKEN`; jangan menyalin nilainya ke source,
browser JavaScript, screenshot, issue, atau chat.

Unggah tiga artifact index menggunakan Blob dashboard atau tooling Vercel yang
dipercaya, dengan pathname persis berikut:

```text
indexes/current/sources.faiss
indexes/current/metadata.json
indexes/current/index_info.json
```

Vercel CLI mendukung `--pathname`. Setelah CLI terhubung ke project dan
credential Blob tersedia secara aman di luar repository, contoh initial upload:

```powershell
vercel blob put .\data\index\sources.faiss --pathname indexes/current/sources.faiss --access private
vercel blob put .\data\index\metadata.json --pathname indexes/current/metadata.json --access private
vercel blob put .\data\index\index_info.json --pathname indexes/current/index_info.json --access private
vercel blob list --prefix indexes/current/
```

Jangan menaruh token pada argumen command yang akan tersimpan di shell history.
Ikuti [panduan resmi Blob CLI](https://vercel.com/docs/cli/blob) untuk autentikasi
dan cek bahwa ketiga pathname terlihat pada store **Private** yang benar.

Repository saat ini membaca dan memvalidasi artifact tersebut, tetapi belum
menyediakan command untuk memublikasikannya otomatis. Perlakukan
`metadata.json` sebagai data privat karena berisi teks chunk sumber.

Untuk mengganti index dengan aman, upload ketiga file dari satu build ke prefix
baru, misalnya `indexes/2026-08-18-a`, lalu ubah
`SKRIPSICHECK_BLOB_INDEX_PREFIX` dan redeploy. Jangan menimpa satu file di
`indexes/current` sementara deployment sedang menerima analisis.

## 3. Hubungkan Neon PostgreSQL

Buat atau hubungkan Neon melalui Vercel Marketplace. Pilih pooled PostgreSQL URL
bila tersedia. Code memilih URL database dalam urutan berikut:

1. `SKRIPSICHECK_DATABASE_URL`
2. `DATABASE_URL`
3. `POSTGRES_URL`
4. fallback lokal `sqlite:///data/skripsicheck.sqlite3`

Integrasi biasanya menginjeksi `DATABASE_URL` atau `POSTGRES_URL`, sehingga
`SKRIPSICHECK_DATABASE_URL` tidak perlu dibuat. Pastikan tidak ada nilai SQLite
yang tertinggal di environment Vercel. Jangan pernah menaruh connection string
nyata di `.env.example` atau Git.

Schema dibuat saat aplikasi mulai. Untuk protected pilot gunakan database baru
dan kosong; proyek belum memiliki migration framework production untuk perubahan
schema jangka panjang.

## 4. Buat shared access token

Gunakan secret unik minimal 32 karakter, idealnya dari password manager. Contoh
membuat 32 byte acak di PowerShell:

```powershell
$secretBytes = New-Object byte[] 32
[Security.Cryptography.RandomNumberGenerator]::Fill($secretBytes)
[Convert]::ToBase64String($secretBytes)
```

Simpan hasilnya sebagai `SKRIPSICHECK_ACCESS_TOKEN` di Vercel Project Settings.
Jangan gunakan `BLOB_READ_WRITE_TOKEN` sebagai access token. Bagikan access
token hanya kepada peserta pilot melalui kanal privat dan rotasi bila bocor.

Shared token hanya invitation gate. UUID session browser membatasi dokumen/report
ke session yang membuatnya, tetapi UUID tersebut bukan identitas atau akun. UI
menyimpan token dan UUID di `sessionStorage`, bukan di URL; menutup/mereset sesi
browser dapat membuat report lama tidak dapat diakses dari UI tersebut.

## 5. Environment variables Vercel

Nilai minimum untuk hosted pilot:

| Variable | Nilai/contoh | Catatan |
|---|---|---|
| `SKRIPSICHECK_STORAGE_BACKEND` | `vercel_blob` | Wajib; jangan memakai filesystem persisten. |
| `SKRIPSICHECK_INDEX_BACKEND` | `vercel_blob` | Mengambil artifact FAISS dari Private Blob. |
| `SKRIPSICHECK_BLOB_DOCUMENT_PREFIX` | `documents` | Harus cocok dengan token handler upload. |
| `SKRIPSICHECK_BLOB_INDEX_PREFIX` | `indexes/current` | Harus cocok dengan pathname artifact. |
| `SKRIPSICHECK_ACCESS_TOKEN` | secret acak 32+ karakter | Wajib untuk Blob mode. |
| `BLOB_READ_WRITE_TOKEN` | diinjeksi integrasi | Secret Blob; jangan diberikan ke pengguna. |
| `DATABASE_URL` atau `POSTGRES_URL` | diinjeksi Neon | Gunakan pooled PostgreSQL URL. |
| `SKRIPSICHECK_DEVICE` | `cpu` | GPU tidak diperlukan/diandalkan. |
| `SKRIPSICHECK_MAX_UPLOAD_MB` | `25` | Batas aplikasi dan browser upload. |

Opsional:

| Variable | Default | Kegunaan |
|---|---:|---|
| `SKRIPSICHECK_INDEX_CACHE_DIR` | temp directory sistem | Cache artifact FAISS pada warm instance. |
| `SKRIPSICHECK_EMBEDDING_BATCH_SIZE` | `32` | Turunkan bila memory menjadi masalah. |
| `SKRIPSICHECK_MAX_ANALYSIS_CHARACTERS` | `1500000` | Menolak dokumen hasil ekstraksi yang terlalu besar sebelum semantic search. |
| `SKRIPSICHECK_MAX_ANALYSIS_PARAGRAPHS` | `2000` | Guard jumlah paragraf; turunkan untuk pilot dengan resource ketat. |
| `SKRIPSICHECK_TOP_K_MATCHES` | `5` | Jumlah match akhir default. |
| `SKRIPSICHECK_MIN_SEMANTIC_SCORE` | `0.40` | Filter candidate semantic. |
| `SKRIPSICHECK_MIN_MATCH_SCORE` | `0.40` | Filter combined score. |

Biarkan `SKRIPSICHECK_SEMANTIC_MODEL_PATH` tidak diset pada Vercel. Aplikasi
akan menemukan `deployment/model` yang disiapkan build. Jika variable tersebut
dipakai, nilainya harus menunjuk directory model yang benar di dalam bundle.

Terapkan variable pada Preview terlebih dahulu. Production baru boleh diaktifkan
setelah checklist lulus. Jangan menampilkan nilai secret pada log atau screenshot.

## 6. Import dan build di Vercel

Import repository GitHub sebagai Vercel project. Repository sudah menyediakan:

- `.python-version` dengan Python 3.12;
- FastAPI entrypoint `app.main:app` di `pyproject.toml`;
- region `sin1`, Fluid Compute, dan `maxDuration` 300 detik di `vercel.json`;
- build hook untuk model multilingual dengan revision yang dipin; dan
- Node handler `/api/blob-upload` untuk direct browser upload.

Model, PyTorch, Sentence Transformers, dan FAISS membuat bundle besar. Pastikan
project mendukung ukuran Function hasil build (aktifkan Large Functions bila
opsi tersebut tersedia) dan periksa ukuran bundle pada build output. Build yang
melewati limit bukan kondisi siap deploy.

Gunakan **Preview deployment yang terlindungi** lebih dahulu. Aktifkan Vercel
Deployment Protection bila tersedia untuk plan Anda, selain shared access token
aplikasi. Jangan langsung membagikan Production URL. Lihat
[Deployment Protection](https://vercel.com/docs/deployment-protection) untuk opsi
yang tersedia pada akun Anda.

## 7. Batas runtime yang perlu diuji

- Vercel Function mempunyai batas request body; upload browser langsung ke Blob
  digunakan agar dokumen sampai 25 MB tidak melewati body FastAPI.
- `/tmp` bersifat sementara dan terbatas. Index diunduh sekali per warm instance,
  sedangkan dokumen hanya dimaterialisasi selama analisis.
- `maxDuration: 300` adalah konfigurasi aplikasi, tetapi batas efektif tetap
  bergantung plan/platform.
- CPU inference dapat mengalami cold start dan timeout pada dokumen panjang atau
  ketika banyak pengguna menganalisis bersamaan.
- Tidak ada GPU, job queue, rate limit, kuota per pengguna, atau retry worker.
- Private Blob masih dapat berstatus beta sesuai dokumentasi Vercel; cek kembali
  availability, limit, serta ketentuannya sebelum menerima dokumen nyata.

Limit Vercel dapat berubah. Verifikasi kembali dokumentasi resmi sebelum deploy:
[FastAPI](https://vercel.com/docs/frameworks/backend/fastapi),
[Function limits](https://vercel.com/docs/functions/limitations),
[Python runtime](https://vercel.com/docs/functions/runtimes/python),
[Private Blob](https://vercel.com/docs/vercel-blob/private-storage), dan
[client upload](https://vercel.com/docs/vercel-blob/client-upload).

## 8. Smoke test Preview

Gunakan hanya sample sintetis. Catat URL Preview, commit SHA, dan tanggal uji.

- [ ] Build selesai dan ukuran Function berada di bawah limit project.
- [ ] `GET /health` mengembalikan status `ok`.
- [ ] `GET /api/runtime` melaporkan `access_required: true`,
      `direct_upload: true`, dan batas upload yang benar.
- [ ] Request API tanpa shared token ditolak `401`.
- [ ] Shared token benar diterima tanpa muncul di URL/log.
- [ ] PDF, DOCX, dan TXT kecil dapat direct-upload ke Private Blob.
- [ ] Path dokumen berbentuk `documents/<session-uuid>/<document-uuid>.<ext>`.
- [ ] Analisis menemukan index Blob dan menghasilkan semantic match.
- [ ] Report tetap dapat diambil setelah refresh dan setelah redeploy (Neon
      benar-benar persisten).
- [ ] Session browser lain tidak dapat membaca dokumen/report milik session awal.
- [ ] Tombol delete menghapus record dan Blob dokumen terkait.
- [ ] Cold-start test berhasil mengunduh serta memvalidasi ketiga artifact index.
- [ ] Log Vercel tidak memuat access token, Blob token, database URL, atau teks
      dokumen.
- [ ] Pengujian timeout dilakukan dengan dokumen mendekati ukuran/kompleksitas
      pilot yang disepakati.

Jika satu item gagal, tetap gunakan mode lokal dan jangan mengundang pengguna.

## 9. Operasional pilot

- Batasi peserta dan jangan pasang URL pada forum publik.
- Beritahu peserta bahwa hasil adalah similarity/needs review, bukan vonis
  plagiarisme dan bukan skor Turnitin.
- Tetapkan masa simpan tertulis. Karena belum ada TTL otomatis, operator harus
  melakukan audit berkala terhadap Blob dan database.
- Hapus dokumen setelah tidak diperlukan dan verifikasi penghapusan di Blob.
- Pantau error, durasi, memory, storage, serta biaya/kuota Vercel dan Neon.
- Rotasi shared access token setelah pilot atau segera setelah dicurigai bocor.
- Simpan backup hanya bila ada persetujuan dan kebijakan yang jelas.

## Readiness record

Isi bagian ini pada PR/release note, bukan dengan secret:

```text
Commit tested:
Preview URL:
Test date:
Tester:
Blob private + connected: PASS/FAIL
Neon persistence: PASS/FAIL
FAISS cold-start load: PASS/FAIL
PDF/DOCX/TXT direct upload: PASS/FAIL
Delete verification: PASS/FAIL
Security/log review: PASS/FAIL
Timeout/load sample: PASS/FAIL
Decision: LOCAL ONLY / PROTECTED PILOT / STOP
```

Status `PROTECTED PILOT` tetap bukan persetujuan untuk public launch.
