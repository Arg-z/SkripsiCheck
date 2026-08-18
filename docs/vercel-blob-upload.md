# Private Vercel Blob browser upload

SkripsiCheck uses Vercel Blob client uploads so PDF, DOCX, and TXT files do not
pass through Vercel's 4.5 MB Function request-body limit. The browser requests a
short-lived client token from `/api/blob-upload`, then uploads the file directly
to the **Private** Blob store with multipart upload enabled.

This adapter is part of the experimental protected-pilot deployment. Follow the
complete setup and smoke-test checklist in [deploy-vercel.md](deploy-vercel.md);
do not treat a successful upload alone as production readiness.

## Environment variables

- `BLOB_READ_WRITE_TOKEN` is created by Vercel when the Private Blob store is
  connected to the project. Never put this value in HTML, browser JavaScript, a
  URL, Git, or screenshots.
- `SKRIPSICHECK_ACCESS_TOKEN` is a separate, high-entropy shared access code used
  by both the Python API and this handler. It restricts who may request a
  client-upload token. Use at least 32 characters. Both Functions must see the
  same value inside one environment; prefer a different token for Preview and
  Production.
- `SKRIPSICHECK_MAX_UPLOAD_MB` is an integer upload limit. It defaults to `25`
  and the token handler refuses values outside `1..500`.

The shared access code is only a temporary invitation gate. It is visible to a
person who enters it in their browser and is not a replacement for user accounts,
rate limiting, quotas, or abuse monitoring. Never reuse the Blob read-write token
as the shared access code.

## Build and test

```bash
npm ci
npm run test:node
npm run build:blob-client
```

The build exposes `window.SkripsiCheckBlob.uploadPrivateDocument(...)` from
`static/blob-client.js`. The page loads this bundle before the main application
script. The browser must select this path only when `/api/runtime` reports
`direct_upload: true`; local mode continues to use the FastAPI multipart route.

The server validates a version-4 session UUID and document UUID and permits only
this pathname shape:

```text
documents/<session-uuid>/<document-uuid>.<pdf|docx|txt>
```

It also restricts content type, maximum size, and multipart mode in the Vercel
client token. Tokens expire after 15 minutes, cannot overwrite an existing Blob,
and never place the shared access code in a callback payload. Finalization is an
explicit Python API step; the token handler does not log an upload callback.

The direct-upload prefix is currently fixed to `documents` in both the browser
bundle and Node token handler. Set `SKRIPSICHECK_BLOB_DOCUMENT_PREFIX=documents`
for the hosted pilot; changing only the Python setting will make finalization
reject the upload.

Private Blob does not provide the project's retention policy. A document remains
stored until the application delete flow removes it (or an operator removes it
after auditing the database). Abandoned uploads can also remain when the browser
does not reach finalization, so the operator must audit the `documents/`
namespace during the pilot.
