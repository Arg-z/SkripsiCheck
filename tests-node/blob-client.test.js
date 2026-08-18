import assert from "node:assert/strict";
import test from "node:test";

import { preparePrivateBlobUpload } from "../frontend/blob-client.js";

const ACCESS_TOKEN = "a-strong-test-token-that-is-at-least-32-characters";
const SESSION_ID = "a1169f80-44c2-49b6-8a43-9286f5f05d39";
const DOCUMENT_ID = "ce6c22ed-e680-4a17-bda8-ef2915f14eb8";

test("browser wrapper prepares a namespaced private document upload", () => {
  const file = new File(["isi skripsi"], "skripsi.txt", { type: "text/plain" });
  const prepared = preparePrivateBlobUpload(file, {
    accessToken: ACCESS_TOKEN,
    sessionId: SESSION_ID,
    documentId: DOCUMENT_ID,
  });

  assert.equal(prepared.pathname, `documents/${SESSION_ID}/${DOCUMENT_ID}.txt`);
  assert.equal(prepared.mediaType, "text/plain");
  const payload = JSON.parse(prepared.clientPayload);
  assert.equal(payload.accessToken, ACCESS_TOKEN);
  assert.equal(payload.sizeBytes, file.size);
});

test("browser wrapper rejects invalid UUID, MIME, extension, and size", () => {
  const textFile = new File(["text"], "skripsi.txt", { type: "text/plain" });
  assert.throws(
    () =>
      preparePrivateBlobUpload(textFile, {
        accessToken: ACCESS_TOKEN,
        sessionId: "../escape",
        documentId: DOCUMENT_ID,
      }),
    /sesi/i,
  );

  const wrongMime = new File(["text"], "skripsi.pdf", { type: "text/plain" });
  assert.throws(
    () =>
      preparePrivateBlobUpload(wrongMime, {
        accessToken: ACCESS_TOKEN,
        sessionId: SESSION_ID,
        documentId: DOCUMENT_ID,
      }),
    /tipe/i,
  );

  const unsupported = new File(["text"], "skripsi.exe", {
    type: "application/octet-stream",
  });
  assert.throws(
    () =>
      preparePrivateBlobUpload(unsupported, {
        accessToken: ACCESS_TOKEN,
        sessionId: SESSION_ID,
        documentId: DOCUMENT_ID,
      }),
    /pdf/i,
  );

  assert.throws(
    () =>
      preparePrivateBlobUpload(textFile, {
        accessToken: ACCESS_TOKEN,
        sessionId: SESSION_ID,
        documentId: DOCUMENT_ID,
        maximumSizeBytes: 1,
      }),
    /melebihi/i,
  );
});
