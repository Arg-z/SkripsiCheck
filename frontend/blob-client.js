import { upload } from "@vercel/blob/client";

const DEFAULT_MAX_FILE_BYTES = 25 * 1024 * 1024;
const UUID_V4_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;
const MEDIA_TYPES_BY_EXTENSION = Object.freeze({
  pdf: "application/pdf",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  txt: "text/plain",
});

function uuidV4() {
  if (!globalThis.crypto?.randomUUID) {
    throw new Error("Browser ini tidak mendukung pembuatan upload ID yang aman.");
  }
  return globalThis.crypto.randomUUID();
}

function fileExtension(filename) {
  const match = /\.([^.]+)$/.exec(String(filename));
  return match ? match[1].toLowerCase() : "";
}

function requireUuid(value, label) {
  if (typeof value !== "string" || !UUID_V4_PATTERN.test(value)) {
    throw new Error(`${label} tidak valid.`);
  }
  return value.toLowerCase();
}

function sameOriginUploadUrl(value) {
  if (!globalThis.location?.href) {
    throw new Error("Upload Blob hanya dapat dijalankan di browser.");
  }
  const resolved = new URL(value, globalThis.location.href);
  if (resolved.origin !== globalThis.location.origin) {
    throw new Error("Endpoint upload harus berada pada origin yang sama.");
  }
  return resolved.href;
}

export function preparePrivateBlobUpload(
  file,
  {
    accessToken,
    sessionId = uuidV4(),
    documentId = uuidV4(),
    maximumSizeBytes = DEFAULT_MAX_FILE_BYTES,
  } = {},
) {
  if (!(file instanceof Blob) || typeof file.name !== "string") {
    throw new TypeError("Pilih dokumen yang valid.");
  }
  if (typeof accessToken !== "string" || accessToken.length === 0 || accessToken.length > 512) {
    throw new Error("Kode akses upload diperlukan.");
  }
  if (!Number.isSafeInteger(maximumSizeBytes) || maximumSizeBytes < 1) {
    throw new TypeError("Batas ukuran upload tidak valid.");
  }
  if (file.size < 1) throw new Error("Dokumen kosong tidak dapat diunggah.");
  if (file.size > maximumSizeBytes) {
    throw new Error(`Ukuran dokumen melebihi ${Math.floor(maximumSizeBytes / 1024 / 1024)} MB.`);
  }

  const safeSessionId = requireUuid(sessionId, "Sesi upload");
  const safeDocumentId = requireUuid(documentId, "ID dokumen");
  const extension = fileExtension(file.name);
  const mediaType = MEDIA_TYPES_BY_EXTENSION[extension];
  if (!mediaType) throw new Error("Hanya PDF, DOCX, dan TXT yang didukung.");
  if (file.type && file.type.toLowerCase() !== mediaType) {
    throw new Error("Tipe dokumen tidak sesuai dengan ekstensi file.");
  }

  const pathname = `documents/${safeSessionId}/${safeDocumentId}.${extension}`;
  return {
    pathname,
    sessionId: safeSessionId,
    documentId: safeDocumentId,
    extension,
    mediaType,
    clientPayload: JSON.stringify({
      accessToken,
      sessionId: safeSessionId,
      documentId: safeDocumentId,
      extension,
      mediaType,
      sizeBytes: file.size,
    }),
  };
}

export async function uploadPrivateDocument(
  file,
  {
    accessToken,
    sessionId,
    documentId,
    maximumSizeBytes = DEFAULT_MAX_FILE_BYTES,
    handleUploadUrl = "/api/blob-upload",
    onUploadProgress,
    abortSignal,
  } = {},
) {
  const prepared = preparePrivateBlobUpload(file, {
    accessToken,
    sessionId,
    documentId,
    maximumSizeBytes,
  });
  const safeHandleUploadUrl = sameOriginUploadUrl(handleUploadUrl);
  const blob = await upload(prepared.pathname, file, {
    access: "private",
    contentType: prepared.mediaType,
    handleUploadUrl: safeHandleUploadUrl,
    clientPayload: prepared.clientPayload,
    multipart: true,
    onUploadProgress,
    abortSignal,
  });

  // Never return the access token or serialized client payload to application
  // state. The caller receives only identifiers and Blob's upload receipt.
  return {
    blob,
    sessionId: prepared.sessionId,
    documentId: prepared.documentId,
    pathname: prepared.pathname,
    extension: prepared.extension,
    mediaType: prepared.mediaType,
    sizeBytes: file.size,
  };
}
