import { createHash, timingSafeEqual } from "node:crypto";

import { handleUpload } from "@vercel/blob/client";

const DEFAULT_MAX_UPLOAD_MB = 25;
const MAX_CLIENT_PAYLOAD_BYTES = 4096;
const CLIENT_TOKEN_LIFETIME_MS = 15 * 60 * 1000;
const UUID_V4_PATTERN =
  /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i;

export const MEDIA_TYPES_BY_EXTENSION = Object.freeze({
  pdf: "application/pdf",
  docx: "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
  txt: "text/plain",
});

export class UploadRequestError extends Error {
  constructor(message, statusCode = 400) {
    super(message);
    this.name = "UploadRequestError";
    this.statusCode = statusCode;
  }
}

export function parseMaximumUploadBytes(rawValue) {
  const value = rawValue === undefined || rawValue === "" ? DEFAULT_MAX_UPLOAD_MB : Number(rawValue);
  if (!Number.isSafeInteger(value) || value < 1 || value > 500) {
    throw new UploadRequestError("Upload service is not configured correctly.", 503);
  }
  return value * 1024 * 1024;
}

export function timingSafeTokenMatches(providedToken, expectedToken) {
  if (typeof providedToken !== "string" || typeof expectedToken !== "string") return false;
  if (providedToken.length === 0 || expectedToken.length === 0) return false;

  // Hash both values before timingSafeEqual so differing UTF-8 lengths do not
  // create an early-return timing oracle.
  const providedDigest = createHash("sha256").update(providedToken, "utf8").digest();
  const expectedDigest = createHash("sha256").update(expectedToken, "utf8").digest();
  return timingSafeEqual(providedDigest, expectedDigest);
}

function parseClientPayload(clientPayload) {
  if (typeof clientPayload !== "string" || clientPayload.length === 0) {
    throw new UploadRequestError("Upload authorization is required.", 401);
  }
  if (Buffer.byteLength(clientPayload, "utf8") > MAX_CLIENT_PAYLOAD_BYTES) {
    throw new UploadRequestError("Upload request is invalid.");
  }
  try {
    const value = JSON.parse(clientPayload);
    if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error();
    return value;
  } catch {
    throw new UploadRequestError("Upload request is invalid.");
  }
}

function requireUuid(value, label) {
  if (typeof value !== "string" || !UUID_V4_PATTERN.test(value)) {
    throw new UploadRequestError(`${label} is invalid.`);
  }
  return value.toLowerCase();
}

export function authorizeDirectUpload({
  pathname,
  clientPayload,
  multipart,
  expectedAccessToken = process.env.SKRIPSICHECK_ACCESS_TOKEN,
  maxUploadMb = process.env.SKRIPSICHECK_MAX_UPLOAD_MB,
}) {
  if (typeof expectedAccessToken !== "string" || expectedAccessToken.length < 32) {
    throw new UploadRequestError("Upload service is not configured correctly.", 503);
  }

  const payload = parseClientPayload(clientPayload);
  if (!timingSafeTokenMatches(payload.accessToken, expectedAccessToken)) {
    throw new UploadRequestError("Upload authorization failed.", 401);
  }
  if (multipart !== true) {
    throw new UploadRequestError("Multipart client upload is required.");
  }

  const sessionId = requireUuid(payload.sessionId, "Upload session");
  const documentId = requireUuid(payload.documentId, "Document ID");
  const extension = typeof payload.extension === "string" ? payload.extension.toLowerCase() : "";
  const mediaType = MEDIA_TYPES_BY_EXTENSION[extension];
  if (!mediaType || payload.mediaType !== mediaType) {
    throw new UploadRequestError("Document type is not supported.");
  }

  const expectedPathname = `documents/${sessionId}/${documentId}.${extension}`;
  if (pathname !== expectedPathname) {
    throw new UploadRequestError("Upload pathname is invalid.");
  }

  const maximumSizeInBytes = parseMaximumUploadBytes(maxUploadMb);
  if (
    !Number.isSafeInteger(payload.sizeBytes) ||
    payload.sizeBytes < 1 ||
    payload.sizeBytes > maximumSizeInBytes
  ) {
    throw new UploadRequestError("Document size is invalid.");
  }

  return {
    allowedContentTypes: [mediaType],
    maximumSizeInBytes,
    addRandomSuffix: false,
    allowOverwrite: false,
    validUntil: Date.now() + CLIENT_TOKEN_LIFETIME_MS,
    tokenPayload: JSON.stringify({
      sessionId,
      documentId,
      pathname: expectedPathname,
      extension,
      mediaType,
      declaredSizeBytes: payload.sizeBytes,
    }),
  };
}

function normalizeRequestBody(body) {
  if (typeof body !== "string") return body;
  try {
    return JSON.parse(body);
  } catch {
    throw new UploadRequestError("Upload request is invalid.");
  }
}

function sendJson(response, statusCode, payload) {
  response.status(statusCode);
  response.setHeader("Cache-Control", "no-store");
  response.setHeader("Content-Security-Policy", "default-src 'none'; frame-ancestors 'none'");
  response.setHeader("Content-Type", "application/json; charset=utf-8");
  response.setHeader("Referrer-Policy", "no-referrer");
  response.setHeader("X-Content-Type-Options", "nosniff");
  response.setHeader("X-Frame-Options", "DENY");
  return response.json(payload);
}

export default async function blobUploadHandler(request, response) {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return sendJson(response, 405, { detail: "Method not allowed." });
  }

  try {
    const body = normalizeRequestBody(request.body);
    if (!body || typeof body !== "object") {
      throw new UploadRequestError("Upload request is invalid.");
    }

    const result = await handleUpload({
      body,
      request,
      onBeforeGenerateToken: async (pathname, clientPayload, multipart) =>
        authorizeDirectUpload({ pathname, clientPayload, multipart }),
    });
    return sendJson(response, 200, result);
  } catch (error) {
    const statusCode = error instanceof UploadRequestError ? error.statusCode : 400;
    const detail =
      error instanceof UploadRequestError ? error.message : "Upload request was rejected.";
    return sendJson(response, statusCode, { detail });
  }
}
