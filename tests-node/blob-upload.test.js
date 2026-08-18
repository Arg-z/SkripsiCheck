import assert from "node:assert/strict";
import test from "node:test";

import {
  authorizeDirectUpload,
  parseMaximumUploadBytes,
  timingSafeTokenMatches,
  UploadRequestError,
} from "../api/blob-upload.js";
import blobUploadHandler from "../api/blob-upload.js";

const ACCESS_TOKEN = "a-strong-test-token-that-is-at-least-32-characters";
const SESSION_ID = "a1169f80-44c2-49b6-8a43-9286f5f05d39";
const DOCUMENT_ID = "ce6c22ed-e680-4a17-bda8-ef2915f14eb8";

function request(overrides = {}) {
  const payload = {
    accessToken: ACCESS_TOKEN,
    sessionId: SESSION_ID,
    documentId: DOCUMENT_ID,
    extension: "pdf",
    mediaType: "application/pdf",
    sizeBytes: 1024,
    ...overrides.payload,
  };
  return {
    pathname: `documents/${SESSION_ID}/${DOCUMENT_ID}.pdf`,
    clientPayload: JSON.stringify(payload),
    multipart: true,
    expectedAccessToken: ACCESS_TOKEN,
    maxUploadMb: "25",
    ...overrides,
    payload: undefined,
  };
}

test("token comparison handles equal and unequal lengths safely", () => {
  assert.equal(timingSafeTokenMatches(ACCESS_TOKEN, ACCESS_TOKEN), true);
  assert.equal(timingSafeTokenMatches("short", ACCESS_TOKEN), false);
  assert.equal(timingSafeTokenMatches("", ACCESS_TOKEN), false);
  assert.equal(timingSafeTokenMatches(undefined, ACCESS_TOKEN), false);
});

test("maximum upload size uses MiB and rejects unsafe configuration", () => {
  assert.equal(parseMaximumUploadBytes(undefined), 25 * 1024 * 1024);
  assert.equal(parseMaximumUploadBytes("10"), 10 * 1024 * 1024);
  assert.throws(() => parseMaximumUploadBytes("0"), UploadRequestError);
  assert.throws(() => parseMaximumUploadBytes("1.5"), UploadRequestError);
  assert.throws(() => parseMaximumUploadBytes("NaN"), UploadRequestError);
});

test("authorized request returns narrow Vercel token constraints", () => {
  const result = authorizeDirectUpload(request());
  assert.deepEqual(result.allowedContentTypes, ["application/pdf"]);
  assert.equal(result.maximumSizeInBytes, 25 * 1024 * 1024);
  assert.equal(result.addRandomSuffix, false);
  assert.equal(result.allowOverwrite, false);
  assert.ok(result.validUntil > Date.now());
  assert.ok(result.validUntil <= Date.now() + 15 * 60 * 1000);
  const tokenPayload = JSON.parse(result.tokenPayload);
  assert.equal(tokenPayload.pathname, `documents/${SESSION_ID}/${DOCUMENT_ID}.pdf`);
  assert.equal(tokenPayload.declaredSizeBytes, 1024);
  assert.equal("accessToken" in tokenPayload, false);
  assert.equal(result.tokenPayload.includes(ACCESS_TOKEN), false);
});

test("wrong access token is rejected without echoing it", () => {
  const input = request({ payload: { accessToken: "incorrect-private-value" } });
  assert.throws(
    () => authorizeDirectUpload(input),
    (error) => {
      assert.equal(error.statusCode, 401);
      assert.equal(error.message.includes("incorrect-private-value"), false);
      return true;
    },
  );
});

test("pathname traversal or mismatched identifiers are rejected", () => {
  assert.throws(
    () => authorizeDirectUpload(request({ pathname: "documents/../victim.pdf" })),
    /pathname/i,
  );
  assert.throws(
    () => authorizeDirectUpload(request({ payload: { sessionId: "not-a-uuid" } })),
    /session/i,
  );
});

test("MIME, extension, size, and multipart constraints are enforced", () => {
  assert.throws(
    () => authorizeDirectUpload(request({ payload: { mediaType: "text/plain" } })),
    /type/i,
  );
  assert.throws(
    () => authorizeDirectUpload(request({ payload: { extension: "exe" } })),
    /type/i,
  );
  assert.throws(
    () => authorizeDirectUpload(request({ payload: { sizeBytes: 26 * 1024 * 1024 } })),
    /size/i,
  );
  assert.throws(() => authorizeDirectUpload(request({ multipart: false })), /multipart/i);
});

test("missing or weak server access token fails closed", () => {
  assert.throws(
    () => authorizeDirectUpload(request({ expectedAccessToken: undefined })),
    (error) => error.statusCode === 503,
  );
  assert.throws(
    () => authorizeDirectUpload(request({ expectedAccessToken: "too-short" })),
    (error) => error.statusCode === 503,
  );
});

test("default authorization reads the shared Python API token environment name", () => {
  const previous = process.env.SKRIPSICHECK_ACCESS_TOKEN;
  process.env.SKRIPSICHECK_ACCESS_TOKEN = ACCESS_TOKEN;
  try {
    const input = request();
    delete input.expectedAccessToken;
    assert.deepEqual(authorizeDirectUpload(input).allowedContentTypes, ["application/pdf"]);
  } finally {
    if (previous === undefined) delete process.env.SKRIPSICHECK_ACCESS_TOKEN;
    else process.env.SKRIPSICHECK_ACCESS_TOKEN = previous;
  }
});

test("upload endpoint returns hardened no-store responses", async () => {
  const headers = new Map();
  let responseStatus;
  let responseBody;
  const response = {
    status(value) {
      responseStatus = value;
      return this;
    },
    setHeader(name, value) {
      headers.set(name.toLowerCase(), value);
    },
    json(value) {
      responseBody = value;
      return value;
    },
  };

  await blobUploadHandler({ method: "GET" }, response);

  assert.equal(responseStatus, 405);
  assert.deepEqual(responseBody, { detail: "Method not allowed." });
  assert.equal(headers.get("cache-control"), "no-store");
  assert.equal(headers.get("x-content-type-options"), "nosniff");
  assert.equal(headers.get("x-frame-options"), "DENY");
  assert.match(headers.get("content-security-policy"), /default-src 'none'/);
});
