import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const BUNDLED_PACKAGES = new Map([
  ["@vercel/blob", "Apache-2.0"],
  ["@vercel/oidc", "Apache-2.0"],
  ["async-retry", "MIT"],
  ["retry", "MIT"],
  ["is-buffer", "MIT"],
  ["is-node-process", "MIT"],
  ["jose", "MIT"],
  ["throttleit", "MIT"],
]);

function escapeRegex(value) {
  return value.replace(/[.*+?^$()|[\]\\]/g, "\\$&");
}

test("browser bundle links to its third-party notices", async () => {
  const bundle = await readFile("static/blob-client.js", "utf8");
  assert.match(
    bundle.slice(0, 160),
    /Third-party notices: \/static\/THIRD_PARTY_NOTICES\.txt/,
  );
});

test("notices cover every package bundled into the browser client", async () => {
  const notices = await readFile("static/THIRD_PARTY_NOTICES.txt", "utf8");

  for (const [packageName, expectedLicense] of BUNDLED_PACKAGES) {
    const packageJson = JSON.parse(
      await readFile(
        "node_modules/" + packageName + "/package.json",
        "utf8",
      ),
    );
    assert.equal(
      packageJson.license,
      expectedLicense,
      packageName + " changed license",
    );
    assert.match(
      notices,
      new RegExp(
        escapeRegex(packageName) + " " + escapeRegex(packageJson.version),
      ),
      packageName + "@" + packageJson.version + " is missing from the notices",
    );
  }

  assert.match(notices, /Apache License\s+Version 2\.0, January 2004/);
  assert.match(notices, /Permission is hereby granted, free of charge/);
});
