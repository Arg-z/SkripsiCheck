"use strict";

(() => {
  const MAX_FILE_BYTES = 25 * 1024 * 1024;
  const ALLOWED_EXTENSIONS = new Set(["pdf", "docx", "txt"]);
  const RISK_ORDER = ["VERY HIGH", "HIGH", "MODERATE", "LOW"];
  const state = {
    file: null,
    documentId: null,
    report: null,
    busy: false,
    toastTimer: null,
  };

  const byId = (id) => document.getElementById(id);
  const elements = {
    dropZone: byId("drop-zone"),
    fileInput: byId("file-input"),
    fileCard: byId("file-card"),
    fileName: byId("file-name"),
    fileSize: byId("file-size"),
    fileType: byId("file-type"),
    removeFile: byId("remove-file-button"),
    analyze: byId("analyze-button"),
    topK: byId("top-k-input"),
    minScore: byId("min-score-input"),
    minScoreOutput: byId("min-score-output"),
    processStatus: byId("process-status"),
    message: byId("upload-message"),
    sourceNotice: byId("source-notice"),
    reportSection: byId("report-section"),
    exportJson: byId("export-json-button"),
    exportHtml: byId("export-html-button"),
    deleteDocument: byId("delete-document-button"),
    toast: byId("toast"),
  };

  function makeElement(tag, className, text) {
    const element = document.createElement(tag);
    if (className) element.className = className;
    if (text !== undefined && text !== null) element.textContent = String(text);
    return element;
  }

  function clampScore(value) {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return 0;
    return Math.min(1, Math.max(0, numeric));
  }

  function formatPercent(value, digits = 1) {
    const percentage = clampScore(value) * 100;
    const formatter = new Intl.NumberFormat("id-ID", {
      minimumFractionDigits: digits,
      maximumFractionDigits: digits,
    });
    return `${formatter.format(percentage)}%`;
  }

  function formatBytes(bytes) {
    if (!Number.isFinite(bytes) || bytes < 1) return "0 byte";
    const units = ["byte", "KB", "MB", "GB"];
    const unitIndex = Math.min(Math.floor(Math.log(bytes) / Math.log(1024)), units.length - 1);
    const amount = bytes / (1024 ** unitIndex);
    return `${new Intl.NumberFormat("id-ID", { maximumFractionDigits: unitIndex ? 1 : 0 }).format(amount)} ${units[unitIndex]}`;
  }

  function fileExtension(filename) {
    const parts = String(filename).split(".");
    return parts.length > 1 ? parts.pop().toLowerCase() : "";
  }

  function riskFromScore(value) {
    const score = clampScore(value);
    if (score < 0.4) return "LOW";
    if (score < 0.6) return "MODERATE";
    if (score < 0.8) return "HIGH";
    return "VERY HIGH";
  }

  function riskClass(risk) {
    return `risk-${String(risk).toLowerCase().replaceAll(" ", "-")}`;
  }

  function bestMatch(paragraph) {
    const matches = Array.isArray(paragraph.matches) ? paragraph.matches : [];
    return matches.reduce((best, match) => (
      !best || clampScore(match.final_score) > clampScore(best.final_score) ? match : best
    ), null);
  }

  function paragraphRisk(paragraph) {
    const match = bestMatch(paragraph);
    return match ? riskFromScore(match.final_score) : "LOW";
  }

  function showMessage(text, type = "error") {
    elements.message.textContent = text || "";
    elements.message.classList.toggle("success", type === "success");
  }

  function showToast(text, type = "info") {
    window.clearTimeout(state.toastTimer);
    elements.toast.textContent = text;
    elements.toast.classList.toggle("error", type === "error");
    elements.toast.hidden = false;
    state.toastTimer = window.setTimeout(() => {
      elements.toast.hidden = true;
    }, 5200);
  }

  function validateFile(file) {
    if (!file) return "Pilih dokumen terlebih dahulu.";
    const extension = fileExtension(file.name);
    if (!ALLOWED_EXTENSIONS.has(extension)) return "Format tidak didukung. Pilih PDF, DOCX, atau TXT.";
    if (file.size === 0) return "Dokumen kosong tidak dapat dianalisis.";
    if (file.size > MAX_FILE_BYTES) return "Ukuran dokumen melebihi batas server default 25 MB.";
    return null;
  }

  function selectFile(file) {
    const error = validateFile(file);
    if (error) {
      state.file = null;
      elements.fileInput.value = "";
      elements.fileCard.hidden = true;
      elements.analyze.disabled = true;
      showMessage(error);
      return;
    }
    state.file = file;
    elements.fileName.textContent = file.name;
    elements.fileSize.textContent = formatBytes(file.size);
    elements.fileType.textContent = fileExtension(file.name).toUpperCase();
    elements.fileCard.hidden = false;
    elements.analyze.disabled = state.busy;
    showMessage("");
  }

  function clearSelectedFile() {
    if (state.busy) return;
    state.file = null;
    elements.fileInput.value = "";
    elements.fileCard.hidden = true;
    elements.analyze.disabled = true;
    showMessage("");
  }

  function setBusy(busy) {
    state.busy = busy;
    elements.analyze.disabled = busy || !state.file;
    elements.removeFile.disabled = busy;
    elements.fileInput.disabled = busy;
    elements.analyze.classList.toggle("loading", busy);
    elements.analyze.querySelector("span").textContent = busy ? "Sedang menganalisis" : "Analisis dokumen";
  }

  function resetProgress() {
    elements.processStatus.hidden = false;
    ["step-upload", "step-analyze", "step-report"].forEach((id) => {
      byId(id).classList.remove("active", "done");
    });
  }

  function updateStep(activeStep) {
    const ids = ["step-upload", "step-analyze", "step-report"];
    const activeIndex = ids.indexOf(activeStep);
    ids.forEach((id, index) => {
      const item = byId(id);
      item.classList.toggle("done", index < activeIndex || activeStep === "complete");
      item.classList.toggle("active", index === activeIndex);
    });
  }

  function extractError(payload, status) {
    if (typeof payload === "string" && payload.trim()) return payload;
    if (payload && typeof payload.detail === "string") return payload.detail;
    if (payload && Array.isArray(payload.detail)) {
      const messages = payload.detail.map((entry) => entry && entry.msg).filter(Boolean);
      if (messages.length) return messages.join("; ");
    }
    return `Permintaan gagal (HTTP ${status}).`;
  }

  async function requestJson(url, options = {}) {
    let response;
    try {
      response = await fetch(url, options);
    } catch (error) {
      throw new Error("Tidak dapat terhubung ke server SkripsiCheck.", { cause: error });
    }
    if (response.status === 204) return null;
    const contentType = response.headers.get("content-type") || "";
    let payload = null;
    if (contentType.includes("application/json")) {
      payload = await response.json();
    } else {
      payload = await response.text();
    }
    if (!response.ok) throw new Error(extractError(payload, response.status));
    return payload;
  }

  async function runAnalysis() {
    const validationError = validateFile(state.file);
    if (validationError) {
      showMessage(validationError);
      return;
    }

    setBusy(true);
    resetProgress();
    updateStep("step-upload");
    showMessage("");
    let uploadedDocumentId = null;
    const previousDocumentId = state.documentId;

    try {
      const formData = new FormData();
      formData.append("file", state.file, state.file.name);
      const uploaded = await requestJson("/api/documents", {
        method: "POST",
        body: formData,
        headers: { Accept: "application/json" },
      });
      uploadedDocumentId = uploaded.id;

      updateStep("step-analyze");
      const analysis = await requestJson("/api/analyses", {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify({
          document_id: uploaded.id,
          top_k: Number(elements.topK.value),
          min_score: Number(elements.minScore.value) / 100,
        }),
      });

      updateStep("step-report");
      const report = await requestJson(analysis.report_url, {
        headers: { Accept: "application/json" },
      });
      renderReport(report);
      state.documentId = uploadedDocumentId;

      // Keep only the newest document from this browser session. The previous
      // report remains available until its replacement has completed safely.
      if (previousDocumentId && previousDocumentId !== uploadedDocumentId) {
        try {
          await requestJson(`/api/documents/${encodeURIComponent(previousDocumentId)}`, { method: "DELETE" });
        } catch (_cleanupError) {
          showToast("Analisis selesai, tetapi dokumen sesi sebelumnya belum berhasil dihapus.", "error");
        }
      }

      updateStep("complete");
      showMessage("Analisis selesai. Tinjau setiap temuan dalam konteks akademik.", "success");
      elements.reportSection.hidden = false;
      elements.reportSection.scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      if (uploadedDocumentId) {
        try {
          await requestJson(`/api/documents/${encodeURIComponent(uploadedDocumentId)}`, { method: "DELETE" });
        } catch (_cleanupError) {
          // Preserve the original analysis error; server-side cleanup can be retried manually.
        }
      }
      const message = error instanceof Error ? error.message : "Analisis tidak dapat diselesaikan.";
      showMessage(message);
      showToast(message, "error");
    } finally {
      setBusy(false);
    }
  }

  function tokenPositions(text) {
    const tokens = [];
    const matcher = /[\p{L}\p{N}]+/gu;
    for (const match of String(text).matchAll(matcher)) {
      tokens.push({ value: match[0].toLocaleLowerCase("id-ID"), start: match.index, end: match.index + match[0].length });
    }
    return tokens;
  }

  function overlapIntervals(text, comparison) {
    const tokens = tokenPositions(text);
    const otherTokens = tokenPositions(comparison);
    if (tokens.length < 2 || otherTokens.length < 2) return [];

    const otherPairs = new Set();
    for (let index = 0; index < otherTokens.length - 1; index += 1) {
      otherPairs.add(`${otherTokens[index].value}\u0000${otherTokens[index + 1].value}`);
    }

    const intervals = [];
    for (let index = 0; index < tokens.length - 1; index += 1) {
      const pair = `${tokens[index].value}\u0000${tokens[index + 1].value}`;
      if (otherPairs.has(pair)) intervals.push([tokens[index].start, tokens[index + 1].end]);
    }
    if (!intervals.length) return [];

    const merged = [intervals[0]];
    for (let index = 1; index < intervals.length; index += 1) {
      const current = intervals[index];
      const previous = merged[merged.length - 1];
      if (current[0] <= previous[1] + 1) previous[1] = Math.max(previous[1], current[1]);
      else merged.push(current);
    }
    return merged;
  }

  function appendHighlightedText(container, text, comparison) {
    const content = String(text || "");
    const intervals = overlapIntervals(content, comparison || "");
    if (!intervals.length) {
      container.textContent = content;
      return;
    }
    let cursor = 0;
    intervals.forEach(([start, end]) => {
      if (start > cursor) container.append(document.createTextNode(content.slice(cursor, start)));
      const mark = makeElement("mark", "", content.slice(start, end));
      container.append(mark);
      cursor = end;
    });
    if (cursor < content.length) container.append(document.createTextNode(content.slice(cursor)));
  }

  function renderMetric(label, value) {
    const metric = makeElement("div", "metric");
    metric.append(makeElement("span", "", label), makeElement("strong", "", formatPercent(value)));
    return metric;
  }

  function renderMatch(match, paragraphText) {
    const card = makeElement("article", "match-card");
    const heading = makeElement("div", "match-heading");
    const sourceName = makeElement("span", "source-file", match.source_file || "Sumber tanpa nama");
    sourceName.title = match.source_file || "Sumber tanpa nama";
    const score = makeElement("span", "final-score", formatPercent(match.final_score));
    heading.append(sourceName, score);

    const label = makeElement("p", "text-label", "Matched source text");
    const matchedText = makeElement("p", "matched-text");
    appendHighlightedText(matchedText, match.matched_text, paragraphText);

    const metrics = makeElement("div", "metrics");
    metrics.append(
      renderMetric("Lexical", match.lexical_similarity),
      renderMetric("Semantic", match.semantic_similarity),
      renderMetric("N-gram", match.ngram_overlap),
      renderMetric("Combined", match.final_score),
    );

    const reasonParts = [match.reason || "Perlu ditinjau manual."];
    if (Number.isInteger(match.page)) reasonParts.push(`Halaman sumber: ${match.page}`);
    const reason = makeElement("p", "match-reason", reasonParts.join(" · "));
    card.append(heading, label, matchedText, metrics, reason);
    return card;
  }

  function renderParagraph(paragraph) {
    const risk = paragraphRisk(paragraph);
    const card = makeElement("article", "paragraph-card");
    card.dataset.risk = risk;

    const header = makeElement("div", "paragraph-header");
    header.append(makeElement("span", "paragraph-number", `Paragraf ${paragraph.number}`));
    const meta = makeElement("div", "paragraph-meta");
    meta.append(makeElement("span", "", `${paragraph.word_count || 0} kata`));
    meta.append(makeElement("span", `risk-label ${riskClass(risk)}`, risk));
    header.append(meta);

    const body = makeElement("div", "paragraph-body");
    body.append(makeElement("p", "text-label", "Teks dokumen"));
    const studentText = makeElement("p", "student-text");
    const primaryMatch = bestMatch(paragraph);
    appendHighlightedText(studentText, paragraph.text, primaryMatch ? primaryMatch.matched_text : "");
    body.append(studentText);

    const matches = Array.isArray(paragraph.matches) ? paragraph.matches : [];
    if (matches.length) {
      const matchList = makeElement("div", "match-list");
      matches.forEach((match) => matchList.append(renderMatch(match, paragraph.text)));
      body.append(matchList);
    } else {
      body.append(makeElement("p", "no-match", `Tidak ada kandidat di atas ambang laporan. ${paragraph.candidates_retrieved || 0} kandidat semantik telah ditinjau.`));
    }
    card.append(header, body);
    return card;
  }

  function riskCounts(paragraphs) {
    const counts = { "VERY HIGH": 0, HIGH: 0, MODERATE: 0, LOW: 0 };
    paragraphs.forEach((paragraph) => { counts[paragraphRisk(paragraph)] += 1; });
    return counts;
  }

  function renderCounts(counts, total) {
    const ids = {
      "VERY HIGH": ["count-very-high", "filter-count-very-high"],
      HIGH: ["count-high", "filter-count-high"],
      MODERATE: ["count-moderate", "filter-count-moderate"],
      LOW: ["count-low", "filter-count-low"],
    };
    Object.entries(ids).forEach(([risk, elementIds]) => {
      elementIds.forEach((id) => { byId(id).textContent = String(counts[risk]); });
    });
    byId("filter-count-all").textContent = String(total);
  }

  function sourceContributions(report) {
    const totalWords = report.paragraphs.reduce((sum, paragraph) => sum + Math.max(0, Number(paragraph.word_count) || 0), 0);
    const uniqueChunks = new Map();
    report.paragraphs.forEach((paragraph) => {
      const words = Math.max(0, Number(paragraph.word_count) || 0);
      const matches = Array.isArray(paragraph.matches) ? paragraph.matches : [];
      matches.forEach((match) => {
        const source = match.source_file || "Sumber tanpa nama";
        const key = `${source.toLocaleLowerCase("id-ID")}\u0000${match.chunk_id || match.matched_text || ""}`;
        const contribution = words * clampScore(match.final_score);
        const previous = uniqueChunks.get(key);
        if (!previous || contribution > previous.contribution) uniqueChunks.set(key, { source, contribution });
      });
    });

    const totals = new Map();
    uniqueChunks.forEach(({ source, contribution }) => {
      totals.set(source, (totals.get(source) || 0) + contribution);
    });
    return Array.from(totals, ([source, contribution]) => ({
      source,
      score: totalWords ? Math.min(1, contribution / totalWords) : 0,
    })).sort((left, right) => right.score - left.score || left.source.localeCompare(right.source, "id-ID"));
  }

  function renderTopSources(report) {
    const list = byId("top-sources-list");
    list.replaceChildren();
    const sources = sourceContributions(report).slice(0, 5);
    if (!sources.length) {
      list.append(makeElement("li", "no-sources", "Belum ada sumber di atas ambang laporan."));
      return;
    }
    sources.forEach(({ source, score }) => {
      const item = makeElement("li");
      const name = makeElement("span", "source-name", source);
      name.title = source;
      const value = makeElement("span", "source-score", formatPercent(score));
      const meter = makeElement("progress", "source-meter");
      meter.max = 1;
      meter.value = score;
      meter.setAttribute("aria-label", `Kontribusi ${source}: ${formatPercent(score)}`);
      item.append(name, value, meter);
      list.append(item);
    });
  }

  function renderMethodology(methodology) {
    const container = byId("weight-bars");
    container.replaceChildren();
    const weights = methodology && methodology.weights ? methodology.weights : {};
    const labels = [
      ["Lexical", weights.lexical],
      ["Semantic", weights.semantic],
      ["N-gram", weights.ngram],
    ];
    labels.forEach(([label, rawWeight]) => {
      const weight = clampScore(rawWeight);
      const row = makeElement("div", "weight-row");
      const meter = makeElement("progress");
      meter.max = 1;
      meter.value = weight;
      meter.setAttribute("aria-label", `${label}: ${formatPercent(weight, 0)}`);
      row.append(makeElement("span", "", label), meter, makeElement("strong", "", formatPercent(weight, 0)));
      container.append(row);
    });
    byId("methodology-text").textContent = methodology && methodology.overall
      ? methodology.overall
      : "Skor keseluruhan mempertimbangkan bobot kata dan temuan unik.";
  }

  function scoreDescription(risk) {
    const descriptions = {
      "VERY HIGH": "Banyak sinyal kemiripan kuat ditemukan. Periksa sumber, kutipan, dan konteks setiap temuan.",
      HIGH: "Sejumlah bagian menunjukkan kemiripan kuat dan memerlukan peninjauan manual.",
      MODERATE: "Ada kemiripan yang layak ditinjau, terutama pada temuan semantic yang tinggi.",
      LOW: "Kemiripan terdeteksi relatif rendah, tetapi hasil tetap perlu dibaca dalam konteks.",
    };
    return descriptions[risk];
  }

  function applyFilter(risk) {
    const normalized = risk === "ALL" || RISK_ORDER.includes(risk) ? risk : "ALL";
    let visible = 0;
    document.querySelectorAll(".paragraph-card").forEach((card) => {
      const show = normalized === "ALL" || card.dataset.risk === normalized;
      card.hidden = !show;
      if (show) visible += 1;
    });
    document.querySelectorAll("[data-risk-filter]").forEach((button) => {
      const active = button.dataset.riskFilter === normalized;
      button.classList.toggle("active", active);
      button.setAttribute("aria-pressed", String(active));
    });
    byId("empty-filter").hidden = visible !== 0;
  }

  function renderReport(report) {
    state.report = report;
    const paragraphs = Array.isArray(report.paragraphs) ? report.paragraphs : [];
    const overall = clampScore(report.overall_similarity);
    const overallRisk = riskFromScore(overall);

    byId("report-filename").textContent = report.document && report.document.filename ? report.document.filename : "Dokumen";
    byId("overall-score").textContent = formatPercent(overall);
    byId("overall-risk").textContent = overallRisk;
    byId("score-description").textContent = scoreDescription(overallRisk);
    const ring = byId("score-ring");
    ring.className = `score-ring ${riskClass(overallRisk)}`;
    ring.setAttribute("aria-label", `Overall similarity ${formatPercent(overall)}`);

    const counts = riskCounts(paragraphs);
    renderCounts(counts, paragraphs.length);
    renderTopSources({ ...report, paragraphs });
    renderMethodology(report.methodology);

    const list = byId("paragraph-list");
    list.replaceChildren();
    paragraphs.forEach((paragraph) => list.append(renderParagraph(paragraph)));
    byId("disclaimer-text").textContent = report.disclaimer || "Similarity tidak selalu menunjukkan plagiarisme. Tinjau hasil secara manual.";
    elements.exportJson.disabled = false;
    elements.exportHtml.disabled = false;
    applyFilter("ALL");
  }

  function safeDownloadName(filename, extension) {
    const base = String(filename || "laporan").replace(/\.[^.]+$/, "").replace(/[^a-zA-Z0-9_-]+/g, "-").replace(/^-+|-+$/g, "") || "laporan";
    return `skripsicheck-${base}.${extension}`;
  }

  function downloadBlob(blob, filename) {
    const url = URL.createObjectURL(blob);
    const anchor = makeElement("a");
    anchor.href = url;
    anchor.download = filename;
    document.body.append(anchor);
    anchor.click();
    anchor.remove();
    window.setTimeout(() => URL.revokeObjectURL(url), 1000);
  }

  function exportJson() {
    if (!state.report) {
      showToast("Belum ada laporan untuk diekspor.", "error");
      return;
    }
    const filename = state.report.document ? state.report.document.filename : "laporan";
    const blob = new Blob([JSON.stringify(state.report, null, 2)], { type: "application/json;charset=utf-8" });
    downloadBlob(blob, safeDownloadName(filename, "json"));
    showToast("Laporan JSON berhasil disiapkan.");
  }

  function appendExportText(parent, tag, text) {
    const element = parent.ownerDocument.createElement(tag);
    element.textContent = String(text || "");
    parent.append(element);
    return element;
  }

  function buildExportDocument(report) {
    const exported = document.implementation.createHTMLDocument("Laporan SkripsiCheck");
    exported.documentElement.lang = "id";
    const charset = exported.createElement("meta");
    charset.setAttribute("charset", "utf-8");
    const viewport = exported.createElement("meta");
    viewport.name = "viewport";
    viewport.content = "width=device-width, initial-scale=1";
    const style = exported.createElement("style");
    style.textContent = "body{max-width:920px;margin:40px auto;padding:0 24px;color:#102a2d;font:15px/1.6 system-ui,sans-serif}h1,h2,h3{font-family:Georgia,serif;font-weight:400}.summary{padding:24px;border-radius:14px;color:#fff;background:#102a2d}.score{font-size:38px}.paragraph{margin:22px 0;padding:20px;border:1px solid #d9e1dd;border-radius:12px}.match{margin-top:14px;padding:14px;background:#eef3f1;border-radius:8px}.metrics{display:flex;flex-wrap:wrap;gap:14px;color:#476064;font-size:12px}.disclaimer{margin-top:28px;padding:16px;border-left:4px solid #0d6861;background:#eef3f1}@media print{body{margin:0}.paragraph{break-inside:avoid}}";
    exported.head.replaceChildren(charset, viewport, style);

    const body = exported.body;
    appendExportText(body, "h1", "SkripsiCheck");
    appendExportText(body, "p", report.document && report.document.filename ? report.document.filename : "Dokumen");
    const summary = exported.createElement("section");
    summary.className = "summary";
    appendExportText(summary, "div", "OVERALL SIMILARITY");
    const score = appendExportText(summary, "strong", formatPercent(report.overall_similarity));
    score.className = "score";
    appendExportText(summary, "p", `${report.matched_paragraphs || 0} dari ${report.total_paragraphs || 0} paragraf memiliki temuan di atas ambang.`);
    body.append(summary);
    appendExportText(body, "h2", "Detail kemiripan");

    const paragraphs = Array.isArray(report.paragraphs) ? report.paragraphs : [];
    paragraphs.forEach((paragraph) => {
      const section = exported.createElement("section");
      section.className = "paragraph";
      appendExportText(section, "h3", `Paragraf ${paragraph.number} · ${paragraphRisk(paragraph)}`);
      appendExportText(section, "p", paragraph.text);
      const matches = Array.isArray(paragraph.matches) ? paragraph.matches : [];
      matches.forEach((match) => {
        const matchSection = exported.createElement("div");
        matchSection.className = "match";
        appendExportText(matchSection, "strong", `${match.source_file || "Sumber"} · ${formatPercent(match.final_score)}`);
        appendExportText(matchSection, "p", match.matched_text);
        const metrics = exported.createElement("p");
        metrics.className = "metrics";
        metrics.textContent = `Lexical ${formatPercent(match.lexical_similarity)} · Semantic ${formatPercent(match.semantic_similarity)} · N-gram ${formatPercent(match.ngram_overlap)}`;
        matchSection.append(metrics);
        appendExportText(matchSection, "small", match.reason || "Perlu ditinjau manual.");
        section.append(matchSection);
      });
      body.append(section);
    });
    const disclaimer = exported.createElement("section");
    disclaimer.className = "disclaimer";
    appendExportText(disclaimer, "strong", "Catatan penting");
    appendExportText(disclaimer, "p", report.disclaimer || "Similarity bukan vonis plagiarisme.");
    body.append(disclaimer);
    return exported;
  }

  function exportHtml() {
    if (!state.report) {
      showToast("Belum ada laporan untuk diekspor.", "error");
      return;
    }
    const exported = buildExportDocument(state.report);
    const serialized = `<!doctype html>\n${new XMLSerializer().serializeToString(exported.documentElement)}`;
    const filename = state.report.document ? state.report.document.filename : "laporan";
    downloadBlob(new Blob([serialized], { type: "text/html;charset=utf-8" }), safeDownloadName(filename, "html"));
    showToast("Laporan HTML berhasil disiapkan.");
  }

  async function deleteCurrentDocument() {
    if (!state.documentId) {
      showToast("Dokumen sudah tidak tersimpan di sesi ini.");
      return;
    }
    const confirmed = window.confirm("Hapus dokumen dan seluruh laporan analisis terkait dari server lokal?");
    if (!confirmed) return;
    elements.deleteDocument.disabled = true;
    try {
      await requestJson(`/api/documents/${encodeURIComponent(state.documentId)}`, { method: "DELETE" });
      state.documentId = null;
      state.report = null;
      elements.reportSection.hidden = true;
      elements.exportJson.disabled = true;
      elements.exportHtml.disabled = true;
      clearSelectedFile();
      elements.processStatus.hidden = true;
      showToast("Dokumen dan laporan terkait berhasil dihapus.");
      byId("workspace").scrollIntoView({ behavior: "smooth", block: "start" });
    } catch (error) {
      showToast(error instanceof Error ? error.message : "Dokumen gagal dihapus.", "error");
    } finally {
      elements.deleteDocument.disabled = false;
    }
  }

  function revealSourceNotice() {
    elements.sourceNotice.hidden = false;
    elements.sourceNotice.scrollIntoView({ behavior: "smooth", block: "nearest" });
    showToast("Endpoint pengelolaan sumber belum tersedia; gunakan CLI untuk saat ini.");
  }

  function registerEvents() {
    elements.fileInput.addEventListener("change", () => selectFile(elements.fileInput.files[0]));
    elements.removeFile.addEventListener("click", clearSelectedFile);
    elements.analyze.addEventListener("click", runAnalysis);
    elements.minScore.addEventListener("input", () => {
      elements.minScoreOutput.textContent = `${elements.minScore.value}%`;
    });

    ["dragenter", "dragover"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        if (!state.busy) elements.dropZone.classList.add("drag-active");
      });
    });
    ["dragleave", "drop"].forEach((eventName) => {
      elements.dropZone.addEventListener(eventName, (event) => {
        event.preventDefault();
        elements.dropZone.classList.remove("drag-active");
      });
    });
    elements.dropZone.addEventListener("drop", (event) => {
      if (!state.busy && event.dataTransfer.files.length) selectFile(event.dataTransfer.files[0]);
    });

    byId("sources-nav-button").addEventListener("click", () => {
      byId("source-panel").scrollIntoView({ behavior: "smooth", block: "center" });
      revealSourceNotice();
    });
    byId("add-sources-button").addEventListener("click", revealSourceNotice);
    byId("rebuild-button").addEventListener("click", revealSourceNotice);
    elements.exportJson.addEventListener("click", exportJson);
    elements.exportHtml.addEventListener("click", exportHtml);
    elements.deleteDocument.addEventListener("click", deleteCurrentDocument);

    byId("risk-filters").addEventListener("click", (event) => {
      const button = event.target.closest("[data-risk-filter]");
      if (button) applyFilter(button.dataset.riskFilter);
    });
    document.querySelectorAll("[data-summary-filter]").forEach((button) => {
      button.addEventListener("click", () => {
        applyFilter(button.dataset.summaryFilter);
        byId("paragraph-list").scrollIntoView({ behavior: "smooth", block: "start" });
      });
    });
  }

  registerEvents();
})();
