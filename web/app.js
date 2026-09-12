const state = {
  documents: [],
  messages: [],
  polling: null,
  asking: false,
};

const elements = {
  dropzone: document.querySelector("#dropzone"),
  fileInput: document.querySelector("#file-input"),
  uploadProgress: document.querySelector("#upload-progress"),
  uploadLabel: document.querySelector("#upload-label"),
  uploadPercent: document.querySelector("#upload-percent"),
  progressFill: document.querySelector("#progress-fill"),
  documentList: document.querySelector("#document-list"),
  emptyLibrary: document.querySelector("#empty-library"),
  documentTemplate: document.querySelector("#document-template"),
  messageTemplate: document.querySelector("#message-template"),
  docCount: document.querySelector("#doc-count"),
  healthDot: document.querySelector("#health-dot"),
  healthLabel: document.querySelector("#health-label"),
  conversation: document.querySelector("#conversation"),
  welcome: document.querySelector("#welcome"),
  questionForm: document.querySelector("#question-form"),
  questionInput: document.querySelector("#question-input"),
  sendButton: document.querySelector("#send-button"),
  composerHint: document.querySelector("#composer-hint"),
  clearChat: document.querySelector("#clear-chat"),
};

async function api(path, options = {}) {
  const response = await fetch(path, options);
  const contentType = response.headers.get("content-type") || "";
  const payload = contentType.includes("application/json") ? await response.json() : await response.text();
  if (!response.ok) {
    const detail = typeof payload === "object" ? payload.detail || payload.message : payload;
    throw new Error(detail || `Request failed (${response.status})`);
  }
  return payload;
}

function toast(message, isError = false) {
  const current = document.querySelector(".toast");
  if (current) current.remove();
  const node = document.createElement("div");
  node.className = `toast${isError ? " error" : ""}`;
  node.textContent = message;
  document.body.appendChild(node);
  setTimeout(() => node.remove(), 4200);
}

function formatBytes(bytes = 0) {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
}

function normalizedStatus(documentItem) {
  return String(documentItem.status || documentItem.current_version?.status || "queued").toLowerCase();
}

function isReady(documentItem) {
  return normalizedStatus(documentItem) === "ready";
}

function statusClass(status) {
  if (status === "ready") return "ready";
  if (status.includes("failed") || status === "rejected" || status === "needs_ocr") return "failed";
  return "working";
}

function renderDocuments() {
  elements.documentList.replaceChildren();
  elements.docCount.textContent = state.documents.length;
  elements.emptyLibrary.classList.toggle("hidden", state.documents.length > 0);

  for (const item of state.documents) {
    const fragment = elements.documentTemplate.content.cloneNode(true);
    const row = fragment.querySelector(".document-row");
    const status = normalizedStatus(item);
    row.classList.add(statusClass(status));
    fragment.querySelector(".document-name").textContent = item.filename || item.display_name || "Untitled document";
    const chunks = item.chunk_count ? ` · ${item.chunk_count} chunks` : "";
    fragment.querySelector(".document-meta").textContent = `${status.replaceAll("_", " ")} · ${formatBytes(item.size_bytes || item.size || 0)}${chunks}`;
    const glyph = (item.filename || "doc").split(".").pop().slice(0, 4).toUpperCase();
    fragment.querySelector(".file-glyph").textContent = glyph;
    fragment.querySelector(".delete-document").addEventListener("click", () => deleteDocument(item.id, item.filename));
    elements.documentList.appendChild(fragment);
  }

  const readyCount = state.documents.filter(isReady).length;
  elements.questionInput.disabled = readyCount === 0 || state.asking;
  elements.sendButton.disabled = readyCount === 0 || state.asking;
  elements.composerHint.textContent = readyCount
    ? `${readyCount} ready document${readyCount === 1 ? "" : "s"} · answers cite retrieved chunks`
    : state.documents.length
      ? "Your documents are being indexed…"
      : "Upload and index a document to begin";

  const needsPolling = state.documents.some((item) => statusClass(normalizedStatus(item)) === "working");
  if (needsPolling && !state.polling) state.polling = setInterval(loadDocuments, 2200);
  if (!needsPolling && state.polling) {
    clearInterval(state.polling);
    state.polling = null;
  }
}

async function loadDocuments() {
  try {
    const result = await api("/api/documents");
    state.documents = Array.isArray(result) ? result : result.items || [];
    renderDocuments();
  } catch (error) {
    elements.healthDot.className = "health-dot bad";
    elements.healthLabel.textContent = "Pipeline unavailable";
  }
}

async function checkHealth() {
  try {
    const result = await api("/api/health");
    const healthy = result.status === "ok";
    elements.healthDot.className = `health-dot ${healthy ? "good" : "bad"}`;
    elements.healthLabel.textContent = result.provider
      ? `${healthy ? "Pipeline online" : "Pipeline degraded"} · ${result.provider}`
      : healthy ? "Pipeline online" : "Pipeline degraded";
  } catch (_) {
    elements.healthDot.className = "health-dot bad";
    elements.healthLabel.textContent = "Pipeline unavailable";
  }
}

async function uploadFiles(files) {
  const accepted = ["pdf", "docx", "md", "markdown", "html", "htm", "txt"];
  const selected = [...files].filter((file) => accepted.includes(file.name.split(".").pop().toLowerCase()));
  if (!selected.length) {
    toast("Choose a PDF, DOCX, Markdown, HTML, or text file.", true);
    return;
  }
  elements.uploadProgress.classList.remove("hidden");
  let completed = 0;
  for (const file of selected) {
    try {
      elements.uploadLabel.textContent = `Uploading ${file.name}`;
      const form = new FormData();
      form.append("file", file);
      await api("/api/documents", { method: "POST", body: form });
      completed += 1;
      const percent = Math.round((completed / selected.length) * 100);
      elements.uploadPercent.textContent = `${percent}%`;
      elements.progressFill.style.width = `${percent}%`;
    } catch (error) {
      toast(`${file.name}: ${error.message}`, true);
    }
  }
  elements.uploadLabel.textContent = "Queued for ingestion";
  await loadDocuments();
  setTimeout(() => {
    elements.uploadProgress.classList.add("hidden");
    elements.progressFill.style.width = "0";
    elements.uploadPercent.textContent = "0%";
    elements.fileInput.value = "";
  }, 900);
}

async function deleteDocument(id, filename) {
  if (!window.confirm(`Delete “${filename || "this document"}” and all of its indexed chunks?`)) return;
  try {
    await api(`/api/documents/${encodeURIComponent(id)}`, { method: "DELETE" });
    state.documents = state.documents.filter((item) => item.id !== id);
    renderDocuments();
    toast("Document removed from the knowledge base.");
  } catch (error) {
    toast(error.message, true);
  }
}

function appendMessage(role, content, sources = [], diagnostics = null) {
  if (elements.welcome) elements.welcome.classList.add("hidden");
  const fragment = elements.messageTemplate.content.cloneNode(true);
  const message = fragment.querySelector(".message");
  message.classList.add(role);
  fragment.querySelector(".message-role").textContent = role === "user" ? "YOU" : "INDEX";
  const body = fragment.querySelector(".message-body");
  const paragraph = document.createElement("p");
  paragraph.textContent = content;
  body.appendChild(paragraph);

  if (sources.length) {
    const wrapper = document.createElement("div");
    wrapper.className = "sources";
    const heading = document.createElement("div");
    heading.className = "sources-heading";
    heading.textContent = `RETRIEVED EVIDENCE · ${sources.length}`;
    wrapper.appendChild(heading);
    sources.forEach((source, index) => {
      const details = document.createElement("details");
      details.className = "source-card";
      const summary = document.createElement("summary");
      const title = document.createElement("span");
      title.textContent = `[${source.source_id || `S${index + 1}`}] ${source.filename || "Document"}`;
      const location = document.createElement("span");
      location.className = "source-location";
      const locationParts = [];
      if (source.page_start) locationParts.push(`page ${source.page_start}`);
      if (source.section_path?.length) locationParts.push(source.section_path.join(" › "));
      location.textContent = locationParts.join(" · ") || `chunk ${source.ordinal ?? index + 1}`;
      summary.append(title, location);
      const sourceText = document.createElement("div");
      sourceText.className = "source-text";
      sourceText.textContent = source.text || "";
      if (Number.isInteger(source.match_count) && source.match_count > 0) {
        const score = document.createElement("div");
        score.className = "score";
        score.textContent = `${source.match_count} EXACT ${source.match_count === 1 ? "MATCH" : "MATCHES"}`;
        sourceText.prepend(score);
      } else if (typeof source.score === "number") {
        const score = document.createElement("div");
        score.className = "score";
        score.textContent = `RELEVANCE ${source.score.toFixed(3)}`;
        sourceText.prepend(score);
      }
      details.append(summary, sourceText);
      wrapper.appendChild(details);
    });
    body.appendChild(wrapper);
  }
  if (diagnostics?.abstained) {
    const note = document.createElement("p");
    note.className = "source-location";
    note.textContent = "The system abstained because the retrieved evidence was not strong enough.";
    body.appendChild(note);
  }
  elements.conversation.appendChild(fragment);
  elements.conversation.scrollTop = elements.conversation.scrollHeight;
  return message;
}

async function askQuestion(question) {
  const trimmed = question.trim();
  if (!trimmed || state.asking) return;
  state.asking = true;
  elements.questionInput.value = "";
  elements.questionInput.style.height = "auto";
  renderDocuments();
  appendMessage("user", trimmed);
  state.messages.push({ role: "user", content: trimmed });
  const pending = appendMessage("assistant", "Searching, ranking, and checking the evidence…");
  pending.querySelector(".message-body").classList.add("thinking");

  try {
    const result = await api("/api/answers", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question: trimmed, history: state.messages.slice(-6, -1) }),
    });
    pending.remove();
    appendMessage("assistant", result.answer, result.sources || [], result.diagnostics || result);
    state.messages.push({ role: "assistant", content: result.answer });
  } catch (error) {
    pending.remove();
    appendMessage("assistant", `I couldn't complete that search: ${error.message}`);
  } finally {
    state.asking = false;
    renderDocuments();
    elements.questionInput.focus();
  }
}

elements.fileInput.addEventListener("change", (event) => uploadFiles(event.target.files));
["dragenter", "dragover"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  elements.dropzone.classList.add("dragging");
}));
["dragleave", "drop"].forEach((eventName) => elements.dropzone.addEventListener(eventName, (event) => {
  event.preventDefault();
  elements.dropzone.classList.remove("dragging");
}));
elements.dropzone.addEventListener("drop", (event) => uploadFiles(event.dataTransfer.files));

elements.questionForm.addEventListener("submit", (event) => {
  event.preventDefault();
  askQuestion(elements.questionInput.value);
});
elements.questionInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    elements.questionForm.requestSubmit();
  }
});
elements.questionInput.addEventListener("input", () => {
  elements.questionInput.style.height = "auto";
  elements.questionInput.style.height = `${Math.min(elements.questionInput.scrollHeight, 150)}px`;
});
elements.clearChat.addEventListener("click", () => {
  state.messages = [];
  [...elements.conversation.querySelectorAll(".message")].forEach((message) => message.remove());
  elements.welcome?.classList.remove("hidden");
});
document.querySelectorAll("[data-question]").forEach((button) => {
  button.addEventListener("click", () => askQuestion(button.dataset.question));
});

checkHealth();
loadDocuments();
