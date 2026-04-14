/* ─── State ──────────────────────────────────────────── */
let currentPage = 0;
let userName = "";
let currentSessionId = null;
let fileName = "";
let documentCount = 0;
let queryCount = 0;
let isStreaming = false;
let currentEventSource = null;
let currentDashboardPath = null;
let fileStatuses = [];
let uploadInProgress = false;
let allFilesProcessed = false;

/* ─── Element refs ───────────────────────────────────── */
const nameInput               = document.getElementById("nameInput");
const nameErr                 = document.getElementById("nameErr");
const startBtn                = document.getElementById("startBtn");
const fileInput               = document.getElementById("fileInput");
const uploadBox               = document.getElementById("uploadBox");
const uploadStatus            = document.getElementById("uploadStatus");
const progressFill            = document.getElementById("progressFill");
const nextToChat              = document.getElementById("nextToChat");
const backToHero              = document.getElementById("backToHero");
const backToUpload            = document.getElementById("backToUpload");
const chatTitle               = document.getElementById("chatTitle");
const chatDoc                 = document.getElementById("chatDoc");
const chatBadge               = document.getElementById("chatBadge");
const chatMsgs                = document.getElementById("chatMsgs");
const chatStatusBar           = document.getElementById("chatStatusBar");
const questionInput           = document.getElementById("questionInput");
const askButton               = document.getElementById("askButton");
const fileStatusList          = document.getElementById("fileStatusList");
const chatInputRow            = document.getElementById("chatInputRow");
const evidencePanel           = document.getElementById("evidencePanel");
const collapseEvidence        = document.getElementById("collapseEvidence");
const metricDocs              = document.getElementById("metricDocs");
const metricQueries           = document.getElementById("metricQueries");
const metricSession           = document.getElementById("metricSession");
const stepDots                = document.querySelectorAll(".step-dot");

// Evidence pane elements
const retrievedContext        = document.getElementById("retrievedContext");
const retrievedImages         = document.getElementById("retrievedImages");
const retrievedGeneratedImages = document.getElementById("retrievedGeneratedImages");
const plotsCount              = document.getElementById("plotsCount");
const retrievedCode           = document.getElementById("retrievedCode");
const retrievedReasoning      = document.getElementById("retrievedReasoning");
const retrievedOutput         = document.getElementById("retrievedOutput");

// Dashboard elements
const dashboardIframe         = document.getElementById("dashboardIframe");
const noDashboardState        = document.getElementById("noDashboardState");
const dashOpenBtn             = document.getElementById("dashOpenBtn");
const dashRefreshBtn          = document.getElementById("dashRefreshBtn");
const slidesCount             = document.getElementById("slidesCount");
const retrievedSlides         = document.getElementById("retrievedSlides");

/* ─── Session ID ─────────────────────────────────────── */
function buildSessionId() {
  if (window.crypto && typeof window.crypto.randomUUID === "function") {
    return window.crypto.randomUUID();
  }
  return "xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx".replace(/[xy]/g, (c) => {
    const r = Math.floor(Math.random() * 16);
    return (c === "x" ? r : (r & 0x3) | 0x8).toString(16);
  });
}

/* ─── Path normaliser ────────────────────────────────── */
/**
 * The backend returns paths in several formats:
 *   /temp/session/outputs/plot_generated/plot_0.png   (already correct)
 *   temp/session/outputs/...                          (missing leading slash)
 *   C:\Users\...\DataLens\backend\temp\...            (Windows absolute)
 *   /absolute/posix/path/to/temp/session/...          (server absolute)
 *
 * We need a URL the browser can fetch, rooted at /temp/...
 */
function normalisePath(raw) {
  if (!raw) return null;
  let p = String(raw).replace(/\\/g, "/");

  // Already a proper /temp/... URL
  if (p.startsWith("/temp/")) return p;

  // Has /temp/ somewhere inside (server absolute path)
  const tempIdx = p.indexOf("/temp/");
  if (tempIdx !== -1) return p.slice(tempIdx);

  // Relative starting with temp/
  if (p.startsWith("temp/")) return "/" + p;

  // Fallback: prepend slash and hope for the best
  return p.startsWith("/") ? p : "/" + p;
}

/* ─── Navigation ─────────────────────────────────────── */
function goTo(n) {
  if (n === 2 && !allFilesProcessed) {
    setUploadStatus("Please finish file processing before going to chat.", "err");
    return;
  }
  document.getElementById("s" + currentPage).classList.remove("visible");
  stepDots[currentPage].classList.remove("active");
  currentPage = n;
  document.getElementById("s" + currentPage).classList.add("visible");
  stepDots[currentPage].classList.add("active");
  window.scrollTo(0, 0);
}

stepDots.forEach((dot) => {
  dot.addEventListener("click", () => goTo(parseInt(dot.dataset.page)));
});

/* ─── Page 1: Start session ──────────────────────────── */
startBtn.addEventListener("click", () => {
  const val = nameInput.value.trim();
  if (!val) {
    nameErr.textContent = "Please enter your name to continue.";
    nameInput.focus();
    return;
  }
  nameErr.textContent = "";
  userName = val;
  currentSessionId = buildSessionId();
  chatTitle.textContent = "Hello, " + userName;
  updateMetrics();
  goTo(1);
});

nameInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") startBtn.click();
});

/* ─── Metrics ────────────────────────────────────────── */
function updateMetrics() {
  metricDocs.textContent    = documentCount;
  metricQueries.textContent = queryCount;
  metricSession.textContent = currentSessionId ? currentSessionId.slice(0, 8) + "…" : "—";
}

/* ─── Page 2: File upload ────────────────────────────── */
fileInput.addEventListener("change", () => {
  const selectedFiles = Array.from(fileInput.files || []);
  if (!selectedFiles.length) return;

  const allowedTypes = ["application/pdf", "text/csv", "application/vnd.ms-excel"];
  const invalid = selectedFiles.filter((f) => !allowedTypes.includes(f.type) && !/\.(pdf|csv)$/i.test(f.name));
  if (invalid.length) {
    setUploadStatus("Only PDF and CSV files are supported.", "err");
    return;
  }

  const existingPdfCount = fileStatuses.filter((entry) => /\.pdf$/i.test(entry.name)).length;
  const existingCsvCount = fileStatuses.filter((entry) => /\.csv$/i.test(entry.name)).length;
  const newPdfCount = selectedFiles.filter((f) => /\.pdf$/i.test(f.name)).length;
  const newCsvCount = selectedFiles.filter((f) => /\.csv$/i.test(f.name)).length;
  const totalPdfCount = existingPdfCount + newPdfCount;
  const totalCsvCount = existingCsvCount + newCsvCount;
  const totalCount = fileStatuses.length + selectedFiles.length;

  if (totalCount > 3) {
    setUploadStatus("Maximum 3 files total: up to 2 PDFs and 1 CSV.", "err");
    return;
  }
  if (totalCsvCount > 1) {
    setUploadStatus("Only one CSV file is allowed across all uploads.", "err");
    return;
  }
  if (totalPdfCount > 2) {
    setUploadStatus("Only up to two PDF files are allowed across all uploads.", "err");
    return;
  }

  const existingNames = new Set(fileStatuses.map((entry) => entry.name));
  const newFiles = selectedFiles.filter((f) => !existingNames.has(f.name));
  if (!newFiles.length) {
    setUploadStatus("All selected files are already processed.", "err");
    return;
  }

  fileStatuses = fileStatuses.concat(newFiles.map((file) => ({ name: file.name, status: "pending" })));
  renderFileStatusList();
  fileName = fileStatuses.map((f) => f.name).join(", ");
  uploadBox.classList.add("has-file");
  uploadAndProcess(newFiles);
});

async function uploadAndProcess(files) {
  allFilesProcessed = false;
  fileStatuses = files.map((file) => ({ name: file.name, status: "pending" }));
  renderFileStatusList();
  setProcessingLock(true);
  setUploadStatus("Processing files…", "");
  progressFill.style.width = "0%";

  const stepWidth = 100 / files.length;

  for (let index = 0; index < files.length; index += 1) {
    const file = files[index];
    updateFileStatus(file.name, "processing");
    setUploadStatus(`Processing ${file.name} (${index + 1}/${files.length})…`, "");

    const formData = new FormData();
    formData.append("files", file);
    if (currentSessionId) {
      formData.append("session_id", currentSessionId);
    }

    try {
      const response = await fetch("/api/upload", { method: "POST", body: formData });
      if (!response.ok) {
        const err = await response.json().catch(() => ({}));
        updateFileStatus(file.name, "err");
        setUploadStatus(err.detail || `Upload failed for ${file.name}.`, "err");
        resetProgress();
        setProcessingLock(false);
        return;
      }

      const data = await response.json();
      if (data.session_id) currentSessionId = data.session_id;
      updateFileStatus(file.name, "done");
      animateProgress(index * stepWidth, (index + 1) * stepWidth, 400);
    } catch (error) {
      updateFileStatus(file.name, "err");
      setUploadStatus(`Upload failed for ${file.name}.`, "err");
      resetProgress();
      setProcessingLock(false);
      return;
    }
  }

  documentCount += files.length;
  allFilesProcessed = true;
  setProcessingLock(false);
  setUploadStatus("Ready", "ok");
  chatDoc.textContent = fileName;
  chatBadge.textContent = "Ready";
  updateMetrics();
}

function renderFileStatusList() {
  if (!fileStatusList) return;
  fileStatusList.innerHTML = fileStatuses
    .map(
      (file) =>
        `<div class="file-status-item ${file.status}"><span>${file.name}</span><span class="status-text">${file.status === "pending" ? "Pending" : file.status === "processing" ? "Processing…" : file.status === "done" ? "Done" : "Error"}</span></div>`
    )
    .join("");
  fileStatusList.classList.toggle("hidden", fileStatuses.length === 0);
}

function updateFileStatus(name, status) {
  const file = fileStatuses.find((entry) => entry.name === name);
  if (file) {
    file.status = status;
  }
  renderFileStatusList();
}

function setProcessingLock(active) {
  uploadInProgress = active;
  nextToChat.disabled = active || !allFilesProcessed;
  fileInput.disabled = active;
  uploadBox.classList.toggle("disabled", active);
  if (chatInputRow) {
    chatInputRow.classList.toggle("hidden", !allFilesProcessed);
  }
  questionInput.disabled = !allFilesProcessed;
  askButton.disabled = !allFilesProcessed;
  if (!allFilesProcessed) {
    questionInput.placeholder = "Upload and process files first to ask a question...";
  } else {
    questionInput.placeholder = "Ask about your document...";
  }
}

function setUploadStatus(msg, tone) {
  uploadStatus.textContent  = msg;
  uploadStatus.className    = "upload-status" + (tone ? " " + tone : "");
}

function animateProgress(from, to, duration) {
  const start = performance.now();
  const tick  = (now) => {
    const p = Math.min((now - start) / duration, 1);
    progressFill.style.width = (from + (to - from) * p) + "%";
    if (p < 1) requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function resetProgress() {
  progressFill.style.width = "0%";
  uploadBox.classList.remove("has-file");
}

nextToChat.addEventListener("click", () => {
  if (!fileName) {
    setUploadStatus("Please select a file first.", "err");
    return;
  }
  if (!currentSessionId) {
    setUploadStatus("Session error — go back and re-enter your name.", "err");
    return;
  }
  if (!allFilesProcessed) {
    setUploadStatus("Please finish processing files before going to chat.", "err");
    return;
  }
  goTo(2);
});

backToHero.addEventListener("click",   () => goTo(0));
backToUpload.addEventListener("click", () => goTo(1));

/* ─── Evidence tabs ──────────────────────────────────── */
const etabs  = document.querySelectorAll(".etab");
const epanes = document.querySelectorAll(".epane");

function switchTab(tabName) {
  etabs.forEach((t)  => t.classList.toggle("active", t.dataset.tab === tabName));
  epanes.forEach((p) => p.classList.toggle("active", p.id === "pane-" + tabName));
}

function setTabBadge(tabName, count) {
  const tab = document.querySelector(`.etab[data-tab="${tabName}"]`);
  if (!tab) return;
  let badge = tab.querySelector(".tab-count");
  if (count > 0) {
    if (!badge) { badge = document.createElement("span"); badge.className = "tab-count"; tab.appendChild(badge); }
    badge.textContent = count;
  } else {
    if (badge) badge.remove();
  }
}

etabs.forEach((tab) => {
  tab.addEventListener("click", () => switchTab(tab.dataset.tab));
});

/* ─── Page 3: Chat ───────────────────────────────────── */
questionInput.addEventListener("input", () => {
  questionInput.style.height = "auto";
  questionInput.style.height = Math.min(questionInput.scrollHeight, 140) + "px";
});

questionInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) { e.preventDefault(); askButton.click(); }
});

askButton.addEventListener("click", () => {
  const query = questionInput.value.trim();
  if (!query) return;
  if (!currentSessionId) { setChatStatus("No active session — upload a document first.", "err"); return; }
  if (isStreaming) return;
  sendQuery(query);
});

function sendQuery(query) {
  questionInput.value = "";
  questionInput.style.height = "auto";

  appendMessage("user", query);
  const typingEl = appendTyping();

  isStreaming             = true;
  askButton.disabled      = true;
  chatBadge.textContent   = "Thinking…";
  chatBadge.classList.add("busy");
  setChatStatus("Connecting…", "");
  clearEvidencePanel();

  queryCount += 1;
  updateMetrics();

  if (currentEventSource) currentEventSource.close();

  const url = `/api/chat?session_id=${currentSessionId}&query=${encodeURIComponent(query)}`;
  currentEventSource = new EventSource(url);

  let aiBubble = null;

  currentEventSource.addEventListener("token", (e) => {
    if (!aiBubble) {
      typingEl.remove();
      aiBubble = appendMessage("ai", "");
    }
    aiBubble.textContent += e.data;
    scrollMsgsToBottom();
    setChatStatus("Streaming…", "");
  });

  currentEventSource.addEventListener("metadata", (e) => {
    try {
      const meta = JSON.parse(e.data);
      populateEvidencePanel(meta);
    } catch {
      retrievedContext.textContent = "Unable to parse retrieved context.";
    }
  });

  currentEventSource.addEventListener("done", () => {
    finishStreaming(typingEl, aiBubble);
    setChatStatus("Answer complete — inspect results in the panel →", "ok");
  });

  currentEventSource.onerror = () => {
    finishStreaming(typingEl, aiBubble);
    if (!aiBubble) { typingEl.remove(); appendMessage("ai", "Connection error — please try again."); }
    setChatStatus("Connection error while streaming response.", "err");
  };
}

function finishStreaming(typingEl, aiBubble) {
  if (typingEl && typingEl.parentNode) typingEl.remove();
  isStreaming           = false;
  askButton.disabled    = false;
  chatBadge.textContent = "Ready";
  chatBadge.classList.remove("busy");
  if (currentEventSource) { currentEventSource.close(); currentEventSource = null; }
  scrollMsgsToBottom();
}

/* ─── Message helpers ────────────────────────────────── */
function appendMessage(role, text) {
  const wrap   = document.createElement("div");
  wrap.className = "msg " + role;
  const avatar = document.createElement("div");
  avatar.className = "msg-avatar " + (role === "ai" ? "ai" : "usr");
  avatar.textContent = role === "ai" ? "DL" : (userName ? userName[0].toUpperCase() : "U");
  const bubble = document.createElement("div");
  bubble.className   = "msg-bubble";
  bubble.textContent = text;
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMsgs.appendChild(wrap);
  scrollMsgsToBottom();
  return bubble;
}

function appendTyping() {
  const wrap   = document.createElement("div");
  wrap.className = "msg ai";
  const avatar = document.createElement("div");
  avatar.className   = "msg-avatar ai";
  avatar.textContent = "DL";
  const bubble = document.createElement("div");
  bubble.className  = "msg-bubble";
  bubble.innerHTML  = '<div class="typing-dots"><span></span><span></span><span></span></div>';
  wrap.appendChild(avatar);
  wrap.appendChild(bubble);
  chatMsgs.appendChild(wrap);
  scrollMsgsToBottom();
  return wrap;
}

function scrollMsgsToBottom() {
  chatMsgs.scrollTop = chatMsgs.scrollHeight;
}

/* ─── Status bar ─────────────────────────────────────── */
function setChatStatus(msg, tone) {
  chatStatusBar.textContent = msg;
  chatStatusBar.className   = "chat-status-bar" + (tone ? " " + tone : "");
  chatStatusBar.classList.remove("hidden");
}

/* ─── Evidence panel ─────────────────────────────────── */
function clearEvidencePanel() {
  retrievedContext.textContent    = "Retrieving context…";
  retrievedImages.innerHTML       = '<span class="no-content-msg">Looking for images…</span>';
  retrievedGeneratedImages.innerHTML = '<div class="no-plots-state"><svg viewBox="0 0 48 48" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 34 16 20 24 28 32 16 42 24"/><path d="M6 42h36"/></svg><span>Generating plots…</span></div>';
  if (plotsCount) plotsCount.textContent = "";
  retrievedCode.textContent       = "Waiting for code…";
  retrievedReasoning.textContent  = "Waiting for reasoning…";
  retrievedOutput.textContent     = "Waiting for output…";

  // Reset dashboard
  dashboardIframe.classList.add("hidden");
  noDashboardState.classList.remove("hidden");
  dashboardIframe.src = "";
  currentDashboardPath = null;

  // Clear tab badges
  ["generated", "dashboard"].forEach((t) => setTabBadge(t, 0));

  switchTab("context");
}

function populateEvidencePanel(meta) {
  // ── Context
  retrievedContext.textContent = meta.retrieved_context || meta.context_used || "No text context retrieved.";

  // ── RAG images
  retrievedImages.innerHTML = "";
  if (meta.image_paths && meta.image_paths.length) {
    meta.image_paths.forEach((src) => {
      const img = document.createElement("img");
      img.src = normalisePath(src);
      img.alt = "Retrieved document image";
      retrievedImages.appendChild(img);
    });
  } else {
    retrievedImages.innerHTML = '<span class="no-content-msg">No images retrieved for this query.</span>';
  }

  // ── Generated plots
  retrievedGeneratedImages.innerHTML = "";
  const plots = meta.plot_paths || [];
  if (plots.length) {
    plots.forEach((rawSrc, i) => {
      const url = normalisePath(rawSrc);

      const item  = document.createElement("div");
      item.className = "plot-item";

      const img   = document.createElement("img");
      img.src     = url;
      img.alt     = "Generated plot " + (i + 1);
      img.onerror = () => { setTimeout(() => { img.src = url + "?t=" + Date.now(); }, 800); };

      const lbl   = document.createElement("div");
      lbl.className = "plot-item-label";
      lbl.textContent = "plot_" + (i + 1) + ".png";

      const openBtn = document.createElement("button");
      openBtn.className = "plot-open-btn";
      openBtn.title     = "Open full size";
      openBtn.innerHTML = '<svg viewBox="0 0 16 16" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M7 3H3a1 1 0 00-1 1v9a1 1 0 001 1h9a1 1 0 001-1V9"/><path d="M10 2h4v4"/><line x1="14" y1="2" x2="7" y2="9"/></svg>';
      openBtn.addEventListener("click", () => window.open(url, "_blank"));

      item.appendChild(img);
      item.appendChild(lbl);
      item.appendChild(openBtn);
      retrievedGeneratedImages.appendChild(item);
    });

    if (plotsCount) plotsCount.textContent = plots.length + " plot" + (plots.length > 1 ? "s" : "");
    setTabBadge("generated", plots.length);
  } else {
    retrievedGeneratedImages.innerHTML = '<div class="no-plots-state"><svg viewBox="0 0 48 48" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="6 34 16 20 24 28 32 16 42 24"/><path d="M6 42h36"/></svg><span>No plots generated for this query</span></div>';
    if (plotsCount) plotsCount.textContent = "";
    setTabBadge("generated", 0);
  }

  // ── Generated slides
  retrievedSlides.innerHTML = "";
  const slides = meta.slide_paths || [];
  if (slides.length) {
    slides.forEach((rawSrc, i) => {
      const url = normalisePath(rawSrc);
      const item = document.createElement("div");
      item.className = "slide-item";

      const title = document.createElement("div");
      title.className = "slide-item-title";
      title.textContent = `Slide deck ${i + 1}`;

      const link = document.createElement("a");
      link.href = url;
      link.target = "_blank";
      link.rel = "noreferrer";
      link.textContent = url.replace("/temp/", "");
      link.className = "slide-item-link";

      item.appendChild(title);
      item.appendChild(link);
      retrievedSlides.appendChild(item);
    });

    if (slidesCount) slidesCount.textContent = slides.length + " file" + (slides.length > 1 ? "s" : "");
    setTabBadge("slides", slides.length);
  } else {
    retrievedSlides.innerHTML = '<div class="no-plots-state"><svg viewBox="0 0 48 48" fill="none" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><rect x="10" y="10" width="28" height="28" rx="4"/><path d="M14 16h20"/><path d="M14 24h20"/><path d="M14 32h12"/></svg><span>No slides generated yet</span></div>';
    if (slidesCount) slidesCount.textContent = "";
    setTabBadge("slides", 0);
  }

  // ── Dashboard
  // Backend returns: temp/{session}/outputs/dashboards/dashboard.html
  const dashRaw = meta.dashboard_url || meta.dashboard_path || meta.dashboard || null;
  if (dashRaw) {
    const dashUrl = normalisePath(dashRaw);
    currentDashboardPath = dashUrl;
    dashboardIframe.src  = dashUrl;
    dashboardIframe.classList.remove("hidden");
    noDashboardState.classList.add("hidden");
    setTabBadge("dashboard", 1);
  } else {
    dashboardIframe.classList.add("hidden");
    noDashboardState.classList.remove("hidden");
    setTabBadge("dashboard", 0);
  }

  // ── Code
  retrievedCode.textContent      = meta.code || "No code was executed for this query.";

  // ── Reasoning
  retrievedReasoning.textContent = meta.reasoning || "No reasoning was returned for this query.";

  // ── Output
  retrievedOutput.textContent    = meta.python_output || meta.insight || "No output was returned for this query.";

  // ── Auto-switch to most relevant tab
  if (dashRaw)                                          switchTab("dashboard");
  else if (slides.length)                              switchTab("slides");
  else if (plots.length)                               switchTab("generated");
  else if (meta.code)                                  switchTab("code");
  else if (meta.python_output || meta.insight)         switchTab("output");
  else if (meta.retrieved_context || meta.context_used) switchTab("context");
}

/* ─── Dashboard buttons ──────────────────────────────── */
dashOpenBtn.addEventListener("click", () => {
  if (currentDashboardPath) window.open(currentDashboardPath, "_blank");
});

dashRefreshBtn.addEventListener("click", () => {
  if (currentDashboardPath) {
    dashboardIframe.src = currentDashboardPath + "?t=" + Date.now();
  }
});

/* ─── Collapse evidence panel ────────────────────────── */
collapseEvidence.addEventListener("click", () => {
  evidencePanel.classList.toggle("collapsed");
});

/* ─── Init ───────────────────────────────────────────── */
goTo(0);
updateMetrics();