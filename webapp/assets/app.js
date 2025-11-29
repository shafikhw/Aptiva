const dom = {
  statusBar: document.getElementById("statusBar"),
  userLabel: document.getElementById("userLabel"),
  userMeta: document.getElementById("userMeta"),
  systemSelect: document.getElementById("systemSelect"),
  carryToggle: document.getElementById("carryToggle"),
  sharePrefToggle: document.getElementById("sharePrefToggle"),
  conversationList: document.getElementById("conversationList"),
  newChatBtn: document.getElementById("newChatBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  chatFeed: document.getElementById("chatFeed"),
  chatTitle: document.getElementById("chatTitle"),
  chatSubtitle: document.getElementById("chatSubtitle"),
  messageForm: document.getElementById("messageForm"),
  messageInput: document.getElementById("messageInput"),
  sendBtn: document.getElementById("sendBtn"),
  stopStreamBtn: document.getElementById("stopStreamBtn"),
  guestBtn: document.getElementById("guestBtn"),
  authToggleBtn: document.getElementById("authToggleBtn"),
  authModal: document.getElementById("authModal"),
  closeAuthModal: document.getElementById("closeAuthModal"),
  themeToggleBtn: document.getElementById("themeToggleBtn"),
  authTabs: document.querySelectorAll(".auth-tab"),
  loginForm: document.getElementById("loginForm"),
  signupForm: document.getElementById("signupForm"),
  logoutBtn: document.getElementById("logoutBtn"),
  forgotForm: document.getElementById("forgotForm"),
  forgotPasswordLink: document.getElementById("forgotPasswordLink"),
  backToLoginBtn: document.getElementById("backToLoginBtn"),
  leaseList: document.getElementById("leaseList"),
  refreshLeasesBtn: document.getElementById("refreshLeasesBtn"),
  personaSelect: document.getElementById("personaSelect"),
  scrapeNotice: document.getElementById("scrapeNotice"),
  scrapeMessage: document.getElementById("scrapeMessage"),
  scrapeToggle: document.getElementById("scrapeToggle"),
  scrapeHide: document.getElementById("scrapeHide"),
  scrapeDetails: document.getElementById("scrapeDetails"),
};

const DEFAULT_PERSONA_MODE = "auto";
const PERSONA_OPTIONS = [
  { value: "auto", label: "Auto" },
  { value: "naturalist", label: "Neighborhood Naturalist" },
  { value: "data", label: "Data Whisperer" },
  { value: "deal", label: "Deal Navigator" },
];

const state = {
  token: localStorage.getItem("aptiva_token") || "",
  user: null,
  system: "system1",
  personaMode: DEFAULT_PERSONA_MODE,
  conversations: [],
  conversationMap: new Map(),
  currentConversationId: null,
  placeholderConversationId: null,
  creatingConversationPromise: null,
  streamAbort: null,
  liveAssistantBubble: null,
  loadingConversationId: null,
  authView: "login",
  leaseDrafts: [],
  scrapeExpanded: false,
  demoMode: false,
  pinnedStatus: "",
  pinnedVariant: "warning",
};

const THEME_STORAGE_KEY = "aptiva_theme";
const COLOR_SCHEME_QUERY = window.matchMedia ? window.matchMedia("(prefers-color-scheme: light)") : null;

const JSON_HEADERS = { "Content-Type": "application/json" };

function resolveThemePreference(preference) {
  if (preference === "light" || preference === "dark") {
    return preference;
  }
  if (COLOR_SCHEME_QUERY && COLOR_SCHEME_QUERY.matches) {
    return "light";
  }
  return "dark";
}

function applyTheme(theme) {
  const normalized = theme === "light" ? "light" : "dark";
  const root = document.documentElement;
  root.dataset.theme = normalized;
  if (dom.themeToggleBtn) {
    const nextLabel = normalized === "light" ? "Dark mode" : "Light mode";
    const nextTheme = normalized === "light" ? "dark" : "light";
    dom.themeToggleBtn.textContent = `${nextTheme.charAt(0).toUpperCase()}${nextTheme.slice(1)} mode`;
    dom.themeToggleBtn.setAttribute("aria-label", `Switch to ${nextTheme} mode`);
    dom.themeToggleBtn.setAttribute("aria-pressed", normalized === "light" ? "true" : "false");
  }
  return normalized;
}

function setThemePreference(theme) {
  const normalized = applyTheme(theme);
  localStorage.setItem(THEME_STORAGE_KEY, normalized);
}

function attachSystemThemeListener() {
  if (!COLOR_SCHEME_QUERY) return;
  const handler = (event) => {
    if (localStorage.getItem(THEME_STORAGE_KEY)) return;
    applyTheme(event.matches ? "light" : "dark");
  };
  if (typeof COLOR_SCHEME_QUERY.addEventListener === "function") {
    COLOR_SCHEME_QUERY.addEventListener("change", handler);
  } else if (typeof COLOR_SCHEME_QUERY.addListener === "function") {
    COLOR_SCHEME_QUERY.addListener(handler);
  }
}

function initThemeControls() {
  const stored = localStorage.getItem(THEME_STORAGE_KEY);
  applyTheme(resolveThemePreference(stored));
  attachSystemThemeListener();
}

function normalizePersonaMode(mode) {
  const key = (mode || "").toLowerCase();
  return PERSONA_OPTIONS.some((p) => p.value === key) ? key : DEFAULT_PERSONA_MODE;
}

function personaLabel(mode) {
  const normalized = normalizePersonaMode(mode);
  return PERSONA_OPTIONS.find((p) => p.value === normalized)?.label || "Auto";
}

function setPersonaMode(mode, { persistConversation } = {}) {
  const normalized = normalizePersonaMode(mode);
  state.personaMode = normalized;
  if (dom.personaSelect) {
    dom.personaSelect.value = normalized;
  }
  if (persistConversation && state.currentConversationId) {
    const convo = state.conversationMap.get(state.currentConversationId);
    if (convo) {
      convo.persona_mode = normalized;
      const idx = state.conversations.findIndex((c) => c.id === convo.id);
      if (idx !== -1) {
        state.conversations[idx] = { ...state.conversations[idx], persona_mode: normalized };
      }
    }
  }
  return normalized;
}

function setScrapeNotice(message) {
  if (!dom.scrapeNotice) return;
  if (message) {
    dom.scrapeMessage.textContent = message;
    dom.scrapeNotice.classList.remove("hidden");
  } else {
    dom.scrapeNotice.classList.add("hidden");
    state.scrapeExpanded = false;
    if (dom.scrapeDetails) {
      dom.scrapeDetails.classList.add("hidden");
    }
    if (dom.scrapeToggle) {
      dom.scrapeToggle.textContent = "Show details";
    }
  }
}

function toggleScrapeDetails() {
  state.scrapeExpanded = !state.scrapeExpanded;
  if (dom.scrapeDetails) {
    dom.scrapeDetails.classList.toggle("hidden", !state.scrapeExpanded);
  }
  if (dom.scrapeToggle) {
    dom.scrapeToggle.textContent = state.scrapeExpanded ? "Hide details" : "Show details";
  }
}

function showInlineScrape(bubble) {
  if (!bubble) return null;
  const body = bubble.querySelector(".message-body");
  if (!body) return null;
  const wrapper = document.createElement("div");
  wrapper.className = "inline-scrape";
  wrapper.innerHTML = `
    <div class="scrape-dot small" aria-hidden="true"></div>
    <div>
      <strong>Retrieving listings…</strong>
      <small>Streaming results in a moment.</small>
    </div>
  `;
  body.appendChild(wrapper);
  return wrapper;
}

function stripStreamingPrefix(text = "") {
  const cleaned = text.replace(/^[\uFEFF]/, "");

  const fenceMatch = cleaned.match(/^```(?:json)?\s*\n([\s\S]*?)```[\s\r\n]*/i);
  if (fenceMatch) {
    const afterFence = cleaned.slice(fenceMatch[0].length).trimStart();
    return afterFence;
  }

  const trimmed = cleaned.trimStart();
  if (!trimmed) return "";
  const firstChar = trimmed[0];

  if (firstChar === "{" || firstChar === "[") {
    const endIdx = findClosingBracketIndex(trimmed);
    if (endIdx === -1) return "";
    const candidate = trimmed.slice(0, endIdx + 1);
    try {
      const parsed = JSON.parse(candidate);
      if (parsed && typeof parsed === "object" && !Array.isArray(parsed)) {
        const remainder = trimmed.slice(endIdx + 1).trimStart();
        return remainder;
      }
    } catch {
      /* fall through */
    }
  }

  return cleaned;
}

function buildPersonaSelect() {
  if (!dom.personaSelect) return;
  dom.personaSelect.innerHTML = PERSONA_OPTIONS.map(
    (opt) => `<option value="${opt.value}">${opt.label}</option>`,
  ).join("");
  dom.personaSelect.value = normalizePersonaMode(state.personaMode);
}

function escapeHtml(value = "") {
  return value
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderMarkdown(text = "") {
  if (!text) return "";
  const escaped = escapeHtml(text).replace(/\r\n/g, "\n");
  const tokens = escaped.split("\n");
  const output = [];
  let listBuffer = [];
  let orderedBuffer = [];

  const flushLists = () => {
    if (listBuffer.length) {
      output.push(`<ul>${listBuffer.join("")}</ul>`);
      listBuffer = [];
    }
    if (orderedBuffer.length) {
      const start = orderedBuffer[0].index || 1;
      const startAttr = start !== 1 ? ` start="${start}"` : "";
      const items = orderedBuffer.map((item) => `<li>${item.text}</li>`).join("");
      output.push(`<ol${startAttr}>${items}</ol>`);
      orderedBuffer = [];
    }
  };

  for (const line of tokens) {
    const bulletMatch = line.match(/^\s*-\s+(.*)/);
    const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
    if (bulletMatch) {
      if (orderedBuffer.length) flushLists();
      listBuffer.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }
    if (orderedMatch) {
      if (listBuffer.length) flushLists();
      orderedBuffer.push({ index: Number(orderedMatch[1]) || 1, text: orderedMatch[2] });
      continue;
    }
    flushLists();
    if (line.trim() === "") {
      output.push("<p></p>");
    } else {
      output.push(`<p>${line}</p>`);
    }
  }
  flushLists();

  let html = output.join("");
  html = html.replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");
  html = html.replace(/(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)/g, "<em>$1</em>");
  html = html.replace(/`([^`]+)`/g, "<code>$1</code>");
  const imgRegex = /!\[([^\]]*)\]\((https?:\/\/[^\s)]+)\)|(https?:\/\/[^\s<>"']+\.(?:png|jpe?g|gif))/gi;
  html = html.replace(imgRegex, (match, alt, mdUrl, bareUrl) => {
    const url = mdUrl || bareUrl;
    const clean = url.replace(/&amp;/g, "&");
    const altText = alt !== undefined ? alt : "Listing image";
    return `<img src="${clean}" alt="${altText}" class="inline-photo" />`;
  });
  html = html.replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/<a href="[^"]+\.(?:png|jpe?g|gif)[^"]*"[^>]*>.*?<\/a>/gi, "");
  return html;
}

function findClosingBracketIndex(text) {
  const stack = [];
  let inString = false;
  let escapeNext = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (escapeNext) {
      escapeNext = false;
      continue;
    }
    if (char === "\\") {
      escapeNext = true;
      continue;
    }
    if (char === '"') {
      inString = !inString;
      continue;
    }
    if (inString) continue;
    if (char === "{" || char === "[") {
      stack.push(char);
    } else if (char === "}" || char === "]") {
      if (!stack.length) return -1;
      const open = stack.pop();
      if ((open === "{" && char !== "}") || (open === "[" && char !== "]")) {
        return -1;
      }
      if (!stack.length) {
        return i;
      }
    }
  }
  return -1;
}

function setStatus(message, variant = "info") {
  if (!dom.statusBar) return;
  const combined = state.pinnedStatus
    ? [state.pinnedStatus, message].filter(Boolean).join(" · ")
    : message;
  const effectiveVariant = state.pinnedStatus ? state.pinnedVariant || variant : variant;
  dom.statusBar.textContent = combined || "";
  dom.statusBar.dataset.variant = effectiveVariant;
}

function pinStatus(message, variant = "warning") {
  state.pinnedStatus = message;
  state.pinnedVariant = variant;
  setStatus("");
}

async function checkHealth() {
  try {
    const res = await api("/api/health");
    const mode = res.details?.supabase?.mode || "supabase";
    if (mode === "memory") {
      state.demoMode = true;
      pinStatus("Supabase unavailable – running in demo (in-memory) mode", "warning");
    } else {
      state.demoMode = false;
      state.pinnedStatus = "";
      state.pinnedVariant = "info";
    }
  } catch (err) {
    console.error(err);
    setStatus("Health check failed.", "error");
  }
}

async function api(path, { method = "GET", json, headers, signal } = {}) {
  const init = { method, headers: { ...(headers || {}) }, signal };
  if (json !== undefined) {
    init.body = JSON.stringify(json);
    init.headers = { ...init.headers, ...JSON_HEADERS };
  }
  if (state.token) {
    init.headers.Authorization = `Bearer ${state.token}`;
  }
  const res = await fetch(path, init);
  if (!res.ok) {
    const text = await res.text();
    throw new Error(text || res.statusText);
  }
  const contentType = res.headers.get("content-type") || "";
  if (contentType.includes("application/json")) {
    return res.json();
  }
  return res.text();
}

function setAuth(token, user) {
  state.token = token;
  state.user = user;
  localStorage.setItem("aptiva_token", token);
  updateAuthUI();
}

async function restoreSession() {
  if (state.token) {
    try {
      const data = await api("/api/auth/me");
      setAuth(state.token, data.user);
      return true;
    } catch (err) {
      console.warn("Session restore failed:", err);
      localStorage.removeItem("aptiva_token");
    }
  }
  return false;
}

async function startGuestSession() {
  setStatus("Creating guest session…");
  const data = await api("/api/auth/guest", { method: "POST" });
  setAuth(data.token, data.user);
  setStatus("Guest session ready.");
  await loadConversations();
}

function updateAuthUI() {
  if (!state.user) return;
  dom.userLabel.textContent = state.user.username || "Guest";
  dom.userMeta.textContent = `${state.user.email} · ${state.user.auth_provider || "guest"}`;
  const share = (state.user.preferences && state.user.preferences.share_preferences) ?? true;
  syncSharePreference(share, { skipSave: true });
}

function toggleModal(show) {
  dom.authModal.classList.toggle("hidden", !show);
  dom.authModal.setAttribute("aria-hidden", String(!show));
  if (show) {
    setAuthView("login");
  }
}

function setAuthView(view) {
  state.authView = view;
  const authViews = {
    login: dom.loginForm,
    signup: dom.signupForm,
    forgot: dom.forgotForm,
  };
  Object.entries(authViews).forEach(([key, element]) => {
    if (element) {
      element.classList.toggle("hidden", key !== view);
    }
  });
  dom.authTabs.forEach((tab) => tab.classList.toggle("active", tab.dataset.mode === view));
}

function syncSharePreference(enabled, { skipSave } = {}) {
  dom.carryToggle.checked = enabled;
  dom.sharePrefToggle.checked = enabled;
  state.user = state.user || {};
  state.user.preferences = { ...(state.user.preferences || {}), share_preferences: enabled };
  if (!skipSave) {
    saveSharePreference(enabled).catch((err) => {
      console.error(err);
      setStatus("Failed to update preference.", "error");
    });
  }
}

async function saveSharePreference(enabled) {
  await api("/api/preferences", {
    method: "POST",
    json: { updates: { share_preferences: enabled } },
  });
  setStatus(`Shared memory ${enabled ? "enabled" : "disabled"} for future chats.`);
}

async function loadConversations() {
  if (!state.user) return;
  setStatus("Loading conversations…");
  const data = await api(`/api/chat/conversations?system=${state.system}`);
  const existingMap = new Map(state.conversationMap);
  const maintainActive =
    state.currentConversationId && existingMap.get(state.currentConversationId)?.messages?.length;
  state.conversations = data.conversations || [];
  const merged = new Map();
  state.conversations.forEach((convo) => {
    const cached = existingMap.get(convo.id);
    const mergedConvo = {
      ...(cached || {}),
      ...convo,
    };
    merged.set(convo.id, mergedConvo);
  });
  state.conversationMap = merged;
  state.conversationMap.forEach((value, key) => {
    if (key !== state.currentConversationId) {
      delete value.messages;
    }
  });
  state.conversations = state.conversations.map((convo) => {
    const { messages, ...rest } = convo;
    return rest;
  });
  if (state.placeholderConversationId) {
    const placeholder = existingMap.get(state.placeholderConversationId);
    if (placeholder) {
      state.conversations = [placeholder, ...state.conversations.filter((c) => c.id !== placeholder.id)];
      state.conversationMap.set(placeholder.id, placeholder);
    }
  }
  renderConversations();
  if (state.currentConversationId && !state.conversationMap.has(state.currentConversationId)) {
    state.currentConversationId = null;
  }
  if (!state.currentConversationId || !maintainActive) {
    state.currentConversationId = null;
    clearChat();
  }
  setStatus("Conversations synced.");
  await loadLeaseDrafts();
}

function renderConversations() {
  dom.conversationList.innerHTML = "";
  if (!state.conversations.length) {
    const empty = document.createElement("p");
    empty.className = "panel-subtle";
    empty.textContent = "No chats yet.";
    dom.conversationList.appendChild(empty);
    return;
  }
  state.conversations.forEach((convo) => {
    const item = document.createElement("div");
    item.className = "conversation-item";
    if (convo.id === state.currentConversationId) {
      item.classList.add("active");
    }
    item.dataset.id = convo.id;
    item.role = "option";
    const topic = (convo.topic && convo.topic.trim()) || "";
    const preview = (convo.preview && convo.preview.trim()) || "";
    const titleText = topic || preview || `Conversation ${convo.id.slice(0, 6)}`;
    const timestamp = formatTimestamp(convo.updated_at || convo.created_at);
    const previewMarkup = preview ? `<p class="conversation-preview">${escapeHtml(preview)}</p>` : "";
    item.innerHTML = `<strong>${escapeHtml(titleText)}</strong><small>${timestamp}</small>${previewMarkup}`;
    dom.conversationList.appendChild(item);
  });
}

function formatTimestamp(value) {
  if (!value) return "New";
  try {
    const date = new Date(value * 1000);
    return date.toLocaleString(undefined, { dateStyle: "short", timeStyle: "short" });
  } catch {
    return "Recent";
  }
}

async function loadConversationDetail(id) {
  if (!id || state.loadingConversationId === id) return;
  const cached = state.conversationMap.get(id);
  if (cached) {
    setPersonaMode(cached.persona_mode || state.personaMode);
  }
  if (cached?.messages?.length) {
    setChatHeading(cached);
    renderMessages(cached.messages);
  } else {
    dom.chatFeed.innerHTML = "";
    dom.chatTitle.textContent = "Loading conversation…";
    dom.chatSubtitle.textContent = "Fetching history.";
  }
  state.loadingConversationId = id;
  try {
    const data = await api(`/api/chat/conversations/${id}`);
    const convo = data.conversation;
    state.conversationMap.set(convo.id, convo);
    const idx = state.conversations.findIndex((c) => c.id === convo.id);
    if (idx !== -1) {
      state.conversations[idx] = convo;
    }
    renderConversations();
    setChatHeading(convo);
    renderMessages(convo.messages || []);
  } finally {
    state.loadingConversationId = null;
  }
}

function clearChat() {
  dom.chatFeed.innerHTML = "";
  dom.chatTitle.textContent = "Start chatting";
  dom.chatSubtitle.textContent = "Preferences sync automatically across chats.";
}

function setChatHeading(convo) {
  dom.chatTitle.textContent = `Chat · ${convo.system === "system2" ? "Lebanon" : "United States"}`;
  const persona = setPersonaMode(convo.persona_mode || state.personaMode);
  dom.chatSubtitle.textContent = `Persona: ${personaLabel(persona)} · Updated ${formatTimestamp(convo.updated_at)}`;
}

function renderMessages(messages) {
  dom.chatFeed.innerHTML = "";
  messages.forEach((msg) => appendMessageBubble(msg.role, msg.content));
  dom.chatFeed.scrollTop = dom.chatFeed.scrollHeight;
}

async function loadLeaseDrafts() {
  if (!state.user) return;
  try {
    const data = await api("/api/lease/drafts");
    state.leaseDrafts = data.drafts || [];
    renderLeaseDrafts();
  } catch (err) {
    console.error(err);
    setStatus("Unable to load lease drafts.", "error");
  }
}

function renderLeaseDrafts() {
  if (!dom.leaseList) return;
  dom.leaseList.innerHTML = "";
  if (!state.leaseDrafts.length) {
    const empty = document.createElement("p");
    empty.className = "panel-subtle";
    empty.textContent = "No drafts yet.";
    dom.leaseList.appendChild(empty);
    return;
  }
  state.leaseDrafts.forEach((draft) => {
    const card = document.createElement("div");
    card.className = "lease-card";
    const title = draft.title || "Lease draft";
    const created = formatTimestamp(draft.created_at);
    const summary = draft.summary || "Generated lease draft";
    card.innerHTML = `<strong>${title}</strong><small>${created}</small><p>${summary}</p>`;
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "ghost-btn";
    btn.dataset.draftId = draft.id;
    btn.textContent = "Download PDF";
    card.appendChild(btn);
    dom.leaseList.appendChild(card);
  });
}

function upsertLeaseDraft(draft) {
  if (!draft || !draft.id) return;
  const existing = state.leaseDrafts.findIndex((d) => d.id === draft.id);
  if (existing !== -1) {
    state.leaseDrafts[existing] = draft;
  } else {
    state.leaseDrafts = [draft, ...state.leaseDrafts];
  }
  renderLeaseDrafts();
}

function handleLeaseListClick(event) {
  const btn = event.target.closest("button[data-draft-id]");
  if (!btn) return;
  const id = btn.dataset.draftId;
  if (!id) return;
  downloadLeaseDraft(id);
}

async function downloadLeaseDraft(id) {
  try {
    const res = await fetch(`/api/lease/drafts/${id}/download`, {
      headers: state.token ? { Authorization: `Bearer ${state.token}` } : {},
    });
    if (!res.ok) {
      const text = await res.text();
      throw new Error(text || "Download failed");
    }
    const blob = await res.blob();
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `lease_${id}.pdf`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    console.error(err);
    setStatus("Unable to download lease draft.", "error");
  }
}

function appendMessageBubble(role, content) {
  const bubble = document.createElement("article");
  bubble.className = `message ${role}`;
  const meta = document.createElement("span");
  meta.className = "meta";
  meta.textContent = role === "user" ? "You" : "Aptiva";
  const body = document.createElement("div");
  body.className = "message-body";
  const fullContent = content || "";
  body.dataset.rawFull = fullContent;
  const cleanContent = stripStreamingPrefix(fullContent);
  body.dataset.raw = cleanContent;
  body.innerHTML = renderMarkdown(cleanContent);
  bubble.append(meta, body);
  dom.chatFeed.appendChild(bubble);
  dom.chatFeed.scrollTop = dom.chatFeed.scrollHeight;
  return bubble;
}

function ensureSharePreferenceBinding() {
  const sync = (checked) => syncSharePreference(checked);
  dom.carryToggle.addEventListener("change", (e) => sync(e.currentTarget.checked));
  dom.sharePrefToggle.addEventListener("change", (e) => sync(e.currentTarget.checked));
}

async function createConversation({ silent = false } = {}) {
  if (state.creatingConversationPromise) {
    return state.creatingConversationPromise;
  }
  const placeholderId = `temp-${Date.now()}`;
  insertPlaceholderConversation(placeholderId);
  dom.newChatBtn.disabled = true;
  const payload = {
    system: state.system,
    carry_preferences: dom.sharePrefToggle.checked,
    persona_mode: state.personaMode,
  };
  const promise = (async () => {
    try {
      const data = await api("/api/chat/conversations", { method: "POST", json: payload });
      replacePlaceholderConversation(placeholderId, data.conversation);
      if (!silent) {
        setStatus("New conversation ready.");
      }
      return data.conversation.id;
    } catch (err) {
      removePlaceholderConversation(placeholderId);
      throw err;
    } finally {
      dom.newChatBtn.disabled = false;
      state.creatingConversationPromise = null;
    }
  })();
  state.creatingConversationPromise = promise;
  return promise;
}

function insertPlaceholderConversation(id) {
  state.placeholderConversationId = id;
  const placeholder = {
    id,
    preview: "Starting new chat…",
    topic: "New chat",
    system: state.system,
    messages: [],
    persona_mode: normalizePersonaMode(state.personaMode),
  };
  setPersonaMode(placeholder.persona_mode);
  state.conversations = [placeholder, ...state.conversations];
  state.conversationMap.set(id, placeholder);
  state.currentConversationId = id;
  renderConversations();
  dom.chatFeed.innerHTML = "";
  dom.chatTitle.textContent = "Starting new chat…";
  dom.chatSubtitle.textContent = `Persona: ${personaLabel(
    placeholder.persona_mode,
  )} · Preparing workspace.`;
}

function replacePlaceholderConversation(tempId, conversation) {
  const idx = state.conversations.findIndex((c) => c.id === tempId);
  if (idx !== -1) {
    state.conversations[idx] = conversation;
  } else {
    state.conversations.unshift(conversation);
  }
  state.conversationMap.delete(tempId);
  state.conversationMap.set(conversation.id, conversation);
  state.currentConversationId = conversation.id;
  state.placeholderConversationId = null;
  renderConversations();
  setChatHeading(conversation);
  renderMessages(conversation.messages || []);
}

function removePlaceholderConversation(tempId) {
  state.conversations = state.conversations.filter((c) => c.id !== tempId);
  state.conversationMap.delete(tempId);
  if (state.currentConversationId === tempId) {
    state.currentConversationId = null;
  }
  renderConversations();
  clearChat();
}

async function handleSendMessage(event) {
  event.preventDefault();
  const text = dom.messageInput.value.trim();
  if (!text) return;
  try {
    dom.sendBtn.disabled = true;
    dom.stopStreamBtn.disabled = false;
    const conversationId = state.currentConversationId || (await createConversation({ silent: true }));
    appendMessageBubble("user", text);
    dom.messageInput.value = "";
    await streamChat({
      message: text,
      system: state.system,
      conversation_id: conversationId,
      persona_mode: state.personaMode,
      carry_preferences: dom.sharePrefToggle.checked,
    });
  } catch (err) {
    console.error(err);
    setStatus(err.message || "Failed to send message.", "error");
  } finally {
    dom.sendBtn.disabled = false;
    dom.messageInput.focus();
  }
}

function handleComposerKeydown(event) {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    if (dom.sendBtn.disabled) return;
    if (typeof dom.messageForm.requestSubmit === "function") {
      dom.messageForm.requestSubmit();
    } else {
      dom.messageForm.dispatchEvent(new Event("submit", { cancelable: true, bubbles: true }));
    }
  }
}

function handlePersonaChange(event) {
  const persona = setPersonaMode(event.target.value, { persistConversation: true });
  if (state.currentConversationId) {
    const active = state.conversationMap.get(state.currentConversationId);
    if (active) {
      setChatHeading({ ...active, persona_mode: persona });
    }
  }
}

async function streamChat(payload) {
  const controller = new AbortController();
  state.streamAbort = controller;
  const response = await fetch("/api/chat/stream", {
    method: "POST",
    headers: {
      ...(state.token ? { Authorization: `Bearer ${state.token}` } : {}),
      "Content-Type": "application/json",
    },
    body: JSON.stringify(payload),
    signal: controller.signal,
  });
  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || "Streaming request failed.");
  }
  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let buffer = "";
  let done = false;
  state.liveAssistantBubble = appendMessageBubble("assistant", "");
  let inlineScrape = null;
  const removeInlineScrape = () => {
    if (inlineScrape && inlineScrape.parentNode) {
      inlineScrape.parentNode.removeChild(inlineScrape);
    }
  };
  try {
    while (!done) {
      const { value, done: readerDone } = await reader.read();
      if (readerDone) break;
      buffer += decoder.decode(value, { stream: true });
      let idx;
      while ((idx = buffer.indexOf("\n\n")) !== -1) {
        const chunk = buffer.slice(0, idx);
        buffer = buffer.slice(idx + 2);
        if (!chunk.trim()) continue;
        const { event, data } = parseSseChunk(chunk);
        if (event === "status") {
          const status = safeJsonParse(data, data);
          if (status === "scrape" && !inlineScrape) {
            inlineScrape = showInlineScrape(state.liveAssistantBubble);
          }
          continue;
        }
        if (event === "token") {
          const text = safeJsonParse(data, "");
          const rendered = appendStreamToken(text || "");
          if (rendered) {
            removeInlineScrape();
          }
        } else if (event === "final") {
          removeInlineScrape();
          const result = safeJsonParse(data, null);
          if (result) {
            finalizeStream(result);
          }
          dom.stopStreamBtn.disabled = true;
          state.streamAbort = null;
          done = true;
          break;
        } else if (event === "error") {
          removeInlineScrape();
          const info = safeJsonParse(data, { error: "Streaming error." });
          throw new Error(info.error || "Streaming error.");
        }
      }
    }
  } finally {
    removeInlineScrape();
    state.streamAbort = null;
    dom.stopStreamBtn.disabled = true;
  }
}

function parseSseChunk(chunk) {
  let event = "message";
  let data = "";
  chunk.split("\n").forEach((line) => {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      data += line.slice(5).trim();
    }
  });
  return { event, data };
}

function appendStreamToken(token) {
  if (!state.liveAssistantBubble) return false;
  const body = state.liveAssistantBubble.querySelector(".message-body");
  const priorFull = body.dataset.rawFull || "";
  const nextFull = priorFull + token;
  body.dataset.rawFull = nextFull;
  const cleaned = stripStreamingPrefix(nextFull);
  body.dataset.raw = cleaned;
  body.innerHTML = renderMarkdown(cleaned);
  dom.chatFeed.scrollTop = dom.chatFeed.scrollHeight;
  return Boolean(cleaned);
}

function finalizeStream(result) {
  if (!result) return;
  const conversation = result.conversation;
  if (conversation) {
    state.conversationMap.set(conversation.id, conversation);
    const idx = state.conversations.findIndex((c) => c.id === conversation.id);
    if (idx !== -1) {
      state.conversations[idx] = conversation;
    } else {
      state.conversations.unshift(conversation);
    }
    state.currentConversationId = conversation.id;
    renderConversations();
    setChatHeading(conversation);
    renderMessages(conversation.messages || []);
  }
  if (result.lease_draft) {
    upsertLeaseDraft(result.lease_draft);
  }
  state.liveAssistantBubble = null;
}

function stopStreaming() {
  if (state.streamAbort) {
    state.streamAbort.abort();
    state.streamAbort = null;
    dom.stopStreamBtn.disabled = true;
    setStatus("Stream cancelled.", "warning");
  }
}

function handleConversationClick(event) {
  const item = event.target.closest(".conversation-item");
  if (!item) return;
  const id = item.dataset.id;
  if (!id) return;
  state.currentConversationId = id;
  renderConversations();
  loadConversationDetail(id).catch((err) => {
    console.error(err);
    setStatus("Failed to load conversation.", "error");
  });
}

async function handleLogin(event) {
  event.preventDefault();
  const formData = new FormData(dom.loginForm);
  try {
    setStatus("Signing in…");
    const data = await api("/api/auth/login", {
      method: "POST",
      json: {
        email: formData.get("email"),
        password: formData.get("password"),
      },
    });
    setAuth(data.token, data.user);
    toggleModal(false);
    await loadConversations();
    setStatus("Signed in successfully.");
  } catch (err) {
    setStatus(err.message || "Login failed.", "error");
  }
}

async function handleSignup(event) {
  event.preventDefault();
  const formData = new FormData(dom.signupForm);
  try {
    setStatus("Creating account…");
    const data = await api("/api/auth/register", {
      method: "POST",
      json: {
        email: formData.get("email"),
        username: formData.get("username"),
        first_name: formData.get("first_name"),
        last_name: formData.get("last_name"),
        password: formData.get("password"),
      },
    });
    setAuth(data.token, data.user);
    toggleModal(false);
    await loadConversations();
    setStatus("Account created.");
  } catch (err) {
    setStatus(err.message || "Signup failed.", "error");
  }
}

async function handleForgotPassword(event) {
  event.preventDefault();
  const formData = new FormData(dom.forgotForm);
  const email = formData.get("email");
  const password = formData.get("password");
  if (!email || !password) return;
  try {
    setStatus("Updating password…");
    await api("/api/auth/forgot-password", {
      method: "POST",
      json: { email, password },
    });
    setStatus("Password updated. Please sign in.");
    setAuthView("login");
  } catch (err) {
    setStatus("Unable to update password.", "error");
  }
}

async function handleLogout() {
  if (!state.token) return;
  try {
    await api("/api/auth/logout", { method: "POST" });
  } catch {
    /* ignore */
  }
  localStorage.removeItem("aptiva_token");
  state.token = "";
  state.user = null;
  setStatus("Signing out…");
  await startGuestSession();
}

function safeJsonParse(value, fallback) {
  try {
    return JSON.parse(value);
  } catch {
    return fallback;
  }
}

function bindEvents() {
  dom.themeToggleBtn?.addEventListener("click", () => {
    const current = document.documentElement.dataset.theme === "light" ? "light" : "dark";
    const next = current === "light" ? "dark" : "light";
    setThemePreference(next);
  });
  dom.systemSelect.addEventListener("change", async (event) => {
    state.system = event.target.value;
    state.currentConversationId = null;
    await loadConversations();
  });
  dom.newChatBtn.addEventListener("click", () => {
    createConversation().catch((err) => setStatus(err.message || "Unable to start chat.", "error"));
  });
  dom.refreshBtn.addEventListener("click", () => loadConversations());
  dom.messageForm.addEventListener("submit", handleSendMessage);
  dom.messageInput.addEventListener("keydown", handleComposerKeydown);
  dom.stopStreamBtn.addEventListener("click", stopStreaming);
  dom.conversationList.addEventListener("click", handleConversationClick);
  dom.personaSelect.addEventListener("change", handlePersonaChange);
  dom.scrapeToggle?.addEventListener("click", toggleScrapeDetails);
  dom.scrapeHide?.addEventListener("click", () => setScrapeNotice(""));
  dom.guestBtn.addEventListener("click", () => startGuestSession());
  dom.authToggleBtn.addEventListener("click", () => toggleModal(true));
  dom.closeAuthModal.addEventListener("click", () => toggleModal(false));
  dom.loginForm.addEventListener("submit", handleLogin);
  dom.signupForm.addEventListener("submit", handleSignup);
  dom.forgotForm.addEventListener("submit", handleForgotPassword);
  dom.logoutBtn.addEventListener("click", handleLogout);
  dom.authTabs.forEach((tab) => tab.addEventListener("click", () => setAuthView(tab.dataset.mode)));
  dom.authModal.addEventListener("click", (event) => {
    if (event.target === dom.authModal) toggleModal(false);
  });
  ensureSharePreferenceBinding();
  dom.forgotPasswordLink.addEventListener("click", () => setAuthView("forgot"));
  dom.backToLoginBtn.addEventListener("click", () => setAuthView("login"));
  dom.refreshLeasesBtn.addEventListener("click", () => loadLeaseDrafts());
  dom.leaseList.addEventListener("click", handleLeaseListClick);
}

async function init() {
  initThemeControls();
  buildPersonaSelect();
  setPersonaMode(state.personaMode);
  bindEvents();
  state.system = dom.systemSelect.value;
  await checkHealth();
  const authed = await restoreSession();
  if (!authed) {
    setStatus("Sign in or start a guest session to begin.");
    toggleModal(true);
    return;
  }
  await loadConversations();
  setStatus("Ready.");
}

init().catch((err) => {
  console.error(err);
  setStatus("Failed to initialize frontend.", "error");
});
