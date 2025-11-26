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
  authTabs: document.querySelectorAll(".auth-tab"),
  loginForm: document.getElementById("loginForm"),
  signupForm: document.getElementById("signupForm"),
  logoutBtn: document.getElementById("logoutBtn"),
  forgotForm: document.getElementById("forgotForm"),
  forgotPasswordLink: document.getElementById("forgotPasswordLink"),
  backToLoginBtn: document.getElementById("backToLoginBtn"),
  leaseList: document.getElementById("leaseList"),
  refreshLeasesBtn: document.getElementById("refreshLeasesBtn"),
};

const state = {
  token: localStorage.getItem("aptiva_token") || "",
  user: null,
  system: "system1",
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
};

const JSON_HEADERS = { "Content-Type": "application/json" };

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
      output.push(`<ol>${orderedBuffer.join("")}</ol>`);
      orderedBuffer = [];
    }
  };

  for (const line of tokens) {
    const bulletMatch = line.match(/^\s*-\s+(.*)/);
    const orderedMatch = line.match(/^\s*(\d+)\.\s+(.*)/);
    if (bulletMatch) {
      flushLists();
      listBuffer.push(`<li>${bulletMatch[1]}</li>`);
      continue;
    }
    if (orderedMatch) {
      flushLists();
      orderedBuffer.push(`<li>${orderedMatch[2]}</li>`);
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
  const imgRegex = /https?:\/\/[^\s<>"']+\.(?:png|jpe?g|gif)/gi;
  html = html.replace(imgRegex, (url) => {
    const clean = url.replace(/&amp;/g, "&");
    return `<img src="${clean}" alt="Listing image" class="inline-photo" />`;
  });
  html = html.replace(/\[(.+?)\]\((https?:\/\/[^\s)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  html = html.replace(/<a href="[^"]+\.(?:png|jpe?g|gif)[^"]*"[^>]*>.*?<\/a>/gi, "");
  return html;
}

function stripPreferencePrefix(text = "") {
  const trimmed = text.replace(/^[\uFEFF]/, "").replace(/^\s+/, "");
  if (!trimmed.startsWith("{")) {
    return text;
  }
  const endIdx = findClosingBraceIndex(trimmed);
  if (endIdx === -1) {
    return text;
  }
  try {
    const obj = JSON.parse(trimmed.slice(0, endIdx + 1));
    if (!obj || typeof obj !== "object" || !("preferences" in obj)) {
      return text;
    }
  } catch {
    return text;
  }
  const remainder = trimmed.slice(endIdx + 1).replace(/^\s+/, "");
  return remainder || "";
}

function findClosingBraceIndex(text) {
  let depth = 0;
  let inString = false;
  for (let i = 0; i < text.length; i += 1) {
    const char = text[i];
    if (char === '"' && text[i - 1] !== "\\") {
      inString = !inString;
    }
    if (inString) continue;
    if (char === "{") depth += 1;
    if (char === "}") {
      depth -= 1;
      if (depth === 0) {
        return i;
      }
    }
  }
  return -1;
}

function setStatus(message, variant = "info") {
  if (!dom.statusBar) return;
  dom.statusBar.textContent = message || "";
  dom.statusBar.dataset.variant = variant;
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
    const title = convo.preview || `Conversation ${convo.id.slice(0, 6)}`;
    item.innerHTML = `<strong>${title || "Untitled chat"}</strong><small>${formatTimestamp(
      convo.updated_at || convo.created_at,
    )}</small>`;
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
  const persona = convo.persona_mode || "auto";
  dom.chatSubtitle.textContent = `Persona: ${persona} · Updated ${formatTimestamp(convo.updated_at)}`;
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
  const rawContent = stripPreferencePrefix(content || "");
  body.dataset.raw = rawContent;
  body.innerHTML = renderMarkdown(rawContent);
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
    system: state.system,
    messages: [],
    persona_mode: "auto",
  };
  state.conversations = [placeholder, ...state.conversations];
  state.conversationMap.set(id, placeholder);
  state.currentConversationId = id;
  renderConversations();
  dom.chatFeed.innerHTML = "";
  dom.chatTitle.textContent = "Starting new chat…";
  dom.chatSubtitle.textContent = "Preparing workspace.";
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
      persona_mode: state.conversationMap.get(conversationId)?.persona_mode,
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
      if (event === "token") {
        const text = safeJsonParse(data, "");
        appendStreamToken(text);
      } else if (event === "final") {
        const result = safeJsonParse(data, null);
        if (result) {
          finalizeStream(result);
        }
        dom.stopStreamBtn.disabled = true;
        state.streamAbort = null;
        done = true;
        break;
      } else if (event === "error") {
        const info = safeJsonParse(data, { error: "Streaming error." });
        throw new Error(info.error || "Streaming error.");
      }
    }
  }
  state.streamAbort = null;
  dom.stopStreamBtn.disabled = true;
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
  if (!state.liveAssistantBubble) return;
  const body = state.liveAssistantBubble.querySelector(".message-body");
  const prior = body.dataset.raw || "";
  body.dataset.raw = stripPreferencePrefix(prior + token);
  body.innerHTML = renderMarkdown(body.dataset.raw);
  dom.chatFeed.scrollTop = dom.chatFeed.scrollHeight;
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
  dom.stopStreamBtn.addEventListener("click", stopStreaming);
  dom.conversationList.addEventListener("click", handleConversationClick);
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
  bindEvents();
  state.system = dom.systemSelect.value;
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
