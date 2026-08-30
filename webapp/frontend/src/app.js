// ===== AI Code Service — Frontend Logic =====

(() => {
  "use strict";

  // ----- State -----
  const state = {
    messages: [],          // {role, content}
    conversations: [],     // {id, title, messages}
    currentConvId: null,
    selectedModel: "",
    isStreaming: false,
    abortController: null,
  };

  // ----- DOM refs -----
  const $ = (s) => document.querySelector(s);
  const el = {
    sidebar: $("#sidebar"),
    modelSelect: $("#model-select"),
    headerModelSelect: $("#header-model-select"),
    modelBadge: $("#model-badge"),
    chatMessages: $("#chat-messages"),
    welcomeScreen: $("#welcome-screen"),
    chatInput: $("#chat-input"),
    btnSend: $("#btn-send"),
    btnStop: $("#btn-stop"),
    btnNewChat: $("#btn-new-chat"),
    btnToggleSidebar: $("#btn-toggle-sidebar"),
    chatHistory: $("#chat-history"),
    backendStatus: $("#backend-status"),
  };

  // ----- Marked config -----
  marked.setOptions({
    highlight: (code, lang) => {
      if (lang && hljs.getLanguage(lang)) {
        return hljs.highlight(code, { language: lang }).value;
      }
      return hljs.highlightAuto(code).value;
    },
    breaks: true,
  });

  // ----- Init -----
  async function init() {
    await loadModels();
    checkBackendHealth();
    setInterval(checkBackendHealth, 30000);
    bindEvents();
    loadConversations();
  }

  // ----- Models -----
  async function loadModels(retries = 3) {
    for (let attempt = 1; attempt <= retries; attempt++) {
      try {
        const resp = await fetch("/v1/models", { cache: "no-store" });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        const models = data.data || [];
        if (models.length === 0) throw new Error("Empty model list");

        // Populate both selectors
        [el.modelSelect, el.headerModelSelect].forEach((sel) => {
          sel.innerHTML = "";
          models.forEach((m) => {
            const opt = document.createElement("option");
            opt.value = m.id;
            opt.textContent = m.id;
            sel.appendChild(opt);
          });
        });
        state.selectedModel = models[0].id;
        [el.modelSelect, el.headerModelSelect].forEach((s) => (s.value = state.selectedModel));
        updateModelDisplay();
        console.log(`✓ Loaded ${models.length} models`);
        return; // success
      } catch (e) {
        console.warn(`Load models attempt ${attempt}/${retries} failed:`, e.message);
        if (attempt < retries) {
          await new Promise((r) => setTimeout(r, attempt * 1500));
        }
      }
    }
    // All retries exhausted
    console.error("Failed to load models after all retries");
    [el.modelSelect, el.headerModelSelect].forEach((s) => {
      s.innerHTML = '<option value="">加载失败，点击重试</option>';
    });
    // Click to retry
    [el.modelSelect, el.headerModelSelect].forEach((s) => {
      s.onclick = () => {
        s.innerHTML = '<option value="">加载中...</option>';
        s.onclick = null;
        loadModels(3);
      };
    });
  }

  function setModel(modelId) {
    state.selectedModel = modelId;
    el.modelSelect.value = modelId;
    el.headerModelSelect.value = modelId;
    updateModelDisplay();
  }

  function updateModelDisplay() {
    el.modelBadge.textContent = state.selectedModel;
  }

  // ----- Health -----
  async function checkBackendHealth() {
    const dot = el.backendStatus.querySelector(".status-dot");
    const text = el.backendStatus.querySelector(".status-text");
    try {
      const resp = await fetch("/health");
      const data = await resp.json();
      if (data.status === "ok") {
        dot.className = "status-dot online";
        text.textContent = "后端已连接";
      } else {
        dot.className = "status-dot";
        text.textContent = "后端降级";
      }
    } catch {
      dot.className = "status-dot offline";
      text.textContent = "后端离线";
    }
  }

  // ----- Conversations -----
  function loadConversations() {
    try {
      const saved = localStorage.getItem("aicode_conversations");
      if (saved) {
        state.conversations = JSON.parse(saved);
        renderHistory();
      }
    } catch {}
  }

  function saveConversations() {
    try {
      localStorage.setItem("aicode_conversations", JSON.stringify(state.conversations));
    } catch {}
  }

  function renderHistory() {
    el.chatHistory.innerHTML = "";
    state.conversations
      .slice()
      .reverse()
      .forEach((conv) => {
        const div = document.createElement("div");
        div.className = "history-item" + (conv.id === state.currentConvId ? " active" : "");
        div.textContent = conv.title || "新对话";
        div.onclick = () => switchConversation(conv.id);
        el.chatHistory.appendChild(div);
      });
  }

  function newConversation() {
    const id = Date.now().toString(36);
    state.conversations.push({ id, title: "", messages: [] });
    state.currentConvId = id;
    state.messages = [];
    clearChat();
    renderHistory();
    saveConversations();
  }

  function switchConversation(id) {
    const conv = state.conversations.find((c) => c.id === id);
    if (!conv) return;
    state.currentConvId = id;
    state.messages = conv.messages || [];
    clearChat();
    if (state.messages.length > 0) {
      el.welcomeScreen.style.display = "none";
      state.messages.forEach((m) => appendMessageDOM(m.role, m.content));
    }
    renderHistory();
  }

  function saveCurrentConversation() {
    const conv = state.conversations.find((c) => c.id === state.currentConvId);
    if (!conv) return;
    conv.messages = state.messages;
    if (!conv.title && state.messages.length > 0) {
      conv.title = state.messages[0].content.slice(0, 30) + (state.messages[0].content.length > 30 ? "..." : "");
    }
    saveConversations();
    renderHistory();
  }

  // ----- Chat rendering -----
  function clearChat() {
    el.chatMessages.innerHTML = "";
    el.chatMessages.appendChild(el.welcomeScreen);
    el.welcomeScreen.style.display = "";
  }

  function appendMessageDOM(role, content) {
    el.welcomeScreen.style.display = "none";
    const msgDiv = document.createElement("div");
    msgDiv.className = `message ${role}`;

    const avatar = document.createElement("div");
    avatar.className = "message-avatar";
    avatar.textContent = role === "user" ? "U" : "AI";

    const contentDiv = document.createElement("div");
    contentDiv.className = "message-content";
    if (role === "assistant") {
      contentDiv.innerHTML = renderMarkdown(content);
      addCopyButtons(contentDiv);
    } else {
      contentDiv.textContent = content;
    }

    msgDiv.appendChild(avatar);
    msgDiv.appendChild(contentDiv);
    el.chatMessages.appendChild(msgDiv);
    scrollToBottom();
    return contentDiv;
  }

  function renderMarkdown(text) {
    // Strip <think>...</think> tags and render as details
    let html = text;
    const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
    html = html.replace(thinkRegex, (_, content) => {
      return `<details class="thinking-block"><summary>💭 思考过程</summary>${marked.parse(content.trim())}</details>`;
    });
    // Render remaining markdown
    // Split by thinking blocks, render non-thinking parts
    const parts = html.split(/(<details class="thinking-block">[\s\S]*?<\/details>)/);
    return parts
      .map((part) => {
        if (part.startsWith('<details class="thinking-block">')) return part;
        return marked.parse(part);
      })
      .join("");
  }

  function addCopyButtons(container) {
    container.querySelectorAll("pre").forEach((pre) => {
      const wrapper = document.createElement("div");
      wrapper.className = "code-block-wrapper";
      pre.parentNode.insertBefore(wrapper, pre);
      wrapper.appendChild(pre);

      const btn = document.createElement("button");
      btn.className = "code-copy-btn";
      btn.textContent = "复制";
      btn.onclick = () => {
        navigator.clipboard.writeText(pre.textContent).then(() => {
          btn.textContent = "已复制!";
          setTimeout(() => (btn.textContent = "复制"), 1500);
        });
      };
      wrapper.appendChild(btn);
    });
  }

  function scrollToBottom() {
    el.chatMessages.scrollTop = el.chatMessages.scrollHeight;
  }

  // ----- Send message (SSE streaming) -----
  async function sendMessage(text) {
    if (!text.trim() || state.isStreaming) return;

    if (!state.currentConvId) newConversation();

    // Add user message
    state.messages.push({ role: "user", content: text });
    appendMessageDOM("user", text);

    // Create assistant placeholder
    const assistantDiv = appendMessageDOM("assistant", "");
    const typingHTML = '<div class="typing-indicator"><span></span><span></span><span></span></div>';
    assistantDiv.innerHTML = typingHTML;

    // UI state
    setStreaming(true);
    scrollToBottom();

    // Performance tracking
    const perf = {
      startTime: performance.now(),
      firstTokenTime: null,
      endTime: null,
      promptTokens: 0,
      completionTokens: 0,
      totalTokens: 0,
    };

    // Build request
    const body = {
      model: state.selectedModel,
      messages: state.messages.map((m) => ({ role: m.role, content: m.content })),
      stream: true,
      max_tokens: 4096,
      temperature: 0.7,
    };

    state.abortController = new AbortController();

    try {
      const resp = await fetch("/v1/chat/completions", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: state.abortController.signal,
      });

      if (!resp.ok) {
        const err = await resp.text();
        assistantDiv.innerHTML = `<p style="color:#ef4444">请求失败: ${resp.status} ${err}</p>`;
        setStreaming(false);
        return;
      }

      // Read SSE stream
      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let fullContent = "";
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          const trimmed = line.trim();
          if (!trimmed || !trimmed.startsWith("data: ")) continue;
          const data = trimmed.slice(6);
          if (data === "[DONE]") continue;

          try {
            const json = JSON.parse(data);
            const delta = json.choices?.[0]?.delta?.content || "";
            if (delta) {
              if (!perf.firstTokenTime) {
                perf.firstTokenTime = performance.now();
              }
              fullContent += delta;
              assistantDiv.innerHTML = renderMarkdown(fullContent);
              addCopyButtons(assistantDiv);
              scrollToBottom();
            }
            // Extract usage from final chunk
            if (json.usage) {
              perf.promptTokens = json.usage.prompt_tokens || 0;
              perf.completionTokens = json.usage.completion_tokens || 0;
              perf.totalTokens = json.usage.total_tokens || 0;
            }
          } catch {}
        }
      }

      perf.endTime = performance.now();

      // Append performance metrics
      if (fullContent) {
        appendPerfMetrics(assistantDiv, perf);
        state.messages.push({ role: "assistant", content: fullContent });
        saveCurrentConversation();
      }
    } catch (e) {
      perf.endTime = performance.now();
      if (e.name === "AbortError") {
        appendPerfMetrics(assistantDiv, perf, true);
        assistantDiv.innerHTML += "<p style='color:#f59e0b'><em>已停止生成</em></p>";
      } else {
        assistantDiv.innerHTML = `<p style="color:#ef4444">连接错误: ${e.message}</p>`;
      }
    } finally {
      setStreaming(false);
      state.abortController = null;
    }
  }

  // ----- Performance metrics display -----
  function appendPerfMetrics(container, perf, aborted = false) {
    if (!perf.endTime) return;

    const totalDuration = ((perf.endTime - perf.startTime) / 1000).toFixed(1);
    const ttft = perf.firstTokenTime
      ? ((perf.firstTokenTime - perf.startTime) / 1000).toFixed(2)
      : null;
    const genDuration = perf.firstTokenTime
      ? (perf.endTime - perf.firstTokenTime) / 1000
      : 0;
    const tps = genDuration > 0 && perf.completionTokens > 0
      ? (perf.completionTokens / genDuration).toFixed(1)
      : null;

    // If backend didn't return usage, estimate from content length
    const completionTok = perf.completionTokens || estimateTokens(container.textContent);
    const promptTok = perf.promptTokens || "—";
    const totalTok = perf.totalTokens || (typeof promptTok === "number" ? promptTok + completionTok : "—");

    const metricsHTML = `
      <div class="perf-metrics">
        <span class="perf-metric"><span class="perf-icon">⏱</span> 总耗时 <span class="perf-value">${totalDuration}s</span></span>
        ${ttft ? `<span class="perf-metric"><span class="perf-icon">⚡</span> 首Token <span class="perf-value">${ttft}s</span></span>` : ""}
        ${tps ? `<span class="perf-metric highlight"><span class="perf-icon">🚀</span> <span class="perf-value">${tps}</span> tok/s</span>` : ""}
        <span class="perf-metric"><span class="perf-icon">📝</span> 输出 <span class="perf-value">${completionTok}</span> tokens</span>
        <span class="perf-metric"><span class="perf-icon">📥</span> 输入 <span class="perf-value">${promptTok}</span> tokens</span>
        <span class="perf-metric"><span class="perf-icon">📊</span> 总计 <span class="perf-value">${totalTok}</span> tokens</span>
        ${aborted ? `<span class="perf-metric"><span class="perf-icon">⚠️</span> <span class="perf-value">未完成</span></span>` : ""}
      </div>
    `;
    container.insertAdjacentHTML("beforeend", metricsHTML);
  }

  function estimateTokens(text) {
    // Rough estimate: ~1.3 tokens per Chinese char, ~0.75 per English word
    const cjk = (text.match(/[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]/g) || []).length;
    const nonCjk = text.length - cjk;
    return Math.ceil(cjk * 1.3 + nonCjk * 0.3);
  }

  function setStreaming(active) {
    state.isStreaming = active;
    el.btnSend.classList.toggle("hidden", active);
    el.btnStop.classList.toggle("hidden", !active);
    el.chatInput.disabled = active;
    if (!active) el.chatInput.focus();
  }

  function stopStreaming() {
    if (state.abortController) {
      state.abortController.abort();
    }
  }

  // ----- Events -----
  function bindEvents() {
    // Send on Enter
    el.chatInput.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage(el.chatInput.value);
        el.chatInput.value = "";
        autoResize();
      }
    });

    el.chatInput.addEventListener("input", () => {
      el.btnSend.disabled = !el.chatInput.value.trim();
      autoResize();
    });

    el.btnSend.addEventListener("click", () => {
      sendMessage(el.chatInput.value);
      el.chatInput.value = "";
      autoResize();
    });

    el.btnStop.addEventListener("click", stopStreaming);

    el.btnNewChat.addEventListener("click", () => {
      newConversation();
      el.chatInput.focus();
    });

    el.btnToggleSidebar.addEventListener("click", () => {
      el.sidebar.classList.toggle("collapsed");
    });

    el.modelSelect.addEventListener("change", () => {
      setModel(el.modelSelect.value);
    });

    el.headerModelSelect.addEventListener("change", () => {
      setModel(el.headerModelSelect.value);
    });

    // Quick prompts
    document.querySelectorAll(".prompt-btn").forEach((btn) => {
      btn.addEventListener("click", () => {
        const prompt = btn.dataset.prompt.replace(/\\n/g, "\n");
        el.chatInput.value = prompt;
        el.btnSend.disabled = false;
        el.chatInput.focus();
      });
    });
  }

  function autoResize() {
    el.chatInput.style.height = "auto";
    el.chatInput.style.height = Math.min(el.chatInput.scrollHeight, 200) + "px";
  }

  // ----- Start -----
  init();
})();
