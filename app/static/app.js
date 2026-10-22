"use strict";

const $ = (id) => document.getElementById(id);
const log = $("log"), input = $("input"), sendBtn = $("send"), modelSel = $("model");
const state = { messages: [], busy: false };

function render(md) {
  // Model output is untrusted: always sanitize before inserting HTML.
  return DOMPurify.sanitize(marked.parse(md, { breaks: true }));
}

function showEmpty() {
  log.innerHTML = '<div class="empty">Your local assistant.<br>Nothing leaves this computer.</div>';
}

function addMessage(role, text = "") {
  if (log.querySelector(".empty")) log.innerHTML = "";
  const el = document.createElement("div");
  el.className = `msg ${role}`;
  if (role === "user") el.textContent = text;
  else if (text) el.innerHTML = render(text);
  log.appendChild(el);
  log.scrollTop = log.scrollHeight;
  return el;
}

function setBanner(text) {
  const b = $("banner");
  b.hidden = !text;
  b.textContent = text || "";
}

async function checkHealth() {
  try {
    const h = await (await fetch("/api/health")).json();
    if (!h.ollama) setBanner(`Ollama is not reachable at ${h.ollama_url}. ${h.hint || ""}`);
    else setBanner("");
    return h.ollama;
  } catch {
    setBanner("The helper backend is not responding.");
    return false;
  }
}

async function loadModels() {
  modelSel.innerHTML = "";
  let data;
  try {
    const r = await fetch("/api/models");
    data = await r.json();
    if (!r.ok) throw new Error(data.error);
  } catch (e) {
    modelSel.innerHTML = '<option value="">no models</option>';
    return;
  }
  const chatModels = data.models.filter((m) => !m.embedding);
  if (!chatModels.length) {
    modelSel.innerHTML = '<option value="">no chat models: run "ollama pull qwen3:8b"</option>';
    return;
  }
  for (const m of chatModels) {
    const o = document.createElement("option");
    o.value = m.name;
    o.textContent = m.parameters ? `${m.name} (${m.parameters})` : m.name;
    modelSel.appendChild(o);
  }
  let saved = null;
  try { saved = localStorage.getItem("model"); } catch {}
  const want = [saved, data.default].find((n) => n && chatModels.some((m) => m.name === n));
  if (want) modelSel.value = want;
}

modelSel.addEventListener("change", () => {
  try { localStorage.setItem("model", modelSel.value); } catch {}
});

async function send(text) {
  state.busy = true;
  sendBtn.disabled = true;
  state.messages.push({ role: "user", content: text });
  addMessage("user", text);
  const el = addMessage("assistant");
  let answer = "", thinking = "", thinkEl = null;
  $("status").textContent = "";

  try {
    const r = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        model: modelSel.value, messages: state.messages,
        think: $("think").checked, use_docs: $("docs").checked,
      }),
    });
    if (!r.ok) throw new Error((await r.json()).error || r.statusText);

    const reader = r.body.getReader();
    const dec = new TextDecoder();
    let buf = "";
    for (;;) {
      const { value, done } = await reader.read();
      if (done) break;
      buf += dec.decode(value, { stream: true });
      let i;
      while ((i = buf.indexOf("\n\n")) >= 0) {
        const line = buf.slice(0, i).trim();
        buf = buf.slice(i + 2);
        if (!line.startsWith("data:")) continue;
        const ev = JSON.parse(line.slice(5));
        if (ev.error) throw new Error(ev.error);
        if (ev.warning) $("status").textContent = ev.warning;
        if (ev.sources) showSources(el, ev.sources);
        if (ev.thinking) {
          thinking += ev.thinking;
          if (!thinkEl) {
            thinkEl = document.createElement("details");
            thinkEl.className = "think";
            thinkEl.innerHTML = "<summary>Thinking</summary><div></div>";
            el.prepend(thinkEl);
          }
          thinkEl.querySelector("div").textContent = thinking;
        }
        if (ev.content) {
          answer += ev.content;
          let body = el.querySelector(".body");
          if (!body) {
            body = document.createElement("div"); body.className = "body";
            el.insertBefore(body, el.querySelector(".sources"));  // sources stay at the bottom
          }
          body.innerHTML = render(answer);
        }
        if (ev.done && ev.tokens_per_s) $("status").textContent = `${ev.tokens} tokens, ${ev.tokens_per_s} tok/s`;
        log.scrollTop = log.scrollHeight;
      }
    }
    state.messages.push({ role: "assistant", content: answer });
  } catch (e) {
    el.classList.add("error");
    el.append(document.createTextNode(`Error: ${e.message}`));
    state.messages.pop(); // drop the unanswered user turn so a retry does not duplicate it
  } finally {
    state.busy = false;
    sendBtn.disabled = false;
    input.focus();
  }
}

$("form").addEventListener("submit", (e) => {
  e.preventDefault();
  const text = input.value.trim();
  if (!text || state.busy) return;
  if (!modelSel.value) { setBanner("Pick a model first (run `ollama pull qwen3:8b` if the list is empty)."); return; }
  input.value = "";
  input.style.height = "auto";
  send(text);
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey && !e.isComposing) { e.preventDefault(); $("form").requestSubmit(); }
});
input.addEventListener("input", () => {
  input.style.height = "auto";
  input.style.height = Math.min(input.scrollHeight, 200) + "px";
});

$("new").addEventListener("click", () => {
  if (state.busy) return;
  state.messages = [];
  showEmpty();
  $("status").textContent = "";
});

function showSources(el, sources) {
  if (!sources.length) return;
  const box = document.createElement("details");
  box.className = "sources";
  const sum = document.createElement("summary");
  sum.textContent = `Sources (${sources.length})`;
  box.appendChild(sum);
  const ol = document.createElement("ol");
  for (const s of sources) {
    const li = document.createElement("li");
    li.value = s.n;
    li.textContent = s.page ? `${s.path}, page ${s.page}` : s.path;  // textContent: file names are untrusted
    li.title = s.snippet;
    ol.appendChild(li);
  }
  box.appendChild(ol);
  el.appendChild(box);
}

let docsTimer = null;
async function refreshDocs() {
  let s;
  try { s = await (await fetch("/api/docs/status")).json(); } catch { return; }
  const out = $("docstatus");
  if (!s.mounted) out.textContent = "Documents folder not mounted";
  else if (s.indexing) out.textContent = `Indexing ${s.progress}`;
  else if (s.error) out.textContent = `Index error: ${s.error}`;
  else out.textContent = `${s.files} files, ${s.chunks} passages` + (s.problems.length ? `, ${s.problems.length} unreadable` : "");
  out.title = s.problems.map((p) => `${p.path}: ${p.error}`).join("\n");
  $("reindex").disabled = s.indexing;
  clearTimeout(docsTimer);
  if (s.indexing) docsTimer = setTimeout(refreshDocs, 1000);
}

$("reindex").addEventListener("click", async () => {
  await fetch("/api/docs/reindex", { method: "POST" });
  refreshDocs();
});

$("docs").addEventListener("change", () => {
  try { localStorage.setItem("docs", $("docs").checked ? "1" : "0"); } catch {}
});
try { $("docs").checked = localStorage.getItem("docs") === "1"; } catch {}

showEmpty();
checkHealth().then(loadModels);
refreshDocs();
