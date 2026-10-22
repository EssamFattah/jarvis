"""Local AI Helper: backend (step 2: chat + your documents).

Streams chat from a local Ollama server to the desktop window over
Server-Sent Events, searches your documents (see rag.py), and serves the UI.
"""
import json
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.concurrency import run_in_threadpool
from fastapi.staticfiles import StaticFiles

from app import rag

OLLAMA_URL = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434").rstrip("/")
DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "")
SYSTEM_PROMPT = os.environ.get(
    "SYSTEM_PROMPT",
    "You are a helpful local assistant. Answer concisely and say so when you are unsure.",
)
# Hosts the UI may be reached on. Anything else (e.g. a DNS-rebinding attack
# from a web page) is rejected.
ALLOWED_HOSTS = {h.strip() for h in os.environ.get("ALLOWED_HOSTS", "127.0.0.1,localhost").split(",") if h.strip()}
STATIC_DIR = Path(__file__).parent / "static"
DOCS_DIR = os.environ.get("DOCS_DIR", "/docs")
INDEX_DB = os.environ.get("INDEX_DB", "/data/index.sqlite")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "qwen3-embedding:0.6b")
NUM_CTX = int(os.environ.get("NUM_CTX", "8192"))      # context window passed to Ollama
TOP_K = int(os.environ.get("TOP_K", "5"))             # document excerpts per question
AUTO_INDEX = os.environ.get("AUTO_INDEX", "1") == "1"  # index changed files at startup

index = rag.DocIndex(DOCS_DIR, INDEX_DB, OLLAMA_URL, EMBED_MODEL)


def _reindex_in_background():
    threading.Thread(target=index.reindex, daemon=True).start()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if AUTO_INDEX:
        _reindex_in_background()
    yield


app = FastAPI(title="Local AI Helper", lifespan=lifespan)


def _hostname(value: str) -> str:
    """'127.0.0.1:7860' -> '127.0.0.1'; '[::1]:7860' -> '[::1]'."""
    if value.startswith("["):
        return value.split("]")[0] + "]"
    return value.split(":")[0]


@app.middleware("http")
async def local_only_guard(request: Request, call_next):
    """Reject requests whose Host is not ours, and cross-origin writes."""
    if _hostname(request.headers.get("host", "")) not in ALLOWED_HOSTS:
        return JSONResponse({"error": "forbidden host"}, status_code=403)
    if request.method not in ("GET", "HEAD", "OPTIONS"):
        origin = request.headers.get("origin")
        if origin and _hostname(origin.split("://", 1)[-1]) not in ALLOWED_HOSTS:
            return JSONResponse({"error": "forbidden origin"}, status_code=403)
    return await call_next(request)


def _sse(payload: dict[str, Any]) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@app.get("/api/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=3) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
        return {"ok": True, "ollama": True, "ollama_url": OLLAMA_URL}
    except Exception as e:  # noqa: BLE001 - report any failure to the UI
        return {
            "ok": True,
            "ollama": False,
            "ollama_url": OLLAMA_URL,
            "error": f"{type(e).__name__}: {e}",
            "hint": "Start the Ollama app on your Mac. If it is running, see 'Ollama not reachable' in the README.",
        }


@app.get("/api/models")
async def models():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            r = await client.get(f"{OLLAMA_URL}/api/tags")
            r.raise_for_status()
            data = r.json()
    except Exception as e:  # noqa: BLE001
        return JSONResponse({"error": f"Cannot reach Ollama at {OLLAMA_URL}: {e}"}, status_code=502)
    items = []
    for m in data.get("models", []):
        d = m.get("details") or {}
        items.append(
            {
                "name": m.get("name"),
                "size": m.get("size"),
                "parameters": d.get("parameter_size"),
                "quantization": d.get("quantization_level"),
                # embedding models cannot chat; the UI hides them
                "embedding": "embed" in (m.get("name") or "").lower(),
            }
        )
    return {"models": items, "default": DEFAULT_MODEL}


@app.get("/api/docs/status")
async def docs_status():
    return index.status()


@app.post("/api/docs/reindex")
async def docs_reindex(wait: bool = False):
    if wait:  # used by tests and scripts
        return await run_in_threadpool(index.reindex)
    if index.indexing:
        return {"started": False, "reason": "already indexing"}
    _reindex_in_background()
    return {"started": True}


@app.post("/api/chat")
async def chat(request: Request):
    body = await request.json()
    model = body.get("model") or DEFAULT_MODEL
    history = [
        {"role": m["role"], "content": str(m["content"])}
        for m in body.get("messages", [])
        if m.get("role") in ("user", "assistant") and isinstance(m.get("content"), str)
    ]
    if not model:
        return JSONResponse({"error": "No model selected."}, status_code=400)
    if not history:
        return JSONResponse({"error": "No messages."}, status_code=400)

    if history[-1]["role"] != "user":
        return JSONResponse({"error": "The last message must be from the user."}, status_code=400)

    # Documents mode: retrieve excerpts for the latest question and wrap them, with
    # the citation rules, around that question. The client's copy of the history
    # stays clean, so each turn retrieves fresh excerpts.
    prelude: list[dict[str, Any]] = []
    if body.get("use_docs") is True:
        question = history[-1]["content"]
        if index.status()["chunks"] == 0:
            msg = ("No documents are indexed yet. Put files in your documents folder, "
                   "then press Reindex." if not index.indexing else "Your documents are still being indexed. Try again in a moment.")
            return StreamingResponse(iter([_sse({"sources": []}), _sse({"content": msg}), _sse({"done": True})]),
                                     media_type="text/event-stream")
        hits, warning = await run_in_threadpool(index.search, question, TOP_K)
        if warning:
            prelude.append({"warning": warning})
        prelude.append({"sources": [{"n": i, "path": h.path, "page": h.page, "snippet": h.text[:300]}
                                    for i, h in enumerate(hits, 1)]})
        if not hits:
            return StreamingResponse(
                iter([_sse(p) for p in prelude] + [_sse({"content": rag.NOT_FOUND}), _sse({"done": True})]),
                media_type="text/event-stream")
        history[-1] = {"role": "user", "content": rag.build_docs_message(question, hits)}

    payload: dict[str, Any] = {
        "model": model,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}] + history,
        "stream": True,
        "options": {"num_ctx": NUM_CTX},
    }
    if body.get("think") is True:
        payload["think"] = True  # only sent when asked: non-thinking models reject it

    async def stream():
        for p in prelude:
            yield _sse(p)
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(300, connect=5)) as client:
                async with client.stream("POST", f"{OLLAMA_URL}/api/chat", json=payload) as r:
                    if r.status_code != 200:
                        text = (await r.aread()).decode("utf-8", "replace")
                        try:
                            text = json.loads(text).get("error", text)
                        except ValueError:
                            pass
                        yield _sse({"error": f"Ollama returned {r.status_code}: {text}"})
                        return
                    async for line in r.aiter_lines():
                        if not line.strip():
                            continue
                        chunk = json.loads(line)
                        if "error" in chunk:
                            yield _sse({"error": chunk["error"]})
                            return
                        msg = chunk.get("message") or {}
                        if msg.get("thinking"):
                            yield _sse({"thinking": msg["thinking"]})
                        if msg.get("content"):
                            yield _sse({"content": msg["content"]})
                        if chunk.get("done"):
                            tokens = chunk.get("eval_count")
                            secs = (chunk.get("eval_duration") or 0) / 1e9
                            yield _sse(
                                {
                                    "done": True,
                                    "tokens": tokens,
                                    "tokens_per_s": round(tokens / secs, 1) if tokens and secs else None,
                                }
                            )
        except httpx.ConnectError:
            yield _sse({"error": f"Cannot reach Ollama at {OLLAMA_URL}. Is the Ollama app running?"})
        except Exception as e:  # noqa: BLE001
            yield _sse({"error": f"{type(e).__name__}: {e}"})

    return StreamingResponse(stream(), media_type="text/event-stream", headers={"Cache-Control": "no-store"})


app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")
