# Local AI Helper

A private AI assistant that runs on your Mac. Qwen runs locally through Ollama, the app runs in Docker, and the UI is a native desktop window (no browser tab).

Status: **step 2 of 5, chat plus your documents.** Web search (step 3), saved chats (step 4) and an evaluation set (step 5) come next.

## One-time setup

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/) and start it.
2. Install [Ollama](https://ollama.com/download) and start it.
3. Pull a chat model. Pick the size that fits your Mac's memory (check with `ollama list` afterwards):

   ```
   ollama pull qwen3:8b
   ```

   On 8 GB of RAM, use a smaller one such as `qwen3:4b`. The model picker lists whatever `ollama list` shows.

4. Pull the embedding model used for document search (639 MB):

   ```
   ollama pull qwen3-embedding:0.6b
   ```

## Run

```
cd ~/Desktop/jarvis
./start.sh
```

The first run creates a Python environment for the desktop window and builds the Docker image, so it takes a couple of minutes. After that it opens in seconds. The window talks to the app at `http://127.0.0.1:7860`, which is bound to your own machine only.

To stop the backend: `docker compose down`.

## Your documents

Tick **My documents** in the top bar to answer from your files instead of the model's general knowledge. Answers cite sources as [1], [2] and list the file (and PDF page) under the answer. If the files do not contain the answer, the helper says so instead of guessing.

- Supported: PDF, Word (.docx), Markdown, plain text. Scanned PDFs (images only) have no text and are listed as unreadable.
- By default the helper searches the `docs` folder inside this project. To use another folder, set `DOCS_FOLDER` in `.env` to its full path (for example `/Users/essam/Documents/Notes`) and restart with `docker compose up -d`.
- The folder is mounted read-only. The index lives in `data/index.sqlite`.
- New or changed files are indexed at startup, or when you press **Reindex** at the bottom of the window. Hover over the file count to see unreadable files.
- Search is hybrid: keyword search (BM25, SQLite FTS5) and meaning search (embeddings), merged with reciprocal rank fusion.
- Text from your files is treated as data, not instructions, but a small local model can still be misled by a hostile document. Only add files you trust.

## Ollama not reachable

The container reaches Ollama at `http://host.docker.internal:11434`. If the UI shows an "Ollama is not reachable" banner while the Ollama app is running, Docker may not be able to reach a loopback-only Ollama. Make Ollama listen on all interfaces, then restart the Ollama app:

```
launchctl setenv OLLAMA_HOST "0.0.0.0:11434"
```

Be aware that this lets other devices on your network reach Ollama, which has no login. Use it only on a network you trust, or keep the macOS firewall on. Skip this step if the banner does not appear.

## Tests

The tests use a mock Ollama, so they need no model or GPU:

```
python3 -m venv .venv-test && .venv-test/bin/pip install -r requirements.txt pytest
.venv-test/bin/pytest
```

## Design notes

- Ollama runs natively on the Mac, not in Docker, because Docker on Apple Silicon cannot use the GPU.
- The backend refuses requests whose `Host` is not `127.0.0.1` or `localhost`, and cross-origin POSTs, so a web page you visit cannot talk to your local assistant.
- Model output is rendered as Markdown and sanitised (DOMPurify) before it reaches the page.
- The client can only send `user` and `assistant` messages. The system prompt is set on the server.
- Inspired by the setup pattern of [Odysseus](https://github.com/odysseus-dev/odysseus) (AGPL-3.0). No code is copied.
- Vendored in `app/static/vendor/`: marked 18.1.1 (MIT) and DOMPurify 3.4.17 (Apache-2.0 or MPL-2.0).
