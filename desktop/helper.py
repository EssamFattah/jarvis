"""Desktop window for Local AI Helper (macOS: native WKWebView via pywebview).

Starts the Docker stack if it is not already up, waits until the backend answers,
then opens the UI in its own window (no browser tabs, no address bar).
"""
import os
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import webview

ROOT = Path(__file__).resolve().parent.parent
PORT = os.environ.get("APP_PORT", "7860")
URL = f"http://127.0.0.1:{PORT}"


def backend_up() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/api/health", timeout=2) as r:
            return r.status == 200
    except Exception:  # noqa: BLE001
        return False


def start_stack() -> None:
    subprocess.run(["docker", "compose", "up", "-d", "--build"], cwd=ROOT, check=True)


def wait_for_backend(seconds: int = 90) -> bool:
    deadline = time.time() + seconds
    while time.time() < deadline:
        if backend_up():
            return True
        time.sleep(1)
    return False


def main() -> int:
    if not backend_up():
        print("Starting the Docker stack...")
        try:
            start_stack()
        except FileNotFoundError:
            print("Docker is not installed or not on PATH. Install Docker Desktop first.")
            return 1
        except subprocess.CalledProcessError:
            print("`docker compose up` failed. Is Docker Desktop running?")
            return 1
        if not wait_for_backend():
            print("The backend did not become ready. Run `docker compose logs app`.")
            return 1

    data = ROOT / "data" / "webview"
    data.mkdir(parents=True, exist_ok=True)
    webview.create_window(
        "Local AI Helper", URL, width=1000, height=780, min_size=(560, 420),
        text_select=True,  # off by default in pywebview; needed to copy answers
    )
    # private_mode=False keeps the remembered model choice between launches.
    webview.start(private_mode=False, storage_path=str(data))
    return 0


if __name__ == "__main__":
    sys.exit(main())
