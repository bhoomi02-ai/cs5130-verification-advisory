"""Record the dashboard's replay as an animated WebP or GIF for slides. Read-only; no model calls.

    python dashboard/record_replay.py --setting dedicated --seconds 40 --out replay.webp

Needs Chrome (or Chromium), and the `websockets` and Pillow packages (both come with
`pip install -e ".[dashboard]"`). Build the data first with dashboard/build_data.py. It serves
dashboard/ on a free localhost port, opens the page in headless Chrome, picks the setting,
plays the replay at the chosen speed and saves screenshots of the replay panel as an animation.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import functools
import http.server
import io
import json
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

HERE = Path(__file__).resolve().parent
CHROMES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "", shutil.which("chromium") or "", shutil.which("chromium-browser") or "",
]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the output clean
        pass


def _port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def record(url: str, chrome: str, setting: str, speed: int, seconds: float, fps: int, width: int) -> list:
    import websockets
    from PIL import Image

    port, profile = _port(), tempfile.mkdtemp(prefix="replay-rec-")
    proc = subprocess.Popen([chrome, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
                             "--no-first-run", "--hide-scrollbars", "about:blank"],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    frames = []
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except OSError:
                time.sleep(0.1)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=100 * 1024 * 1024) as ws:
            n = 0

            async def cmd(method, **params):
                nonlocal n
                n += 1
                await ws.send(json.dumps({"id": n, "method": method, "params": params}))
                while True:
                    msg = json.loads(await ws.recv())
                    if msg.get("id") == n:
                        return msg.get("result", {})

            async def js(expr):
                r = await cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
                return r.get("result", {}).get("value")

            await cmd("Page.enable")
            await cmd("Emulation.setDeviceMetricsOverride", width=width, height=1000, deviceScaleFactor=2, mobile=False)
            await cmd("Page.navigate", url=url)
            for _ in range(200):
                if await js("!!(window.__dashboard && document.querySelector('#replay-mount svg'))"):
                    break
                await asyncio.sleep(0.1)
            pending = await js(f"(window.__dashboard.select({json.dumps(setting)}, false), document.querySelector('#replay-mount .replay-pending')?.textContent || '')")
            if pending:
                raise SystemExit(f"{setting} is {pending}: nothing to record yet")
            await js(f"[...document.querySelectorAll('#replay-mount .ar-spd')].find(b => b.dataset.v === '{speed}').click()")
            await js("document.getElementById('replay').scrollIntoView()")
            await asyncio.sleep(0.5)
            box = json.loads(await js("JSON.stringify((r => ({x: r.left + scrollX, y: r.top + scrollY, width: r.width, height: r.height}))(document.getElementById('replay').getBoundingClientRect()))"))
            end = time.time() + seconds
            while time.time() < end:
                t0 = time.time()
                shot = await cmd("Page.captureScreenshot", format="png", clip={**box, "scale": 1}, captureBeyondViewport=True)
                frames.append(Image.open(io.BytesIO(base64.b64decode(shot["data"]))).convert("RGB"))
                await asyncio.sleep(max(0, 1 / fps - (time.time() - t0)))
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)
    return frames


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--setting", default="dedicated")
    parser.add_argument("--speed", type=int, choices=[1, 2, 4], default=4)
    parser.add_argument("--seconds", type=float, default=30)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--width", type=int, default=1280, help="page width in CSS pixels (below 560 the replay is stacked)")
    parser.add_argument("--out", type=Path, default=Path("replay.webp"), help=".webp or .gif")
    args = parser.parse_args(argv)
    chrome = next((c for c in CHROMES if c and Path(c).exists()), None)
    if chrome is None:
        print("Chrome or Chromium is needed", file=sys.stderr)
        return 1
    if not (HERE / "data" / "runs.json").exists():
        print("No dashboard/data/: run python dashboard/build_data.py first", file=sys.stderr)
        return 1
    handler = functools.partial(_Quiet, directory=str(HERE))
    port = _port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        frames = asyncio.run(record(f"http://127.0.0.1:{port}/", chrome, args.setting, args.speed, args.seconds, args.fps, args.width))
    finally:
        server.shutdown()
    if not frames:
        print("No frames captured", file=sys.stderr)
        return 1
    duration = int(1000 / args.fps)
    if args.out.suffix.lower() == ".gif":
        frames = [f.resize((f.width // 2, f.height // 2)) for f in frames]
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=duration, loop=0, optimize=True)
    else:
        frames[0].save(args.out, save_all=True, append_images=frames[1:], duration=duration, loop=0, quality=80, method=4)
    print(f"Saved {len(frames)} frames ({args.seconds:g} s at {args.speed}× speed) of {args.setting} to {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
