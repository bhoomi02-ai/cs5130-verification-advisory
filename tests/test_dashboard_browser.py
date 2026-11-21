"""Browser check of the web dashboard with the REAL results (dashboard/ + data built from results/).

Skipped unless Chrome (or Chromium), the `websockets` package and results/results.jsonl are all
present, so a fresh clone and CI skip it; run it on the machine that has the study results.

It serves the page, drives headless Chrome through the DevTools protocol and checks that:
- the page loads and the replay mounts for every finished setting, with no JavaScript errors;
- an unfinished setting shows "pending (n/N)" in the replay instead of replaying;
- with prefers-reduced-motion the replay shows its finished state;
- opening run details (every run whose checker failed something, plus a sample of the rest)
  never brings an `expected` key into the page: not in the JSON the page fetches, not in the DOM.
"""

from __future__ import annotations

import asyncio
import functools
import os
import http.server
import json
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import urllib.request
from pathlib import Path

import pytest

from advisory import dashboard_export as ex

websockets = pytest.importorskip("websockets")

CHROME_CANDIDATES = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    shutil.which("google-chrome") or "",
    shutil.which("chromium") or "",
    shutil.which("chromium-browser") or "",
]
CHROME = next((c for c in CHROME_CANDIDATES if c and Path(c).exists()), None)
RESULTS = Path(os.environ.get("DASHBOARD_TEST_RESULTS") or ex.RESULTS_DIR)  # override to test another results folder

pytestmark = [
    pytest.mark.skipif(CHROME is None, reason="needs Chrome or Chromium"),
    pytest.mark.skipif(not (RESULTS / "results.jsonl").exists(), reason="needs the real results/ (study-results)"),
]


class _Quiet(http.server.SimpleHTTPRequestHandler):
    def log_message(self, *args):  # keep the output clean
        pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture(scope="module")
def site():
    """dashboard/ static files plus data built from the real results, served on localhost."""
    root = Path(tempfile.mkdtemp(prefix="dash-site-"))
    src = ex.ROOT / "dashboard"
    for name in ("index.html", "style.css", "app.js", "replay.js"):
        shutil.copy(src / name, root / name)
    shutil.copytree(src / "vendor", root / "vendor")
    meta = ex.build(RESULTS, root / "data")
    handler = functools.partial(_Quiet, directory=str(root))
    port = _free_port()
    server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield {"url": f"http://127.0.0.1:{port}/", "meta": meta, "root": root}
    server.shutdown()
    shutil.rmtree(root, ignore_errors=True)


class Page:
    """Just enough of the DevTools protocol: evaluate JS, collect exceptions and console errors."""

    def __init__(self, ws):
        self.ws, self.n, self.errors = ws, 0, []

    async def cmd(self, method, **params):
        self.n += 1
        await self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            msg = json.loads(await self.ws.recv())
            if msg.get("method") == "Runtime.exceptionThrown":
                d = msg["params"]["exceptionDetails"]
                self.errors.append((d.get("exception") or {}).get("description") or d.get("text"))
            elif msg.get("method") == "Runtime.consoleAPICalled" and msg["params"].get("type") == "error":
                self.errors.append(json.dumps(msg["params"]["args"])[:300])
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(msg["error"])
                return msg.get("result", {})

    async def js(self, expr):
        r = await self.cmd("Runtime.evaluate", expression=expr, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise AssertionError(f"JS error in {expr[:80]}: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    async def wait_for(self, expr, timeout=30):
        end = time.time() + timeout
        while time.time() < end:
            if await self.js(expr):
                return
            await asyncio.sleep(0.2)
        raise AssertionError(f"timed out waiting for {expr}")


async def _with_page(url, body, reduced_motion=False):
    profile = tempfile.mkdtemp(prefix="dash-chrome-")
    port = _free_port()
    proc = subprocess.Popen(
        [CHROME, "--headless=new", f"--remote-debugging-port={port}", f"--user-data-dir={profile}",
         "--no-first-run", "--no-default-browser-check", "about:blank"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        for _ in range(100):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                break
            except OSError:
                time.sleep(0.1)
        ws_url = next(t["webSocketDebuggerUrl"] for t in tabs if t["type"] == "page")
        async with websockets.connect(ws_url, max_size=50 * 1024 * 1024) as ws:
            page = Page(ws)
            await page.cmd("Runtime.enable")
            await page.cmd("Page.enable")
            if reduced_motion:
                await page.cmd("Emulation.setEmulatedMedia", features=[{"name": "prefers-reduced-motion", "value": "reduce"}])
            await page.cmd("Page.navigate", url=url)
            await page.wait_for("document.getElementById('headline') && !document.getElementById('headline').textContent.startsWith('Loading')")
            await body(page)
            return page.errors
    finally:
        proc.terminate()
        shutil.rmtree(profile, ignore_errors=True)


def test_replay_mounts_for_every_finished_setting_without_errors(site):
    complete = site["meta"]["complete_conditions"]
    pending = [c for c in site["meta"]["conditions"] if not c["complete"]]
    seen = {}

    async def body(page):
        for condition in complete:
            await page.js(f"window.__dashboard.select({json.dumps(condition)}, false)")
            await page.wait_for("document.querySelectorAll('#replay-mount svg').length === 2")
            seen[condition] = await page.js("document.getElementById('replay-sub').textContent")
            await asyncio.sleep(0.6)  # let a few animation frames run
        for c in pending:
            await page.js(f"window.__dashboard.select({json.dumps(c['id'])}, false)")
            seen[c["id"]] = await page.js("document.getElementById('replay-mount').textContent")

    errors = asyncio.run(_with_page(site["url"], body))
    assert errors == []
    runs = json.loads((site["root"] / "data" / "runs.json").read_text())["runs"]
    for condition in complete:
        n = sum(1 for r in runs if r["condition"] == condition and r["status"] != "failed")
        assert seen[condition].startswith(f"{n} recorded"), seen[condition]
    for c in pending:
        assert seen[c["id"]] == f"pending ({c['runs_done']}/{c['runs_expected']})"


def test_replay_shows_the_finished_state_with_reduced_motion(site):
    out = {}

    async def body(page):
        await page.wait_for("document.querySelectorAll('#replay-mount svg').length === 2")
        out["text"] = await page.js("document.querySelector('#replay-mount svg').textContent")
        out["runs"] = await page.js("document.querySelector('#replay-mount .ar-cnt b').textContent")

    errors = asyncio.run(_with_page(site["url"], body, reduced_motion=True))
    assert errors == []
    assert "Replay complete" in out["text"]
    done, total = out["runs"].split(" / ")
    assert done == total


def test_run_details_never_show_the_answer_key(site):
    runs = json.loads((site["root"] / "data" / "runs.json").read_text())["runs"]
    with_failed_checks = [r["run_id"] for r in runs if r["outcome"] in ("still_wrong", "broken", "fixed")]
    sample = with_failed_checks + [r["run_id"] for r in runs[::25]]
    leaks = {}

    async def body(page):
        for run_id in dict.fromkeys(sample):
            await page.js(f"window.__dashboard.openRun({json.dumps(run_id)})")
            await page.wait_for("!document.getElementById('run-detail').textContent.startsWith('Loading')")
            html = await page.js("document.getElementById('run-detail').innerHTML")
            keys = await page.js(
                "fetch('data/runs/' + encodeURIComponent(" + json.dumps(run_id) + ".replace(/[^A-Za-z0-9._-]/g, '_')) + '.json')"
                ".then(r => r.ok ? r.json() : {}).then(d => { const ks = new Set();"
                " (function walk(v) { if (Array.isArray(v)) v.forEach(walk);"
                " else if (v && typeof v === 'object') Object.entries(v).forEach(([k, x]) => { ks.add(k); walk(x); }); })(d);"
                " return [...ks]; })")
            if {"expected", "correct_value", "wrong_value"} & set(keys) or '"expected"' in html or "expected:" in html:
                leaks[run_id] = html[:200]

    errors = asyncio.run(_with_page(site["url"], body))
    assert errors == []
    assert len(with_failed_checks) > 10  # the sample really includes runs with failed checks
    assert leaks == {}


def test_compare_view_status_strip_and_seeded_types_render_from_the_data(site):
    pairs = json.loads((site["root"] / "data" / "pairs.json").read_text())
    seeded = json.loads((site["root"] / "data" / "seeded.json").read_text())
    out = {}

    async def body(page):
        out["pair"] = await page.js("document.getElementById('pair-body').innerText")
        out["disabled"] = await page.js("[...document.querySelectorAll('#pair-b option:disabled')].map(o => o.value)")
        out["types"] = await page.js("document.querySelectorAll('#type-table tbody tr').length")
        out["status"] = await page.js("document.getElementById('dataset').innerText")
        out["png"] = await page.js("document.querySelectorAll('.png-btn').length")
        # the stacked replay layout below 560 px
        await page.cmd("Emulation.setDeviceMetricsOverride", width=400, height=900, deviceScaleFactor=1, mobile=True)
        await page.js("window.dispatchEvent(new Event('resize'))")
        await page.wait_for("(document.querySelector('#replay-mount svg') || {}).getAttribute && document.querySelector('#replay-mount svg').getAttribute('viewBox') === '0 0 360 620'")
        out["width"] = await page.js("[document.documentElement.scrollWidth, window.innerWidth]")

    errors = asyncio.run(_with_page(site["url"], body))
    assert errors == []
    assert "Change, B − A" in out["pair"] and "Sign test over tasks" in out["pair"]
    pending = [c["id"] for c in site["meta"]["conditions"] if not c["complete"]]
    assert sorted(out["disabled"]) == sorted(pending)
    assert out["types"] == len(seeded["by_error_type"])
    assert "data as of" in out["status"] and "32 tasks × 3 runs" in out["status"]
    for c in site["meta"]["conditions"]:
        if not c["complete"]:
            assert f"still running ({c['runs_done']}/{c['runs_expected']})" in out["status"]
    assert out["png"] >= 7
    assert out["width"][0] == out["width"][1] == 400


def test_motion_plays_once_and_never_hides_data_for_reduced_motion(site):
    out = {}

    async def scroll_all(page):
        out["pending_at_load"] = await page.js("document.querySelectorAll('.anim.pending').length")
        await page.js("""new Promise(async (done) => {
            for (let y = 0; y < document.body.scrollHeight; y += 250) { window.scrollTo(0, y); await new Promise(r => setTimeout(r, 60)); }
            done(true); })""")
        await asyncio.sleep(1.5)
        out["pending_after_scroll"] = await page.js("document.querySelectorAll('.anim.pending').length")
        out["runs_hidden"] = await page.js("document.getElementById('runs-body').hidden")
        out["s04"] = await page.js("document.getElementById('s04-summary').textContent")

    async def reduced(page):
        out["reduced_anim"] = await page.js("document.querySelectorAll('.anim').length")
        out["reduced_hidden"] = await page.js(
            "[...document.querySelectorAll('.anim-bar, .anim-pt, .chip')].filter(e => getComputedStyle(e).opacity === '0').length")

    assert asyncio.run(_with_page(site["url"], scroll_all)) == []
    assert asyncio.run(_with_page(site["url"], reduced, reduced_motion=True)) == []
    assert out["pending_at_load"] > 0 and out["pending_after_scroll"] == 0
    assert out["runs_hidden"] is True  # the run list starts folded away
    assert out["s04"].startswith("On one known-wrong S04 answer")
    assert out["reduced_anim"] == 0 and out["reduced_hidden"] == 0
