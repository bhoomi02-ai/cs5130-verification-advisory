# Results dashboard (static web page)

A single-page, read-only view of the study results. It never calls a model, never reads the answer key and never spends money: the page only loads small JSON files that `build_data.py` derives from `results/`.

> All nine views are built. Regenerate the data after each batch; unfinished settings show as pending.

## View it

```bash
python dashboard/build_data.py              # (re)build dashboard/data/ from results/
python -m http.server 8000 -d dashboard     # then open http://localhost:8000
```

The page loads `data/*.json` with `fetch`, so it needs a web server (opening `index.html` as a file shows a hint instead). It is plain HTML, CSS and JavaScript with d3 7.9.0 included in `vendor/` (checked against the npm registry's SHA-512), so there is no build step and nothing is loaded from the internet.

| # | View | Reads |
|---|---|---|
| 1 | Headline sentence and data set, no-check gauge, and KPI tiles for a chosen setting (task success with CI, catch rate, false alarms, fixed vs broken, cost per run, total spend vs the $20 cap) | `summary.json`, `spend.json`, `meta.json` |
| 2 | Shown right after the replay: task success per setting with 95% CIs and the paired change vs `none` with CI and sign test; success with and without human review in a collapsible section | `summary.json` |
| 3 | Catch rate and false-alarm rate with counts, and MCC | `summary.json` |
| 4 | Fixed (wrong → right) vs broken (right → wrong) per setting | `summary.json` |
| 5 | Scenario × setting heatmap, right out of 3 runs per task; S01–S10 rows open to show their variants | `heatmap.json` |
| 6 | Budget bar, cost and tokens per run, cost vs success | `spend.json`, `summary.json` |
| 7 | Verifier sanity checks: seeded errors by type × verifier (rejected, for the right reason, false alarms on the controls) with a Total row, every seeded answer's decision (folded), and a one-sentence summary of the pre-study S04 check with its table folded | `seeded.json`, `sanity_s04.json` |
| 8 | Shown first, under the headline: replay of recorded runs for the chosen setting (its picker stays in sync with view 1) (play/pause, 1×/2×/4×, restart; counters, draft-vs-final accuracy, run log). Unfinished settings show `pending (n/N)` and are not replayed | `runs.json` |
| 2b | Compare two finished settings: success of each with CI, paired change B − A with CI and sign test, and the runs right only with A or only with B (each opens in the run table) | `pairs.json`, `summary.json` |
| 9 | Folded behind "Browse all N runs" (opens by itself when a run is opened from another view): every run, filterable by setting, task and outcome; human-review runs in their own table with their route; open a run for its draft, verdicts, issues, calculator use, revision, final answer and failed checks | `runs.json`, `runs/<run_id>.json` |

Design notes: unfinished settings show `pending (n/N)` everywhere and never a number; every rate prints its count and interval next to it; tasks with scoring under review (T06-v003) get a † note beside every figure they affect; every chart has an `aria-label`, printed values and a tooltip on hover and keyboard focus; the setting picker is a radio group (arrow keys); heatmap rows are buttons with `aria-expanded`; the layout works at 400 px with no sideways page scroll (wide tables scroll inside their box); `prefers-reduced-motion` turns off transitions.

## Regenerate the data

```bash
python dashboard/build_data.py                                   # results/ -> dashboard/data/
python dashboard/build_data.py --results path/to/results         # e.g. results extracted from study-results
```

`dashboard/data/` is **gitignored on `main`** (it is derived from the study results, which live on the `study-results` branch). Re-run the script after every batch; it replaces the folder.

To get the results on a fresh clone first:

```bash
git fetch origin study-results
git archive origin/study-results results | tar -x
python dashboard/build_data.py
```

## What the script writes

| File | Contents | Built from |
|---|---|---|
| `meta.json` | Study settings (model, temperature, top-p, tasks × repeats), each setting's progress, which settings are complete, tasks whose scoring is under review | `results.jsonl`, `data/study_tasks.txt` |
| `summary.json` | Headline sentence; per setting: task success with 95% CI and counts, success without human review, human-review routes, first-draft success, kept / fixed / broken / still wrong, catch rate and false-alarm rate with counts, MCC, verifier calls, cost and tokens; paired comparisons vs `none` | `metrics.summarize`, `metrics.compare_conditions` (the same code as `scripts/analyze.py`) |
| `heatmap.json` | Right final answers out of saved runs per task and setting, grouped under S01–S10 with the variants as sub-rows | `results.jsonl` |
| `spend.json` | Total spend vs the $20 cap; by setting, model and day | `runs.jsonl` (every call once, as `scripts/spend.py`) |
| `seeded.json` | Seeded-set catch rate, false alarms and "caught with the right reason" per setting, and every seeded review's verdict and decision | `results/seeded/`, `right_reason.py` |
| `sanity_s04.json` | The pre-study verifier checks on a known-wrong and a known-right S04 answer | `runs.jsonl` (`*-check` calls) |
| `runs.json` | One row per run, in the order of each run's first logged call (for the replay and the run table) | `results.jsonl`, `runs.jsonl` |
| `runs/<run_id>.json` | One run's draft, reviews (issues, reason, calculator use), revision, final answer, and the checks it failed | `results/answers/` |

## Rules the script enforces (and the tests check)

- **Pending, never a number.** A setting is complete only when every study run is saved and none failed. Unfinished settings are exported as `{"status": "pending", "done": 54, "total": 96, "label": "pending (54/96)"}` with no rates; the same applies to each setting's seeded reviews (31 answers).
- **No answer key.** The checker's `expected` values and the seeded labels' `correct_value` / `wrong_value` are dropped from every file, at any depth. Failed checks show only what the answer reported (`observed`) and the reason code (e.g. `value_mismatch`). The seeded labels contribute only the right-reason decision.
- **No account identifiers.** API keys, Groq organization IDs, e-mail addresses and home-folder paths are redacted from every string (e.g. Groq's rate-limit errors carry the organization ID).
- **Sample sizes everywhere.** Every rate comes with its counts (for example `{"value": 1.0, "caught": 11, "n": 11}`) and task success with its 95% bootstrap CI over the 32 tasks.
- **Spend** is the total of `runs.jsonl`. A run's own `cost_usd` includes its share of the shared first draft, so per-run costs are shown as recorded and labelled that way.
- **Human review** has three routes: an abstain, an unreadable verdict, or a reject whose revision the second check did not accept. Task success is reported both with and without those runs.
- **T06-v003** is graded as the checker has it, with a "scoring under review" note (`SCORING_UNDER_REVIEW` in `src/advisory/dashboard_export.py`).

Tests: `tests/test_dashboard_export.py` builds from a tiny fixture that contains answer-key values and account identifiers on purpose, and fails if any of them reach the output.

## Colours

All colours are CSS variables at the top of `style.css` (palette "midnight teal + amber"): `--right` (teal) for right and fixed, `--wrong` (red) for wrong and broken, `--pending` (amber) for pending and human review, `--accent-from`/`--accent-to` (teal → blue) for highlights, and one `--c-<setting>` colour per setting (slate, sky, indigo, teal, rose, lavender). `app.js` and `replay.js` read these variables, so a palette change is one edit. Every text colour has at least 4.5:1 contrast on every background.

## Mockup

`mockup/dashboard_figma.svg` (and a PNG render) is the original concept design (violet to cyan on near-black); the page has since moved to the midnight teal palette above. Its numbers are placeholders from the 10 × 5 pilot baseline; the page will show the main study (32 tasks × 3 runs).

The top of the page shows when the data was built ("Data as of …") and a progress bar per setting, so a setting still running is never quoted as a result.

## Motion

Chart bars grow out of the zero line, intervals open, points fade in, and heatmap and seeded-grid cells fill in one setting at a time, once, when each part first scrolls into view. KPI numbers count up when the setting changes. The printed values never move, nothing animates with `prefers-reduced-motion`, and print/PDF always shows the finished state.

## Slides and report

- **PNG:** every chart has a PNG button. It downloads the chart at 2× with its caption and a "32 tasks × 3 runs · data as of …" footer.
- **PDF:** "Save as PDF" (or the browser's print) lays the page out as 16:10 pages, one view per page, dark theme kept; the replay and the run table are left out.
- **Replay clip:** `python dashboard/record_replay.py --setting dedicated --seconds 40 --out replay.webp` records the replay panel as an animated WebP (or `.gif`) in headless Chrome. It refuses settings that are still pending.

## Replay module

`replay.js` is the team's reference module (`assets/replay.js`), adapted to the real data and the page:

- `normalize()` maps the real field values: `verdict` is `accept` / `reject` / `abstain` / `null` (`null` with a verifier means the reply could not be read and gets an UNREADABLE stamp); `status` `failed` runs are not replayed; `human_review_route` labels the amber "?" runs in the log.
- Colours are read from the page's CSS variables (`--cyan`, `--pink`, `--amber`, …), so the replay and the rest of the page match.
- The play, speed and restart buttons are real `<button>`s with `aria-pressed`. With `prefers-reduced-motion` the finished state is shown and nothing moves until you press Play.
- Below 560 px it switches to a stacked layout (task → advisor → verifier → final answer → checker, top to bottom) so the diagram stays readable on phones; it re-mounts when the window crosses that width.

The replay's "Final accuracy" counts every replayed run, human-review runs included with the final answer as recorded (the same as "all runs counted" in view 2).

## Tests

- `tests/test_dashboard_export.py`: the data script, from a tiny fixture with planted answer-key values and account identifiers.
- `tests/test_dashboard_browser.py`: builds the data from the real `results/`, serves the page and drives headless Chrome. It checks that the replay mounts for every finished setting with no JavaScript errors, that unfinished settings show `pending (n/N)`, that reduced motion shows the finished state, and that opening run details never brings an `expected` key into the page (neither in the fetched JSON nor in the DOM). It is skipped without Chrome, the `websockets` package or `results/` (set `DASHBOARD_TEST_RESULTS` to point it at another results folder).
