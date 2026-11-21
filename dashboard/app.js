/* CS 5130 verification study dashboard. Display only: every number comes from data/*.json,
   which dashboard/build_data.py derives from results/. Nothing here calls a model. */
(() => {
  "use strict";

  const ORDER = ["none", "self", "peer", "dedicated", "weakened", "cross_family"];
  const VERIFIED = ORDER.slice(1);
  const SHORT = {
    none: "None", self: "Self", peer: "Peer", dedicated: "Dedicated",
    weakened: "No calculator", cross_family: "Cross-family",
  };
  // Every colour comes from the CSS variables in style.css, so the palette lives in one place.
  const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
  const PINK = css("--wrong"), CYAN = css("--right"), AMBER = css("--pending");
  const VIOLET = css("--accent-to"), ACCENT_FROM = css("--accent-from");
  const BG = css("--bg"), TEXT = css("--text"), MUTED = css("--muted"), FAINT = css("--faint"), TRACK = css("--line-soft");
  const color = (c) => css(`--c-${c}`) || MUTED;
  const reduceMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  const state = { data: null, selected: null };

  // ------------------------------------------------------------------ formatting

  const pct = (v, d = 1) => (v == null ? "–" : `${(v * 100).toFixed(d)}%`);
  const ci = (lo, hi) => (lo == null || hi == null ? "" : `95% CI ${pct(lo)}–${pct(hi)}`);
  const usd = (v, d = 2) => (v == null ? "–" : `$${v.toFixed(d)}`);
  const pts = (v) => (v == null ? "–" : `${v >= 0 ? "+" : "−"}${Math.abs(v * 100).toFixed(1)} pts`);
  const kfmt = (v) => (v == null ? "–" : v >= 1000 ? `${(v / 1000).toFixed(1)}k` : `${Math.round(v)}`);
  const pval = (p) => (p == null ? "–" : p < 0.001 ? "p < 0.001" : `p = ${p.toFixed(3)}`);
  const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (ch) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[ch]));
  const isPending = (entry) => !entry || entry.status !== "complete";
  const pendingLabel = (entry) => (entry && entry.label) || "pending";

  // ------------------------------------------------------------------ tooltip

  const tip = document.getElementById("tooltip");
  function showTip(html, x, y) {
    tip.innerHTML = html;
    tip.hidden = false;
    const r = tip.getBoundingClientRect();
    const left = Math.min(Math.max(8, x + 14), window.innerWidth - r.width - 8);
    const top = y + 14 + r.height > window.innerHeight ? y - r.height - 10 : y + 14;
    tip.style.left = `${left}px`;
    tip.style.top = `${Math.max(8, top)}px`;
  }
  const hideTip = () => { tip.hidden = true; };
  function attachTip(selection, html) {
    selection
      .attr("tabindex", 0)
      .on("mouseenter mousemove", (event, d) => showTip(html(d), event.clientX, event.clientY))
      .on("mouseleave blur", hideTip)
      .on("focus", function (event, d) {
        const b = this.getBoundingClientRect();
        showTip(html(d), b.left + b.width / 2, b.top + b.height / 2);
      });
  }
  window.addEventListener("scroll", hideTip, { passive: true });
  document.addEventListener("keydown", (e) => { if (e.key === "Escape") hideTip(); });

  // ------------------------------------------------------------------ data

  async function loadAll() {
    const names = ["meta", "summary", "heatmap", "spend", "seeded", "sanity_s04", "runs", "pairs"];
    const files = await Promise.all(names.map(async (n) => {
      const res = await fetch(`data/${n}.json`, { cache: "no-store" });
      if (!res.ok) throw new Error(`data/${n}.json: HTTP ${res.status}`);
      return res.json();
    }));
    return Object.fromEntries(names.map((n, i) => [n, files[i]]));
  }

  function slotWidth(el) {
    return Math.max(280, Math.floor(el.getBoundingClientRect().width));
  }

  function reviewNoteText() {
    const notes = state.data.meta.scoring_under_review || {};
    return Object.entries(notes).map(([task, text]) => `${task}: ${text}`).join(" ");
  }

  // ------------------------------------------------------------------ gauge

  function gauge(container, { value, big, label, detail, size = 300, pending = false, ariaLabel }) {
    const w = size, h = detail ? size * 0.92 : size * 0.8, r = size * 0.4, thick = Math.max(10, size * 0.06);
    const start = -0.75 * Math.PI, end = 0.75 * Math.PI;
    const id = `g${Math.random().toString(36).slice(2, 8)}`;
    const svg = d3.select(container).append("svg")
      .attr("viewBox", `0 0 ${w} ${h}`)
      .attr("role", "img")
      .attr("aria-label", ariaLabel);
    const defs = svg.append("defs");
    const grad = defs.append("linearGradient").attr("id", id).attr("x1", 0).attr("y1", 1).attr("x2", 1).attr("y2", 0);
    grad.append("stop").attr("offset", 0).attr("stop-color", ACCENT_FROM);
    grad.append("stop").attr("offset", 1).attr("stop-color", VIOLET);
    const glow = defs.append("filter").attr("id", `${id}f`).attr("x", "-30%").attr("y", "-30%").attr("width", "160%").attr("height", "160%");
    glow.append("feGaussianBlur").attr("stdDeviation", size * 0.02).attr("result", "b");
    const merge = glow.append("feMerge");
    merge.append("feMergeNode").attr("in", "b");
    merge.append("feMergeNode").attr("in", "SourceGraphic");
    const g = svg.append("g").attr("transform", `translate(${w / 2},${r + thick})`);
    const arc = d3.arc().innerRadius(r - thick / 2).outerRadius(r + thick / 2).cornerRadius(thick / 2);
    g.append("path").attr("d", arc({ startAngle: start, endAngle: end })).attr("fill", TRACK);
    if (!pending && value != null) {
      g.append("path").attr("d", arc({ startAngle: start, endAngle: start + (end - start) * value }))
        .attr("fill", `url(#${id})`).attr("filter", `url(#${id}f)`);
    }
    g.append("text").attr("text-anchor", "middle").attr("y", size * 0.04)
      .attr("fill", pending ? AMBER : TEXT).attr("font-weight", 700)
      .attr("font-size", pending ? size * 0.075 : size * 0.18).text(big);
    g.append("text").attr("text-anchor", "middle").attr("y", size * 0.14).attr("fill", MUTED)
      .attr("font-size", size * 0.05).text(label);
    if (detail) {
      g.append("text").attr("text-anchor", "middle").attr("y", size * 0.37).attr("fill", MUTED)
        .attr("font-size", size * 0.046).text(detail);
    }
    return svg;
  }

  // ------------------------------------------------------------------ motion: once, when scrolled into view

  const seenAnim = new Set();
  const animObserver = "IntersectionObserver" in window ? new IntersectionObserver((entries) => {
    entries.forEach((e) => {
      if (!e.isIntersecting) return;
      animObserver.unobserve(e.target);
      requestAnimationFrame(() => requestAnimationFrame(() => e.target.classList.remove("pending")));
    });
  }, { threshold: 0.15 }) : null;
  // Marks start hidden (class "pending") and play once; redraws after that (resize, print) appear at once.
  function animateWhenVisible(node, key) {
    if (!node || reduceMotion || !animObserver || seenAnim.has(key)) return;
    seenAnim.add(key);
    node.classList.add("anim", "pending");
    animObserver.observe(node);
  }

  // KPI numbers count up to their value when the setting changes. The final text is what was printed.
  const COUNT_FMT = { pct0: (v) => `${Math.round(v)}%`, pct1: (v) => `${v.toFixed(1)}%`, usd2: (v) => `$${v.toFixed(2)}`, usd3: (v) => `$${v.toFixed(3)}`, int: (v) => `${Math.round(v)}` };
  const num = (value, fmt) => `<span data-count="${value}" data-fmt="${fmt}">${COUNT_FMT[fmt](value)}</span>`;
  function countUp(scope) {
    if (reduceMotion) return;
    scope.querySelectorAll("[data-count]").forEach((el) => {
      const target = +el.dataset.count, fmt = COUNT_FMT[el.dataset.fmt];
      d3.select(el).transition().duration(650).ease(d3.easeCubicOut)
        .tween("text", () => (t) => { el.textContent = fmt(target * t); })
        .on("end interrupt", () => { el.textContent = fmt(target); });
    });
  }

  // ------------------------------------------------------------------ 1. hero + explore

  function renderHero() {
    const { summary, meta } = state.data;
    const s = meta.study;
    document.getElementById("headline").textContent = summary.headline.text;
    // One short line: data set, advisor, when the data was built, and anything still running.
    const asof = new Date(meta.generated_at);
    const when = isNaN(asof) ? meta.generated_at
      : `${asof.toLocaleDateString("en-US", { month: "short", day: "numeric", timeZone: "UTC" })}, ${asof.toISOString().slice(11, 16)} UTC`;
    const running = meta.conditions.filter((c) => !c.complete)
      .map((c) => `<span class="pending">${esc(SHORT[c.id])} still running (${c.runs_done}/${c.runs_expected})</span>`);
    document.getElementById("dataset").innerHTML =
      [`<strong>${s.n_tasks} tasks × ${s.repeats} runs</strong>`, `advisor ${esc(s.advisor_model)}`, `data as of ${esc(when)}`, ...running].join(" · ");
    const note = document.getElementById("headline-note");
    const tasks = Object.keys(meta.scoring_under_review || {});
    if (tasks.length) {
      note.hidden = false;
      note.textContent = `${tasks.length === 1 ? "One task's" : `${tasks.length} tasks'`} scoring is under review (${tasks.join(", ")}); details beside each affected figure.`;
    }

    const none = summary.conditions.none;
    const slot = document.querySelector("#hero-gauge .gauge-slot");
    slot.innerHTML = "";
    if (isPending(none)) {
      gauge(slot, { pending: true, big: pendingLabel(none), label: "No-check baseline", ariaLabel: `No-check baseline: ${pendingLabel(none)}` });
    } else {
      const t = none.task_success;
      gauge(slot, {
        value: t.value, big: pct(t.value, 0), label: "Baseline accuracy (no check)",
        detail: `${t.right} of ${t.n} runs · ${ci(t.ci_low, t.ci_high)}`,
        ariaLabel: `No-check baseline accuracy ${pct(t.value)}, ${t.right} of ${t.n} runs, ${ci(t.ci_low, t.ci_high)}`,
      });
    }
  }

  // The same setting picker appears twice (above the replay and in "Explore a setting"); both stay in sync.
  const PICKERS = ["replay-pills", "pills"];
  function renderPills() {
    const { summary } = state.data;
    PICKERS.forEach((id) => {
      const box = document.getElementById(id);
      if (!box) return;
      box.innerHTML = "";
      ORDER.forEach((c) => {
        const entry = summary.conditions[c];
        const b = document.createElement("button");
        b.type = "button";
        b.className = "pill";
        b.setAttribute("role", "radio");
        b.setAttribute("aria-checked", String(c === state.selected));
        b.tabIndex = c === state.selected ? 0 : -1;
        b.dataset.condition = c;
        b.innerHTML = esc(SHORT[c]) + (isPending(entry) ? `<span class="tag">${esc(pendingLabel(entry))}</span>` : "");
        b.addEventListener("click", () => select(c, id));
        box.appendChild(b);
      });
      box.onkeydown = (e) => {
        const i = ORDER.indexOf(state.selected);
        const step = { ArrowRight: 1, ArrowDown: 1, ArrowLeft: -1, ArrowUp: -1 }[e.key];
        if (step) { e.preventDefault(); select(ORDER[(i + step + ORDER.length) % ORDER.length], id); }
        if (e.key === "Home") { e.preventDefault(); select(ORDER[0], id); }
        if (e.key === "End") { e.preventDefault(); select(ORDER[ORDER.length - 1], id); }
      };
    });
  }

  // focusIn: id of the picker to keep keyboard focus in (or a truthy value for the first one).
  function select(c, focusIn) {
    state.selected = c;
    renderPills();
    renderKpis();
    countUp(document.getElementById("kpis"));
    renderReplay();
    if (focusIn) {
      const box = document.getElementById(typeof focusIn === "string" ? focusIn : PICKERS[1]);
      box.querySelector(`.pill[data-condition="${c}"]`).focus();
    }
  }

  function tile(cls, k, v, d, vClass = "") {
    return `<div class="kpi ${cls}"><p class="k">${k}</p><p class="v ${vClass}">${v}</p>${d ? `<p class="d">${d}</p>` : ""}</div>`;
  }

  function renderKpis() {
    const { summary, spend } = state.data;
    const c = state.selected;
    const e = summary.conditions[c];
    const box = document.getElementById("kpis");
    const spendTile = tile("",
      "Total spend (all settings)",
      `${usd(spend.total_usd)}`,
      `of the $${spend.budget_usd.toFixed(0)} cap · ${spend.calls.toLocaleString()} model calls`);
    if (isPending(e)) {
      const p = `<span class="pending">${esc(pendingLabel(e))}</span>`;
      box.innerHTML =
        tile("", "Task success", p, "runs still in progress", "small") +
        tile("", "Catch rate", p, "", "small") +
        tile("", "False alarms", p, "", "small") +
        tile("", "Fixed vs broken", p, "", "small") +
        tile("", "Cost per run", p, "", "small") + spendTile;
      return;
    }
    const t = e.task_success;
    const v = e.verifier;
    const o = e.outcomes;
    const review = c !== "none" ? "" : "";
    box.innerHTML =
      `<div class="kpi" id="kpi-success"><p class="k">Task success</p><div class="mini"></div>` +
      `<p class="d">${t.right}/${t.n} runs · ${ci(t.ci_low, t.ci_high)}${review}</p></div>` +
      (v
        ? tile("", "Catch rate", num(v.catch_rate.value * 100, "pct0"), `${v.catch_rate.caught}/${v.catch_rate.n} wrong drafts rejected · MCC ${v.mcc == null ? "–" : v.mcc.toFixed(2)}`, "good") +
          tile("", "False alarms", num(v.false_alarm_rate.value * 100, "pct1"), `${v.false_alarm_rate.flagged}/${v.false_alarm_rate.n} right drafts rejected`, v.false_alarm_rate.value > 0 ? "bad" : "")
        : tile("", "Catch rate", "–", "no verifier in this setting", "small") + tile("", "False alarms", "–", "no verifier in this setting", "small")) +
      tile("", "Fixed vs broken",
        `<span class="ok">+${num(o.fixed, "int")}</span> <span class="tag-small">/</span> <span class="no">−${num(o.broken, "int")}</span>`,
        `${o.fixed} wrong → right · ${o.broken} right → wrong (of ${e.n_runs} runs)`) +
      tile("", "Cost per run", num(e.cost.mean_cost_usd, "usd3"),
        `${kfmt(e.cost.mean_tokens_in)} in / ${kfmt(e.cost.mean_tokens_out)} out tokens · includes the shared draft`) +
      spendTile;
    gauge(box.querySelector("#kpi-success .mini"), {
      value: t.value, big: pct(t.value, 1), label: SHORT[c], size: 200,
      ariaLabel: `${SHORT[c]} task success ${pct(t.value)}, ${t.right} of ${t.n} runs, ${ci(t.ci_low, t.ci_high)}`,
    });
  }

  // ------------------------------------------------------------------ shared row chart

  // Rows of: name (left) and a printed value (right) on one line, a mark on the line below.
  function rowChart(slot, { rows, domain, height = 46, zero = null, axisFormat, ariaLabel, mark, valueText, tipHtml, tickCount = 5 }) {
    slot.innerHTML = "";
    const w = slotWidth(slot), m = { l: 14, r: 18, t: 4, b: 26 };
    const h = m.t + rows.length * height + m.b;
    const svg = d3.select(slot).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("role", "group").attr("aria-label", ariaLabel);
    const x = d3.scaleLinear().domain(domain).range([m.l, w - m.r]).nice();
    svg.append("g").attr("class", "grid").attr("transform", `translate(0,${h - m.b})`)
      .call(d3.axisBottom(x).ticks(tickCount).tickSize(-(h - m.b - m.t)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`)
      .call(d3.axisBottom(x).ticks(tickCount).tickFormat(axisFormat));
    if (zero != null) {
      svg.append("line").attr("x1", x(zero)).attr("x2", x(zero)).attr("y1", m.t).attr("y2", h - m.b)
        .attr("stroke", MUTED).attr("stroke-dasharray", "4 4").attr("stroke-width", 1);
    }
    const row = svg.selectAll("g.row").data(rows).join("g").attr("class", "row mark")
      .attr("transform", (d, i) => `translate(0,${m.t + i * height})`)
      .attr("role", "img")
      .attr("aria-label", (d) => `${SHORT[d.condition]}: ${valueText(d)}`);
    row.append("rect").attr("class", "focus-ring").attr("x", 0).attr("y", 0).attr("width", w).attr("height", height - 4)
      .attr("rx", 6).attr("fill", "transparent").attr("stroke", "transparent");
    row.append("text").attr("class", "lbl-strong").attr("x", m.l).attr("y", 14).text((d) => SHORT[d.condition]);
    row.append("text").attr("x", w - m.r).attr("y", 14).attr("text-anchor", "end")
      .attr("class", (d) => (d.pending ? "lbl-pending" : "lbl-muted")).text(valueText);
    row.each(function (d) { if (!d.pending) mark(d3.select(this), d, x, 28); });
    attachTip(row, tipHtml);
    // Bars grow out of the zero line, interval lines open from their centre, points fade in.
    const origin = x(zero != null ? zero : x.domain()[0]);
    svg.selectAll("g.row rect:not(.focus-ring)").each(function () {
      if (this.getAttribute("transform")) this.classList.add("anim-pt");
      else { this.classList.add("anim-bar"); this.style.transformOrigin = `${origin}px 0px`; }
    });
    svg.selectAll("g.row line").each(function () {
      const cx = (+this.getAttribute("x1") + +this.getAttribute("x2")) / 2;
      this.classList.add("anim-ci"); this.style.transformOrigin = `${cx}px 0px`;
    });
    svg.selectAll("g.row circle").classed("anim-pt", true);
    animateWhenVisible(svg.node(), (slot.closest("figure") || slot).id || ariaLabel);
    return { svg, x };
  }

  // ------------------------------------------------------------------ 2. comparison

  function renderCompare() {
    const { summary } = state.data;
    document.getElementById("interval-note").textContent = summary.interval_note;
    const rows = ORDER.map((c) => {
      const e = summary.conditions[c];
      return isPending(e) ? { condition: c, pending: true, label: pendingLabel(e) } : { condition: c, ...e.task_success };
    });
    const done = rows.filter((r) => !r.pending);
    const lo = Math.min(...done.map((r) => r.ci_low ?? r.value));
    const base = summary.conditions.none;
    rowChart(document.querySelector("#success-chart .chart-slot"), {
      rows,
      domain: [Math.max(0, Math.floor((lo - 0.05) * 10) / 10), 1],
      zero: isPending(base) ? null : base.task_success.value,
      axisFormat: (v) => `${Math.round(v * 100)}%`,
      ariaLabel: "Task success per setting with 95% intervals over tasks; the dashed line is the no-check baseline.",
      valueText: (d) => (d.pending ? d.label : `${pct(d.value)} (${d.right}/${d.n}) · ${pct(d.ci_low, 0)}–${pct(d.ci_high, 0)}`),
      mark: (g, d, x, y) => {
        g.append("line").attr("x1", x(d.ci_low)).attr("x2", x(d.ci_high)).attr("y1", y).attr("y2", y)
          .attr("stroke", color(d.condition)).attr("stroke-width", 3).attr("stroke-linecap", "round").attr("opacity", 0.55);
        g.append("circle").attr("cx", x(d.value)).attr("cy", y).attr("r", 6).attr("fill", color(d.condition))
          .attr("stroke", BG).attr("stroke-width", 2);
      },
      tipHtml: (d) => (d.pending ? `<strong>${SHORT[d.condition]}</strong><br>${esc(d.label)} runs saved; no numbers until it finishes.`
        : `<strong>${SHORT[d.condition]}</strong><br>Task success ${pct(d.value)}<br>${d.right} of ${d.n} runs right<br>${ci(d.ci_low, d.ci_high)} (bootstrap over tasks)`),
    });

    const comps = Object.fromEntries(summary.comparisons.map((c) => [c.condition, c]));
    const prow = VERIFIED.map((c) => {
      const e = summary.conditions[c];
      if (isPending(e) || !comps[c]) return { condition: c, pending: true, label: pendingLabel(e) };
      return { condition: c, ...comps[c] };
    });
    const pdone = prow.filter((r) => !r.pending);
    const span = Math.max(0.05, ...pdone.map((r) => Math.max(Math.abs(r.ci_low), Math.abs(r.ci_high))));
    rowChart(document.querySelector("#paired-chart .chart-slot"), {
      rows: prow,
      domain: [-Math.min(span, 0.1) - 0.02, span + 0.02],
      zero: 0,
      axisFormat: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}`,
      ariaLabel: "Change in task success against no check, in percentage points, with 95% intervals.",
      valueText: (d) => (d.pending ? d.label : `${pts(d.delta)} · ${pts(d.ci_low).replace(" pts", "")} to ${pts(d.ci_high)} · ${pval(d.sign_test_p)}`),
      mark: (g, d, x, y) => {
        g.append("line").attr("x1", x(d.ci_low)).attr("x2", x(d.ci_high)).attr("y1", y).attr("y2", y)
          .attr("stroke", color(d.condition)).attr("stroke-width", 3).attr("stroke-linecap", "round").attr("opacity", 0.55);
        g.append("rect").attr("x", x(d.delta) - 6).attr("y", y - 6).attr("width", 12).attr("height", 12)
          .attr("transform", `rotate(45 ${x(d.delta)} ${y})`).attr("fill", color(d.condition)).attr("stroke", BG).attr("stroke-width", 2);
      },
      tipHtml: (d) => (d.pending ? `<strong>${SHORT[d.condition]}</strong><br>${esc(d.label)}: no comparison until it finishes.`
        : `<strong>${SHORT[d.condition]} vs none</strong><br>${pts(d.delta)} (95% CI ${pts(d.ci_low)} to ${pts(d.ci_high)})<br>` +
          `${d.n_pairs} paired runs over ${d.n_tasks} tasks<br>Right only here: ${d.runs_only_condition_right} runs; right only without a check: ${d.runs_only_none_right}<br>` +
          `Better on ${d.tasks_better} tasks, worse on ${d.tasks_worse} · sign test ${pval(d.sign_test_p)}`),
    });

    const t = document.getElementById("hr-table");
    const head = `<thead><tr><th>Setting</th><th class="num">All runs counted</th><th class="num">Sent to human review</th><th>Why (abstain · unreadable · revision not accepted)</th><th class="num">Not counting human review</th></tr></thead>`;
    const body = ORDER.map((c) => {
      const e = summary.conditions[c];
      if (isPending(e)) return `<tr><td>${SHORT[c]}</td><td class="pending" colspan="4">${esc(pendingLabel(e))}</td></tr>`;
      const a = e.task_success, b = e.task_success_excl_human_review, h = e.human_review;
      return `<tr><td>${SHORT[c]}</td>` +
        `<td class="num">${pct(a.value)} <span class="tag-small">(${a.right}/${a.n})</span><span class="ci">${ci(a.ci_low, a.ci_high)}</span></td>` +
        `<td class="num">${h.n} <span class="tag-small">of ${e.n_runs}</span><span class="ci">${h.final_right} of them right</span></td>` +
        `<td>${h.routes.abstain} · ${h.routes.unreadable} · ${h.routes.revision_not_accepted}</td>` +
        `<td class="num">${pct(b.value)} <span class="tag-small">(${b.right}/${b.n})</span><span class="ci">${ci(b.ci_low, b.ci_high)}</span></td></tr>`;
    }).join("");
    t.innerHTML = head + `<tbody>${body}</tbody>`;
  }

  // ------------------------------------------------------------------ 3. catches

  function renderCatch() {
    const { summary } = state.data;
    const rows = VERIFIED.map((c) => {
      const e = summary.conditions[c];
      return isPending(e) ? { condition: c, pending: true, label: pendingLabel(e) } : { condition: c, ...e.verifier, n_runs: e.n_runs };
    });
    const slot = document.querySelector("#catch-chart .chart-slot");
    rowChart(slot, {
      rows,
      height: 62,
      domain: [0, 1],
      axisFormat: (v) => `${Math.round(v * 100)}%`,
      ariaLabel: "Catch rate (wrong drafts rejected) and false-alarm rate (right drafts rejected) per setting.",
      valueText: (d) => (d.pending ? d.label : `MCC ${d.mcc == null ? "–" : d.mcc.toFixed(2)}`),
      mark: (g, d, x, y) => {
        const bar = (yy, v, fill, text) => {
          g.append("rect").attr("x", x(0)).attr("y", yy - 6).attr("height", 12).attr("rx", 3)
            .attr("width", Math.max(2, x(v) - x(0))).attr("fill", fill);
          const tx = x(v) + 6, inside = tx > x.range()[1] - 150;
          g.append("text").attr("x", inside ? x(v) - 6 : tx).attr("y", yy + 4).attr("text-anchor", inside ? "end" : "start")
            .attr("class", "lbl").style("fill", inside ? BG : null).style("font-weight", inside ? 700 : null).text(text);
        };
        bar(y, d.catch_rate.value, CYAN, `caught ${pct(d.catch_rate.value, 0)} (${d.catch_rate.caught}/${d.catch_rate.n})`);
        bar(y + 18, d.false_alarm_rate.value, PINK, `false alarms ${pct(d.false_alarm_rate.value)} (${d.false_alarm_rate.flagged}/${d.false_alarm_rate.n})`);
      },
      tipHtml: (d) => (d.pending ? `<strong>${SHORT[d.condition]}</strong><br>${esc(d.label)}`
        : `<strong>${SHORT[d.condition]}</strong><br>Catch rate ${pct(d.catch_rate.value)}: rejected ${d.catch_rate.caught} of ${d.catch_rate.n} wrong drafts<br>` +
          `False alarms ${pct(d.false_alarm_rate.value)}: rejected ${d.false_alarm_rate.flagged} of ${d.false_alarm_rate.n} right drafts<br>` +
          `MCC ${d.mcc == null ? "–" : d.mcc.toFixed(2)}${d.missing_verdicts ? `<br>${d.missing_verdicts} unreadable verdict(s), counted as not rejected` : ""}<br>` +
          `Mean verifier calls per run: ${d.mean_verifier_calls == null ? "–" : d.mean_verifier_calls.toFixed(1)}`),
    });
    const small = summary.conditions.none;
    if (!isPending(small)) {
      const wrong = small.n_runs - small.first_draft_success.right;
      slot.insertAdjacentHTML("beforeend", `<p class="note">Only ${wrong} of ${small.n_runs} first drafts were wrong, so each catch rate rests on ${wrong} drafts: one more or fewer catch moves it by ${(100 / wrong).toFixed(0)} points.</p>`);
    }
  }

  // ------------------------------------------------------------------ 4. fixed vs broken

  function renderFix() {
    const { summary } = state.data;
    const rows = ORDER.map((c) => {
      const e = summary.conditions[c];
      return isPending(e) ? { condition: c, pending: true, label: pendingLabel(e) } : { condition: c, ...e.outcomes, n: e.n_runs };
    });
    const max = Math.max(3, ...rows.filter((r) => !r.pending).map((r) => Math.max(r.fixed, r.broken)));
    rowChart(document.querySelector("#fix-chart .chart-slot"), {
      rows,
      domain: [-max, max],
      zero: 0,
      axisFormat: (v) => `${Math.abs(v)}`,
      tickCount: Math.min(8, max * 2),
      ariaLabel: "Runs broken by the revision (left, pink) and fixed (right, cyan) per setting.",
      valueText: (d) => (d.pending ? d.label : d.condition === "none" ? "no review, nothing revised" : `fixed ${d.fixed} · broken ${d.broken} · net ${d.fixed - d.broken >= 0 ? "+" : "−"}${Math.abs(d.fixed - d.broken)}`),
      mark: (g, d, x, y) => {
        if (d.broken) g.append("rect").attr("x", x(-d.broken)).attr("y", y - 7).attr("width", x(0) - x(-d.broken)).attr("height", 14).attr("rx", 3).attr("fill", PINK);
        if (d.fixed) g.append("rect").attr("x", x(0)).attr("y", y - 7).attr("width", x(d.fixed) - x(0)).attr("height", 14).attr("rx", 3).attr("fill", CYAN);
        if (d.broken) g.append("text").attr("x", x(-d.broken) - 6).attr("y", y + 4).attr("text-anchor", "end").attr("class", "lbl").text(`−${d.broken}`);
        if (d.fixed) g.append("text").attr("x", x(d.fixed) + 6).attr("y", y + 4).attr("class", "lbl").text(`+${d.fixed}`);
        if (!d.fixed && !d.broken) g.append("text").attr("x", x(0) + 6).attr("y", y + 4).attr("class", "lbl-muted").text("0");
      },
      tipHtml: (d) => (d.pending ? `<strong>${SHORT[d.condition]}</strong><br>${esc(d.label)}`
        : `<strong>${SHORT[d.condition]}</strong> (${d.n} runs)<br><span class="ok">Fixed ${d.fixed}</span>: wrong draft, right final<br>` +
          `<span class="no">Broken ${d.broken}</span>: right draft, wrong final<br>Kept right ${d.kept_correct} · still wrong ${d.still_wrong}`),
    });
  }

  // ------------------------------------------------------------------ 5. heatmap

  const heatColor = d3.scaleLinear().domain([0, 0.5, 1]).range([PINK, VIOLET, CYAN]).interpolate(d3.interpolateRgb); // wrong → middle → right

  function chip(cell, where, col = 0) {
    if (!cell || cell.status === "pending") {
      return `<span class="chip pending-chip" aria-label="${esc(where)}: pending">pending</span>`;
    }
    if (!cell.n) return `<span class="chip pending-chip" aria-label="${esc(where)}: no runs">–</span>`;
    const share = cell.right / cell.n;
    const failed = cell.failed ? ` <small>(+${cell.failed} failed)</small>` : "";
    return `<span class="chip mark-cell" role="img" style="background:${heatColor(share)};--col:${col}" ` +
      `data-tip="${esc(`<strong>${where}</strong><br>${cell.right} of ${cell.n} runs right${cell.failed ? `<br>${cell.failed} failed run(s) not counted` : ""}`)}" ` +
      `aria-label="${esc(`${where}: ${cell.right} of ${cell.n} right`)}">${cell.right}/${cell.n}${failed}</span>`;
  }

  function renderHeatmap() {
    const { heatmap, meta } = state.data;
    const review = meta.scoring_under_review || {};
    const table = document.getElementById("heat-table");
    const head = `<caption class="sr-only">Right final answers per scenario and setting</caption><thead><tr><th scope="col">Scenario</th>` +
      heatmap.conditions.map((c) => {
        const p = heatmap.pending[c];
        return `<th scope="col" class="num" style="text-align:center">${SHORT[c]}${p ? `<br><span class="pending tag-small">${esc(p.label)}</span>` : ""}</th>`;
      }).join("") + `</tr></thead>`;
    const body = heatmap.groups.map((g, gi) => {
      const flagged = g.tasks.some((t) => review[t]);
      const groupCells = heatmap.conditions.map((c, col) => `<td class="cell">${chip(g.cells[c], `${g.scenario} (all ${g.tasks.length} tasks), ${SHORT[c]}`, col)}</td>`).join("");
      const sub = g.rows.map((r) =>
        `<tr class="sub-row" data-group="${gi}" hidden><th scope="row" style="font-weight:500">${esc(r.task_id)}${review[r.task_id] ? ` <span class="dagger" title="${esc(review[r.task_id])}">†</span><span class="sr-only"> (scoring under review)</span>` : ""}</th>` +
        heatmap.conditions.map((c, col) => `<td class="cell">${chip(r.cells[c], `${r.task_id}, ${SHORT[c]}`, col)}</td>`).join("") + `</tr>`).join("");
      return `<tr><th scope="row"><button type="button" class="expander" aria-expanded="false" data-group="${gi}">` +
        `<span class="tri" aria-hidden="true">▶</span>${esc(g.scenario)} <span class="tag-small">${g.tasks.length} task${g.tasks.length > 1 ? "s" : ""}</span>` +
        `${flagged ? ' <span class="dagger" aria-hidden="true">†</span>' : ""}</button></th>${groupCells}</tr>${sub}`;
    }).join("");
    table.innerHTML = head + `<tbody>${body}</tbody>`;
    animateWhenVisible(table, "heat-table");
    table.querySelectorAll(".expander").forEach((btn) => btn.addEventListener("click", () => {
      const open = btn.getAttribute("aria-expanded") !== "true";
      btn.setAttribute("aria-expanded", String(open));
      table.querySelectorAll(`tr.sub-row[data-group="${btn.dataset.group}"]`).forEach((tr) => { tr.hidden = !open; });
    }));
    table.querySelectorAll(".mark-cell").forEach((el) => {
      el.addEventListener("mouseenter", (e) => showTip(el.dataset.tip, e.clientX, e.clientY));
      el.addEventListener("mousemove", (e) => showTip(el.dataset.tip, e.clientX, e.clientY));
      el.addEventListener("mouseleave", hideTip);
      el.addEventListener("blur", hideTip);
      el.addEventListener("focus", () => { const b = el.getBoundingClientRect(); showTip(el.dataset.tip, b.left + b.width / 2, b.bottom); });
    });
    document.getElementById("heat-legend").innerHTML =
      `<span>0 right</span><span class="ramp" style="background:linear-gradient(90deg,${PINK},${VIOLET},${CYAN})"></span><span>all right</span>` +
      `<span>· scenario rows add up their tasks (3 runs each)</span>`;
  }

  // ------------------------------------------------------------------ 6. cost

  function renderCost() {
    const { summary, spend } = state.data;
    document.getElementById("cost-sub").textContent = spend.note;
    const share = Math.min(1, spend.total_usd / spend.budget_usd);
    const top = spend.by_condition.slice(0, 6).map((r) => `${esc(r.name)} ${usd(r.cost_usd)}`).join(" · ");
    document.getElementById("budget").innerHTML =
      `<div class="row"><span><span class="big">${usd(spend.total_usd)}</span> spent of the $${spend.budget_usd.toFixed(0)} cap</span>` +
      `<span>${usd(spend.left_usd)} left · ${spend.calls.toLocaleString()} model calls (${spend.failed_calls} failed)</span></div>` +
      `<div class="bar" role="progressbar" aria-label="Spend against the cap" aria-valuemin="0" aria-valuemax="${spend.budget_usd}" aria-valuenow="${spend.total_usd.toFixed(2)}" aria-valuetext="${usd(spend.total_usd)} of $${spend.budget_usd}">` +
      `<div class="fill" style="width:${(share * 100).toFixed(1)}%"></div></div>` +
      `<div class="row"><span>Largest items: ${top}</span><span>Source: run log, every call counted once</span></div>`;

    const rows = ORDER.map((c) => {
      const e = summary.conditions[c];
      return isPending(e) ? { condition: c, pending: true, label: pendingLabel(e) } : { condition: c, ...e.cost, n: e.n_runs, success: e.task_success };
    });
    const done = rows.filter((r) => !r.pending);
    rowChart(document.querySelector("#cost-chart .chart-slot"), {
      rows,
      domain: [0, Math.max(...done.map((r) => r.mean_cost_usd)) * 1.1],
      axisFormat: (v) => `$${v.toFixed(2)}`,
      ariaLabel: "Mean cost per run per setting, with tokens.",
      valueText: (d) => (d.pending ? d.label : `${usd(d.mean_cost_usd, 3)} · ${kfmt(d.mean_tokens_in)} in / ${kfmt(d.mean_tokens_out)} out`),
      mark: (g, d, x, y) => {
        g.append("rect").attr("x", x(0)).attr("y", y - 6).attr("height", 12).attr("rx", 3)
          .attr("width", Math.max(2, x(d.mean_cost_usd) - x(0))).attr("fill", color(d.condition));
      },
      tipHtml: (d) => (d.pending ? `<strong>${SHORT[d.condition]}</strong><br>${esc(d.label)}`
        : `<strong>${SHORT[d.condition]}</strong> (${d.n} runs)<br>Mean ${usd(d.mean_cost_usd, 4)} per run, total ${usd(d.total_cost_usd)}<br>` +
          `${kfmt(d.mean_tokens_in)} tokens in, ${kfmt(d.mean_tokens_out)} out per run<br>` +
          `${d.extra_cost_per_extra_correct == null ? "" : `${usd(d.extra_cost_per_extra_correct)} per extra right answer vs none<br>`}Includes the shared first draft`),
    });
    renderScatter(done);
  }

  function renderScatter(points) {
    const slot = document.querySelector("#scatter-chart .chart-slot");
    slot.innerHTML = "";
    const w = slotWidth(slot), h = 260, m = { l: 44, r: 16, t: 12, b: 34 };
    const svg = d3.select(slot).append("svg").attr("viewBox", `0 0 ${w} ${h}`).attr("role", "group")
      .attr("aria-label", "Mean cost per run against task success, one point per finished setting, with 95% intervals.");
    const x = d3.scaleLinear().domain(d3.extent(points, (p) => p.mean_cost_usd)).range([m.l, w - m.r]).nice();
    x.domain([x.domain()[0] - (x.domain()[1] - x.domain()[0]) * 0.08, x.domain()[1] + (x.domain()[1] - x.domain()[0]) * 0.08]);
    const ylo = Math.min(...points.map((p) => p.success.ci_low));
    const y = d3.scaleLinear().domain([Math.max(0, Math.floor(ylo * 10) / 10 - 0.05), 1]).range([h - m.b, m.t]);
    svg.append("g").attr("class", "grid").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickSize(-(w - m.l - m.r)).tickFormat(""));
    svg.append("g").attr("class", "axis").attr("transform", `translate(0,${h - m.b})`).call(d3.axisBottom(x).ticks(5).tickFormat((v) => `$${v.toFixed(3)}`));
    svg.append("g").attr("class", "axis").attr("transform", `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5).tickFormat((v) => `${Math.round(v * 100)}%`));
    svg.append("text").attr("class", "lbl-muted").attr("x", w - m.r).attr("y", h - 4).attr("text-anchor", "end").text("mean cost per run");
    const g = svg.selectAll("g.pt").data(points).join("g").attr("class", "pt mark").attr("role", "img")
      .attr("aria-label", (d) => `${SHORT[d.condition]}: ${usd(d.mean_cost_usd, 3)} per run, task success ${pct(d.success.value)}`);
    g.append("line").attr("x1", (d) => x(d.mean_cost_usd)).attr("x2", (d) => x(d.mean_cost_usd))
      .attr("y1", (d) => y(d.success.ci_low)).attr("y2", (d) => y(d.success.ci_high))
      .attr("stroke", (d) => color(d.condition)).attr("stroke-width", 2).attr("opacity", 0.5);
    g.append("circle").attr("cx", (d) => x(d.mean_cost_usd)).attr("cy", (d) => y(d.success.value)).attr("r", 7)
      .attr("fill", (d) => color(d.condition)).attr("stroke", BG).attr("stroke-width", 2);
    // Printed labels, nudged apart vertically so close points stay readable.
    const labels = points.map((p) => ({ p, x: x(p.mean_cost_usd), y: y(p.success.value) - 12 }))
      .sort((a, b) => a.y - b.y);
    // Points closer than ~40px share one label column to the right of the group.
    labels.forEach((l) => { l.x0 = l.x; l.y0 = l.y; });
    labels.forEach((l) => {
      const near = labels.filter((o) => Math.abs(o.x0 - l.x0) < 40 && Math.abs(o.y0 - l.y0) < 40);
      l.x = Math.max(...near.map((o) => o.x0)) + (near.length > 1 ? 8 : 0);
    });
    for (let i = 1; i < labels.length; i++) {
      for (let j = 0; j < i; j++) {
        if (Math.abs(labels[i].x - labels[j].x) < 110 && labels[i].y - labels[j].y < 15) labels[i].y = labels[j].y + 15;
      }
    }
    labels.forEach((l) => {
      const text = `${SHORT[l.p.condition]} ${pct(l.p.success.value)}`;
      const right = l.x + 10 + text.length * 6.6 < w - m.r;
      svg.append("text").attr("class", "lbl").attr("x", l.x + (right ? 10 : -10)).attr("y", l.y)
        .attr("text-anchor", right ? "start" : "end")
        .text(text);
    });
    g.selectAll("line, circle").classed("anim-pt", true);
    animateWhenVisible(svg.node(), "scatter-chart");
    attachTip(g, (d) => `<strong>${SHORT[d.condition]}</strong><br>${usd(d.mean_cost_usd, 4)} per run (includes the shared draft)<br>` +
      `Task success ${pct(d.success.value)} (${d.success.right}/${d.success.n}), ${ci(d.success.ci_low, d.success.ci_high)}`);
  }

  // ------------------------------------------------------------------ 7. sanity checks

  function renderSanity() {
    const { seeded, sanity_s04: s04 } = state.data;

    // One sentence from the S04 cases of the models used in the study; the full table folds away.
    const used = s04.cases.filter((c) => c.used_in_study);
    const NAME = { dedicated: "dedicated", weakened: "no-calculator", cross_family: "cross-family" };
    const who = (c) => (c.setting === "cross_family" ? `cross-family (${c.verifier_model.replace("openai/", "")})` : NAME[c.setting] || c.setting);
    const list = (xs) => (xs.length <= 1 ? xs.join("") : `${xs.slice(0, -1).join(", ")} and ${xs[xs.length - 1]}`);
    const wrong = used.filter((c) => c.answer_was === "wrong");
    const right = used.filter((c) => c.answer_was === "correct");
    const rejected = [...new Set(wrong.filter((c) => c.verdict === "reject").map(who))];
    const accepted = [...new Set(wrong.filter((c) => c.verdict === "accept").map(who))];
    const rightOk = right.filter((c) => c.verdict === "accept").length;
    const rightOdd = right.filter((c) => c.verdict !== "accept").length;
    document.getElementById("s04-summary").innerHTML =
      `On one known-wrong S04 answer, the ${esc(list(rejected))} verifier${rejected.length === 1 ? "" : "s"} <span class="ok">rejected it</span>` +
      (accepted.length ? `; the ${esc(list(accepted))} verifier <span class="no">accepted it</span>` : "") + `. ` +
      `On a known-right answer, ${rightOk} of ${right.length} attempts accepted it` +
      (rightOdd ? ` (${rightOdd} gave ${rightOdd === 1 ? "an unreadable verdict" : "other verdicts"}).` : ".");

    const t = document.getElementById("s04-table");
    const rows = s04.cases.map((c) => {
      const right = c.verdict_right == null ? `<span class="pending">no verdict</span>` : c.verdict_right ? `<span class="ok">yes</span>` : `<span class="no">no</span>`;
      const verdict = c.verdict === "reject" ? `<span class="${c.answer_was === "wrong" ? "ok" : "no"}">reject</span>`
        : c.verdict === "accept" ? `<span class="${c.answer_was === "wrong" ? "no" : "ok"}">accept</span>` : `<span class="pending">${esc(c.verdict)}</span>`;
      const issue = (c.issues && c.issues[0]) || c.reason || "";
      return `<tr><td>${SHORT[c.setting]}${c.used_in_study ? "" : ' <span class="tag-small">(not used in the study)</span>'}</td>` +
        `<td>${esc(c.verifier_model)}</td><td>${c.answer_was === "wrong" ? '<span class="no">wrong</span>' : '<span class="ok">right</span>'}</td>` +
        `<td>${verdict}</td><td class="num">${c.calls}</td><td>${right}</td><td>${esc(issue.length > 140 ? `${issue.slice(0, 140)}…` : issue)}</td></tr>`;
    }).join("");
    document.getElementById("s04-wording").textContent = s04.wording;
    t.innerHTML = `<thead><tr><th>Setting</th><th>Verifier model</th><th>Answer was</th><th>Verdict</th>` +
      `<th class="num">Calls</th><th>Verdict right?</th><th>What it said</th></tr></thead><tbody>${rows}</tbody>`;
  }

  // ------------------------------------------------------------------ 2b. compare two settings

  function setupPair() {
    const { pairs, meta } = state.data;
    const done = new Set(pairs.complete);
    const opts = (sel) => ORDER.map((c) => {
      const m = meta.conditions.find((x) => x.id === c);
      return `<option value="${c}"${done.has(c) ? "" : " disabled"}${c === sel ? " selected" : ""}>${SHORT[c]}${done.has(c) ? "" : ` (pending ${m.runs_done}/${m.runs_expected})`}</option>`;
    }).join("");
    const a = document.getElementById("pair-a"), b = document.getElementById("pair-b");
    const best = state.data.summary.headline.best;
    a.innerHTML = opts("none");
    b.innerHTML = opts(done.has(best) ? best : pairs.complete.find((c) => c !== "none"));
    a.addEventListener("change", renderPair);
    b.addEventListener("change", renderPair);
    document.getElementById("pair-swap").addEventListener("click", () => { const t = a.value; a.value = b.value; b.value = t; renderPair(); });
  }

  function pairList(items, side, other) {
    if (!items.length) return `<p class="tag-small">None.</p>`;
    return `<ul>${items.map((x) => {
      const id = side === "a" ? x.a_run_id : x.b_run_id;
      const mine = side === "a" ? x.a_outcome : x.b_outcome;
      const theirs = side === "a" ? x.b_outcome : x.a_outcome;
      const st = side === "a" ? x.b_status : x.a_status;
      return `<li><button type="button" class="linkish" data-run="${esc(id)}">${esc(x.task_id)} · run ${x.run_number}</button>` +
        `${x.scoring_note ? ' <span class="dagger" title="Scoring under review">†</span>' : ""} ` +
        `<span class="tag-small">draft ${x.draft_correct ? "right" : "wrong"} · here ${esc(OUTCOME[mine] || mine)}; ${esc(SHORT[other])}: ${esc(st === "human_review" ? "human review" : OUTCOME[theirs] || theirs)}</span></li>`;
    }).join("")}</ul>`;
  }

  function renderPair() {
    const { pairs, summary } = state.data;
    const a = document.getElementById("pair-a").value, b = document.getElementById("pair-b").value;
    const body = document.getElementById("pair-body");
    if (a === b) { body.innerHTML = `<p class="tag-small">Pick two different settings.</p>`; return; }
    const p = pairs.pairs.find((x) => x.a === a && x.b === b);
    if (!p) { body.innerHTML = `<p class="pending">pending: both settings must finish before they can be compared.</p>`; return; }
    const sa = summary.conditions[a].task_success, sb = summary.conditions[b].task_success;
    body.innerHTML =
      `<div class="pair-head">` +
      tile("", `${SHORT[a]} (A)`, pct(sa.value), `${sa.right}/${sa.n} runs · ${ci(sa.ci_low, sa.ci_high)}`) +
      tile("", `${SHORT[b]} (B)`, pct(sb.value), `${sb.right}/${sb.n} runs · ${ci(sb.ci_low, sb.ci_high)}`) +
      tile("", "Change, B − A", `<span class="${p.delta > 0 ? "ok" : p.delta < 0 ? "no" : ""}">${pts(p.delta)}</span>`,
        `95% CI ${pts(p.ci_low)} to ${pts(p.ci_high)} · ${p.n_pairs} paired runs, ${p.n_tasks} tasks`) +
      tile("", "Sign test over tasks", pval(p.sign_test_p),
        `B better on ${p.tasks_b_better} task${p.tasks_b_better === 1 ? "" : "s"}, A better on ${p.tasks_a_better}; the rest tie`, "small") +
      `</div><figure class="chart" id="pair-chart"><figcaption>Change in task success, B − A, with its 95% interval</figcaption><div class="chart-slot"></div></figure>` +
      `<div class="pair-lists"><div><h3>Right only with ${esc(SHORT[b])} (${p.runs_only_b_right.length})</h3>${pairList(p.runs_only_b_right, "b", a)}</div>` +
      `<div><h3>Right only with ${esc(SHORT[a])} (${p.runs_only_a_right.length})</h3>${pairList(p.runs_only_a_right, "a", b)}</div></div>` +
      `<p class="note">${esc(pairs.note)} Open a run to see its draft, verdicts and final answer in “Every run”.</p>`;
    const slot = body.querySelector("#pair-chart .chart-slot");
    const span = Math.max(0.05, Math.abs(p.ci_low), Math.abs(p.ci_high)) + 0.02;
    rowChart(slot, {
      rows: [{ condition: b, ...p }], domain: [-span, span], zero: 0, height: 44,
      axisFormat: (v) => `${v > 0 ? "+" : ""}${Math.round(v * 100)}`,
      ariaLabel: `Change in task success from ${SHORT[a]} to ${SHORT[b]}: ${pts(p.delta)}, 95% CI ${pts(p.ci_low)} to ${pts(p.ci_high)}.`,
      valueText: (d) => `${pts(d.delta)} · ${pts(d.ci_low).replace(" pts", "")} to ${pts(d.ci_high)} · ${pval(d.sign_test_p)}`,
      mark: (g, d, x, y) => {
        g.append("line").attr("x1", x(d.ci_low)).attr("x2", x(d.ci_high)).attr("y1", y).attr("y2", y)
          .attr("stroke", color(b)).attr("stroke-width", 3).attr("stroke-linecap", "round").attr("opacity", 0.6);
        g.append("circle").attr("cx", x(d.delta)).attr("cy", y).attr("r", 7).attr("fill", color(b)).attr("stroke", BG).attr("stroke-width", 2);
      },
      tipHtml: (d) => `<strong>${SHORT[b]} − ${SHORT[a]}</strong><br>${pts(d.delta)} (95% CI ${pts(d.ci_low)} to ${pts(d.ci_high)})<br>` +
        `Runs right only with B: ${d.runs_only_b_right.length}; only with A: ${d.runs_only_a_right.length}<br>Sign test over tasks ${pval(d.sign_test_p)}`,
    });
    slot.querySelectorAll(".row text").forEach((t) => { if (t.textContent === SHORT[b]) t.textContent = `${SHORT[b]} − ${SHORT[a]}`; });
    body.querySelectorAll("button.linkish").forEach((btn) => btn.addEventListener("click", () => {
      openRun(btn.dataset.run);
      document.getElementById("run-detail").scrollIntoView({ block: "start", behavior: reduceMotion ? "auto" : "smooth" });
    }));
    addPngButtons(body);
  }

  // ------------------------------------------------------------------ 7b. seeded errors by type

  const DECISION = {
    right_reason: ["✓", "rejected, named the error"], consequence_only: ["~", "rejected, named only a consequence"],
    wrong_reason: ["✗?", "rejected for another reason"], abstained: ["?", "abstained"], missing_verdict: ["–", "verdict unreadable"],
    not_flagged: ["·", "accepted"], false_alarm: ["!", "rejected a correct control"],
  };

  function renderSeededTypes() {
    const { seeded } = state.data;
    const head = `<thead><tr><th scope="col">Error type</th>${VERIFIED.map((c) => {
      const e = seeded.per_condition[c];
      return `<th scope="col" style="text-align:center">${SHORT[c]}${isPending(e) ? `<br><span class="pending tag-small">${esc(pendingLabel(e))}</span>` : ""}</th>`;
    }).join("")}</tr></thead>`;
    const body = seeded.by_error_type.map((g) => {
      const label = g.seeded_error ? g.error_type.replace(/_/g, " ") : "correct controls";
      const cells = VERIFIED.map((c, col) => {
        const v = g.per_condition[c];
        if (!v || v.status === "pending") return `<td class="cell"><span class="chip pending-chip">pending</span></td>`;
        if (!g.seeded_error) {
          const share = v.n ? 1 - v.rejected / v.n : 0;
          return `<td class="cell"><span class="chip" role="img" style="background:${heatColor(share)};--col:${col}" aria-label="${esc(`${SHORT[c]}, controls: ${v.rejected} of ${v.n} false alarms`)}">` +
            `${v.rejected}/${v.n}<small>false alarms</small></span></td>`;
        }
        const share = v.n ? v.right_reason / v.n : 0;
        return `<td class="cell"><span class="chip" role="img" style="background:${heatColor(share)};--col:${col}" aria-label="${esc(`${SHORT[c]}, ${label}: ${v.rejected} of ${v.n} rejected, ${v.right_reason} for the right reason`)}">` +
          `${v.rejected}/${v.n}<small>${v.right_reason} right reason</small></span></td>`;
      }).join("");
      return `<tr><th scope="row">${esc(label)}<span class="ci">${g.n_answers} answer${g.n_answers === 1 ? "" : "s"}</span></th>${cells}</tr>`;
    }).join("");
    const total = VERIFIED.map((c) => {
      const e = seeded.per_condition[c];
      if (isPending(e)) return `<td class="cell pending tag-small">${esc(pendingLabel(e))}</td>`;
      const rr = e.right_reason;
      return `<td class="cell total"><b>${e.catch_rate.caught}/${e.catch_rate.n}</b> caught` +
        `<span class="ci">${rr.right_reason} for the right reason</span>` +
        `<span class="ci">${e.false_alarm_rate.flagged}/${e.false_alarm_rate.n} false alarms${e.missing_verdicts ? ` · ${e.missing_verdicts} unreadable` : ""}</span>` +
        `<span class="ci">${usd(e.cost_usd)}</span></td>`;
    }).join("");
    document.getElementById("type-table").innerHTML = head + `<tbody>${body}</tbody>` +
      `<tfoot><tr><th scope="row">Total<span class="ci">17 errors, 14 controls</span></th>${total}</tr></tfoot>`;
    animateWhenVisible(document.getElementById("type-table"), "type-table");

    const real = state.data.summary.conditions;
    const lines = VERIFIED.filter((c) => !isPending(real[c]) && !isPending(seeded.per_condition[c])).map((c) =>
      `${SHORT[c]} ${seeded.per_condition[c].catch_rate.caught}/${seeded.per_condition[c].catch_rate.n} seeded vs ${real[c].verifier.catch_rate.caught}/${real[c].verifier.catch_rate.n} real`);
    document.getElementById("type-note").textContent = lines.length
      ? `Errors caught, seeded vs the advisor's real errors: ${lines.join(" · ")}. The seeded errors are single, visible slips that every verifier catches; the advisor's real errors are compounding arithmetic, where only the verifier with the calculator caught them all.`
      : "";

    const answers = [...new Set(seeded.reviews.map((r) => r.answer_id))].sort();
    const byKey = new Map(seeded.reviews.map((r) => [`${r.answer_id}|${r.condition}`, r]));
    const rows = answers.map((id) => {
      const any = seeded.reviews.find((r) => r.answer_id === id);
      const cells = VERIFIED.map((c) => {
        if (isPending(seeded.per_condition[c])) return `<td class="pending tag-small">pending</td>`;
        const r = byKey.get(`${id}|${c}`);
        if (!r) return `<td class="tag-small">–</td>`;
        const [sym, words] = DECISION[r.decision] || ["?", r.decision];
        const kind = r.seeded_error ? "err" : "ctl";
        const issue = (r.issues || [])[0] || "";
        return `<td><span class="dec ${esc(r.decision)} ${kind}" title="${esc(words + (issue ? `: ${issue}` : ""))}">${sym}</span> <span class="tag-small">${esc(words)}</span></td>`;
      }).join("");
      return `<tr><td>${esc(id)}<span class="ci">${esc(any.seeded_error ? (any.error_type || "").replace(/_/g, " ") : "correct control")}</span></td>${cells}</tr>`;
    }).join("");
    document.getElementById("answer-table").innerHTML =
      `<thead><tr><th>Seeded answer</th>${VERIFIED.map((c) => `<th>${SHORT[c]}</th>`).join("")}</tr></thead><tbody>${rows}</tbody>`;
  }

  // ------------------------------------------------------------------ PNG export of a chart

  const STYLE_PROPS = ["fill", "stroke", "stroke-width", "stroke-dasharray", "stroke-linecap", "opacity", "fill-opacity",
    "font-size", "font-weight", "font-family", "text-anchor", "dominant-baseline"];
  function svgToPng(svg, caption, filename) {
    const clone = svg.cloneNode(true);
    const src = [svg, ...svg.querySelectorAll("*")], dst = [clone, ...clone.querySelectorAll("*")];
    src.forEach((el, i) => {
      const cs = getComputedStyle(el);
      dst[i].setAttribute("style", STYLE_PROPS.map((k) => `${k}:${cs.getPropertyValue(k)}`).join(";"));
    });
    const vb = svg.viewBox.baseVal;
    const scale = 2, pad = 24, capH = caption ? 34 : 0;
    const w = vb.width + pad * 2, h = vb.height + pad * 2 + capH;
    clone.setAttribute("width", vb.width); clone.setAttribute("height", vb.height);
    const data = new XMLSerializer().serializeToString(clone);
    const img = new Image();
    img.onload = () => {
      const canvas = document.createElement("canvas");
      canvas.width = w * scale; canvas.height = h * scale;
      const ctx = canvas.getContext("2d");
      ctx.scale(scale, scale);
      ctx.fillStyle = css("--card"); ctx.fillRect(0, 0, w, h);
      if (caption) { ctx.fillStyle = TEXT; ctx.font = "600 15px -apple-system, Segoe UI, Helvetica, Arial, sans-serif"; ctx.fillText(caption, pad, pad + 12); }
      ctx.drawImage(img, pad, pad + capH, vb.width, vb.height);
      ctx.fillStyle = FAINT; ctx.font = "11px -apple-system, Segoe UI, Helvetica, Arial, sans-serif";
      ctx.fillText(`CS 5130 verification study · 32 tasks × 3 runs · data as of ${state.data.meta.generated_at.slice(0, 10)}`, pad, h - 8);
      canvas.toBlob((blob) => {
        const a = document.createElement("a");
        a.href = URL.createObjectURL(blob); a.download = filename; a.click();
        setTimeout(() => URL.revokeObjectURL(a.href), 2000);
      }, "image/png");
    };
    img.src = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(data)}`;
  }

  function addPngButtons(scope = document) {
    scope.querySelectorAll("figure.chart").forEach((fig) => {
      const cap = fig.querySelector("figcaption");
      if (!cap || cap.querySelector(".png-btn")) return;
      const btn = document.createElement("button");
      btn.type = "button"; btn.className = "png-btn"; btn.textContent = "PNG";
      const label = cap.textContent.trim();
      btn.setAttribute("aria-label", `Download “${label}” as PNG`);
      btn.addEventListener("click", () => {
        const svg = fig.querySelector(".chart-slot svg");
        if (svg) svgToPng(svg, label, `${fig.id || "chart"}.png`);
      });
      cap.appendChild(btn);
    });
  }

  // ------------------------------------------------------------------ 8. replay

  let replay = null;
  function renderReplay() {
    const { meta, summary, runs } = state.data;
    const c = state.selected;
    const entry = summary.conditions[c];
    const s = meta.study;
    const mountEl = document.getElementById("replay-mount");
    if (replay) { replay.destroy(); replay = null; }
    mountEl.innerHTML = "";
    document.getElementById("replay-dataset").textContent =
      `Data set: main study, ${s.n_tasks} tasks × ${s.repeats} runs, advisor ${s.advisor_model}. Showing: ${SHORT[c]}. The same choice drives “Explore a setting” below.`;
    if (isPending(entry)) {
      document.getElementById("replay-sub").textContent = `${SHORT[c]} is still running, so it is not replayed yet.`;
      mountEl.innerHTML = `<p class="replay-pending">${esc(pendingLabel(entry))}</p>`;
      return;
    }
    const rows = runs.runs.filter((r) => r.condition === c).map(AdvisoryReplay.normalize).filter(Boolean);
    document.getElementById("replay-sub").textContent =
      `${rows.length} recorded ${SHORT[c]} runs in the order they were run.` +
      (reduceMotion ? " Your system asks for reduced motion, so the finished state is shown; press Play to animate." : "");
    replay = AdvisoryReplay.mount(mountEl, { runs: rows, setting: c, measured: true, advisorLabel: s.advisor_model });
  }

  // ------------------------------------------------------------------ 9. run table

  const OUTCOME = {
    kept_correct: "kept correct", fixed: "fixed", broken: "broken", still_wrong: "still wrong", failed: "failed call",
  };
  const ROUTE = { abstain: "verifier abstained", unreadable: "verdict unreadable", revision_not_accepted: "revision not accepted" };
  const PAGE = 40;
  const runState = { shown: PAGE, current: null };

  // Defensive: the export already drops these keys; never render them even if a file had them.
  const FORBIDDEN = new Set(["expected", "correct_value", "wrong_value"]);
  function stripForbidden(v) {
    if (Array.isArray(v)) return v.map(stripForbidden);
    if (v && typeof v === "object") {
      return Object.fromEntries(Object.entries(v).filter(([k]) => !FORBIDDEN.has(k)).map(([k, x]) => [k, stripForbidden(x)]));
    }
    return v;
  }

  function expandRuns(open = true) {
    const btn = document.getElementById("runs-toggle");
    document.getElementById("runs-body").hidden = !open;
    btn.setAttribute("aria-expanded", String(open));
    btn.textContent = open ? "Hide the run list" : `Browse all ${state.data.runs.runs.length} runs`;
  }

  function setupRunFilters() {
    const { runs, meta } = state.data;
    expandRuns(false);
    document.getElementById("runs-toggle").addEventListener("click", () =>
      expandRuns(document.getElementById("runs-toggle").getAttribute("aria-expanded") !== "true"));
    running = new Set(meta.conditions.filter((c) => !c.complete).map((c) => c.id));
    const sSel = document.getElementById("f-setting");
    sSel.innerHTML = `<option value="">All settings</option>` + ORDER.map((c) => {
      const m = meta.conditions.find((x) => x.id === c);
      return `<option value="${c}">${SHORT[c]}${m && !m.complete ? ` (still running, ${m.runs_done}/${m.runs_expected})` : ""}</option>`;
    }).join("");
    const tasks = [...new Set(runs.runs.map((r) => r.task_id))].sort();
    document.getElementById("f-task").innerHTML = `<option value="">All tasks</option>` + tasks.map((t) => `<option>${esc(t)}</option>`).join("");
    ["f-setting", "f-task", "f-outcome"].forEach((id) => document.getElementById(id).addEventListener("change", () => { runState.shown = PAGE; renderRuns(); }));
    document.getElementById("run-more").addEventListener("click", () => { runState.shown += PAGE; renderRuns(); });
  }

  let running = new Set();
  function runRow(r) {
    const hr = r.status === "human_review";
    const outcome = hr ? `<span class="badge hr">human review</span> <span class="tag-small">${esc(ROUTE[r.human_review_route] || "")}</span>`
      : `<span class="badge ${r.outcome}">${esc(OUTCOME[r.outcome] || r.outcome)}</span>`;
    const verdicts = r.condition === "none" ? '<span class="tag-small">no check</span>'
      : `${esc(r.verdict || "unreadable")}${r.second_verdict ? ` → ${esc(r.second_verdict)}` : ""}`;
    const tick = (v) => (v ? '<span class="ok">right</span>' : '<span class="no">wrong</span>');
    return `<tr data-run="${esc(r.run_id)}"${runState.current === r.run_id ? ' aria-current="true"' : ""}>` +
      `<td><button type="button" class="linkish" data-run="${esc(r.run_id)}">${esc(r.task_id)} · run ${r.run_number}</button>${r.scoring_note ? ' <span class="dagger" title="Scoring under review">†</span>' : ""}</td>` +
      `<td>${SHORT[r.condition]}${running.has(r.condition) ? ' <span class="tag-small pending">still running</span>' : ""}</td><td>${r.status === "failed" ? "–" : tick(r.initial_correct)}</td><td>${verdicts}</td>` +
      `<td>${r.status === "failed" ? "–" : tick(r.final_correct)}</td><td>${outcome}</td><td class="num">${usd(r.cost_usd, 3)}</td></tr>`;
  }

  function renderRuns() {
    const { runs } = state.data;
    const fs = document.getElementById("f-setting").value;
    const ft = document.getElementById("f-task").value;
    const fo = document.getElementById("f-outcome").value;
    const order = Object.fromEntries(ORDER.map((c, i) => [c, i]));
    const match = runs.runs.filter((r) => (!fs || r.condition === fs) && (!ft || r.task_id === ft))
      .slice().sort((a, b) => a.task_id.localeCompare(b.task_id) || order[a.condition] - order[b.condition] || a.run_number - b.run_number);
    const decided = match.filter((r) => r.status !== "human_review" && (!fo || (fo !== "human_review" && r.outcome === fo)));
    const hr = match.filter((r) => r.status === "human_review" && (!fo || fo === "human_review" || r.outcome === fo));
    const head = `<thead><tr><th>Run</th><th>Setting</th><th>Draft</th><th>Verdict</th><th>Final</th><th>Outcome</th><th class="num">Cost</th></tr></thead>`;
    const shown = decided.slice(0, runState.shown);
    document.getElementById("run-table").innerHTML = head + `<tbody>${shown.map(runRow).join("") || '<tr><td colspan="7" class="tag-small">No runs match.</td></tr>'}</tbody>`;
    document.getElementById("hr-run-table").innerHTML = head + `<tbody>${hr.map(runRow).join("") || '<tr><td colspan="7" class="tag-small">No human-review runs match.</td></tr>'}</tbody>`;
    const more = document.getElementById("run-more");
    more.hidden = decided.length <= runState.shown;
    more.textContent = `Show more (${decided.length - shown.length} left)`;
    const pending = fs && state.data.meta.conditions.find((x) => x.id === fs && !x.complete);
    document.getElementById("run-count").textContent =
      `${decided.length} decided run${decided.length === 1 ? "" : "s"} and ${hr.length} human-review run${hr.length === 1 ? "" : "s"} match.` +
      (pending ? ` ${SHORT[fs]} is still running (${pending.runs_done}/${pending.runs_expected}); these are its saved runs so far, not a result.` : "");
    document.querySelectorAll("table.runs tbody tr[data-run]").forEach((tr) => {
      tr.addEventListener("click", (e) => { if (!e.target.closest("button")) openRun(tr.dataset.run); });
    });
    document.querySelectorAll("table.runs button.linkish").forEach((b) => b.addEventListener("click", () => openRun(b.dataset.run)));
  }

  function answerBlock(title, answer) {
    if (answer == null) return `<h4>${title}</h4><p class="tag-small">none</p>`;
    const obs = answer && answer.metrics && Array.isArray(answer.metrics.observations) ? answer.metrics.observations : [];
    const obsHtml = obs.length
      ? `<ul class="obs">${obs.map((o) => `<li>${esc([o.subject, o.measure, o.stage, o.rounding !== "NONE" ? o.rounding : ""].filter(Boolean).join(" · "))}: <b>${esc(o.value)}</b></li>`).join("")}</ul>`
      : "";
    const extra = answer && typeof answer === "object"
      ? `${answer.status ? `<p class="tag-small">status ${esc(answer.status)}${answer.answer_type ? ` · ${esc(answer.answer_type)}` : ""}${Array.isArray(answer.trades) ? ` · ${answer.trades.length} trade(s)` : ""}</p>` : ""}`
      : "";
    return `<h4>${title}</h4>${extra}${obsHtml}<details><summary class="tag-small">Full answer</summary><pre>${esc(typeof answer === "string" ? answer : JSON.stringify(answer, null, 2))}</pre></details>`;
  }

  function checksBlock(title, check) {
    if (!check) return `<h4>${title}</h4><p class="tag-small">not graded</p>`;
    const failed = (check.checks || []).filter((c) => !c.passed);
    const head = `<h4>${title}: <span class="${check.correct ? "ok" : "no"}">${check.correct ? "right" : "wrong"}</span>${check.format_failure ? " (format failure)" : ""}</h4>`;
    if (!failed.length) return head + `<p class="tag-small">All ${(check.checks || []).length} checks passed.</p>`;
    return head + `<ul>${failed.map((c) => {
      const d = c.details || {};
      const items = (d.items || []).map((it) => `${esc(it.metric || "")} ${esc(it.reason || "")}${it.observed !== undefined ? `, the answer reported ${esc(it.observed)}` : ""}`);
      const missing = (d.missing_observation || []).length ? `missing: ${esc(d.missing_observation.join(", "))}` : "";
      const reasons = [d.reason, ...(d.reasons || [])].filter(Boolean).map(esc);
      return `<li><b>${esc(c.check_type)}</b> (${esc(c.check_id)}): ${[...new Set(reasons)].join(", ")}${missing ? ` · ${missing}` : ""}` +
        `${items.length ? `<ul>${items.map((x) => `<li>${x}</li>`).join("")}</ul>` : ""}</li>`;
    }).join("")}</ul><p class="tag-small">Only what the answer reported and why the check failed; the answer key stays on the scorer side.</p>`;
  }

  function reviewBlock(rv) {
    const calc = (rv.calculations || []).length
      ? `<details><summary class="tag-small">Calculator (${rv.calculations.length} expression${rv.calculations.length > 1 ? "s" : ""})</summary><ul class="obs">${rv.calculations.map((c) => `<li>${esc(c.expression)} = ${esc(c.result ?? c.error ?? "")}</li>`).join("")}</ul></details>`
      : "";
    return `<h4>${rv.stage === "second" ? "Second check (of the revision)" : "Verifier review"}: ` +
      `<span class="${rv.verdict === "accept" ? "ok" : rv.verdict === "reject" ? "no" : "pending"}">${esc(rv.verdict || "unreadable")}</span> ` +
      `<span class="tag-small">${rv.calls ?? "?"} call(s)</span></h4>` +
      (rv.reason ? `<p>${esc(rv.reason)}</p>` : "") +
      ((rv.issues || []).length ? `<ul>${rv.issues.map((i) => `<li>${esc(i)}</li>`).join("")}</ul>` : "") +
      (rv.error ? `<p class="no">${esc(rv.error)}</p>` : "") + calc;
  }

  async function openRun(runId) {
    expandRuns(true);
    const panel = document.getElementById("run-detail");
    runState.current = runId;
    document.querySelectorAll("table.runs tbody tr").forEach((tr) => tr.toggleAttribute("aria-current", tr.dataset.run === runId));
    const row = state.data.runs.runs.find((r) => r.run_id === runId);
    panel.hidden = false;
    panel.innerHTML = `<p class="tag-small">Loading ${esc(runId)}…</p>`;
    let d;
    try {
      const res = await fetch(`data/runs/${encodeURIComponent(runId.replace(/[^A-Za-z0-9._-]/g, "_"))}.json`, { cache: "no-store" });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      d = stripForbidden(await res.json());
    } catch (err) {
      panel.innerHTML = `<p class="no">No saved answer file for ${esc(runId)}${row && row.status === "failed" ? " (the API call failed, so there is no answer)" : ""}.</p>`;
      panel.focus();
      return;
    }
    const hr = row && row.status === "human_review";
    panel.innerHTML =
      `<button type="button" class="pill close" id="run-close">Close</button>` +
      `<h3>${esc(d.task_id)} · ${SHORT[d.condition]} · run ${d.run_number}</h3>` +
      `<div class="meta"><span>status <b>${esc(d.status)}</b></span>` +
      `${row ? `<span>outcome <b>${esc(hr ? `human review (${ROUTE[row.human_review_route] || ""})` : OUTCOME[row.outcome] || row.outcome)}</b></span>` : ""}` +
      `<span>verifier calls <b>${d.verifier_calls ?? 0}</b></span>${row ? `<span>cost <b>${usd(row.cost_usd, 4)}</b> (includes the shared draft)</span>` : ""}</div>` +
      (d.scoring_note ? `<p class="note review-note">${esc(d.scoring_note)}</p>` : "") +
      `<div class="cols"><div>${answerBlock("First draft", d.draft)}${checksBlock("Checker on the draft", d.draft_check)}</div>` +
      `<div>${d.reviews.length ? d.reviews.map(reviewBlock).join("") : `<h4>Verifier</h4><p class="tag-small">No verifier in this setting.</p>`}</div></div>` +
      `<div class="cols"><div>${d.revision ? answerBlock("Revision", d.revision) : ""}${answerBlock("Final answer", d.final)}</div>` +
      `<div>${checksBlock("Checker on the final answer", d.final_check)}</div></div>`;
    document.getElementById("run-close").addEventListener("click", () => {
      panel.hidden = true; runState.current = null;
      const back = document.querySelector(`table.runs button.linkish[data-run="${CSS.escape(runId)}"]`);
      (back || document.getElementById("runs-h")).focus();
    });
    panel.focus();
    panel.scrollIntoView({ block: "nearest", behavior: reduceMotion ? "auto" : "smooth" });
  }

  // ------------------------------------------------------------------ boot

  function renderNotes() {
    const text = reviewNoteText();
    document.querySelectorAll("[data-review]").forEach((el) => {
      el.hidden = !text;
      el.textContent = text;
    });
  }

  function renderFooter() {
    const { meta } = state.data;
    const done = meta.conditions.map((c) => `${SHORT[c.id]} ${c.runs_done}/${c.runs_expected}`).join(" · ");
    const st = meta.study;
    document.getElementById("foot-meta").textContent =
      `Main study: ${st.n_tasks} tasks (${st.n_scenarios} scenarios S01–S10 and their variants) × ${st.repeats} runs · advisor ${st.advisor_model}, ` +
      `temperature ${st.temperature}, top-p ${st.top_p}, ${st.thinking_level} thinking · data generated ${meta.generated_at.replace("T", " ").replace("+00:00", " UTC")} · runs saved: ${done}.`;
  }

  function renderCharts() {
    renderCompare();
    renderPair();
    renderCatch();
    renderFix();
    renderCost();
    addPngButtons();
  }

  async function main() {
    try {
      state.data = await loadAll();
    } catch (err) {
      const msg = location.protocol === "file:"
        ? "Open this page through a local web server so it can load data/*.json: run <code>python -m http.server -d dashboard</code> and visit http://localhost:8000."
        : `Could not load the data (${esc(err.message)}). Run <code>python dashboard/build_data.py</code> first.`;
      document.getElementById("headline").innerHTML = `<span class="error">${msg}</span>`;
      return;
    }
    state.selected = state.data.summary.headline.best || "none";
    renderHero();
    renderPills();
    renderKpis();
    renderHeatmap();
    renderSanity();
    renderSeededTypes();
    setupPair();
    setupRunFilters();
    renderRuns();
    renderReplay();
    renderNotes();
    renderFooter();
    renderCharts();
    document.getElementById("print-btn").addEventListener("click", () => window.print());
    // Charts are drawn for the width they have; redraw them for the printed page and back afterwards.
    window.addEventListener("beforeprint", () => { renderCharts(); renderKpis(); });
    window.addEventListener("afterprint", () => { renderCharts(); renderKpis(); });
    let replayLayout = AdvisoryReplay.layoutFor(document.getElementById("replay-mount").getBoundingClientRect().width);
    let last = window.innerWidth;
    window.addEventListener("resize", () => {
      if (Math.abs(window.innerWidth - last) < 8) return;
      last = window.innerWidth;
      clearTimeout(main.t);
      main.t = setTimeout(() => {
        renderCharts(); renderKpis();
        const now = AdvisoryReplay.layoutFor(document.getElementById("replay-mount").getBoundingClientRect().width);
        if (now !== replayLayout) { replayLayout = now; renderReplay(); }
      }, reduceMotion ? 0 : 120);
    });
  }

  window.__dashboard = { state, openRun, select, svgToPng };
  main();
})();
