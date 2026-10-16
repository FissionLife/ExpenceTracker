/* WalletLedger dashboard: reads /api/dashboard and draws every chart as plain SVG (no libraries).
   Filters in the bar re-query the server, so the numbers always match what the bot shows. */
(() => {
  "use strict";

  // ------------------------------------------------------------------------------ helpers
  const $ = (sel, el = document) => el.querySelector(sel);
  const SVG_NS = "http://www.w3.org/2000/svg";
  const COLORS = Array.from({ length: 10 }, (_, i) => `var(--c${i})`);
  const catColor = (name, i) => (name === "Other" ? COLORS[9] : COLORS[i % 9]);
  // "Other" and "Uncategorized" are groupings, not real categories: filtering on them would match nothing
  const canFilter = (name) => name !== "Other" && name !== "Uncategorized";

  function h(tag, attrs, ...kids) {
    const el = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined || v === false) continue;
      if (k === "class") el.className = v;
      else if (k === "text") el.textContent = v;
      else if (k === "style") el.style.cssText = v; // not setAttribute: the CSP blocks inline style attributes
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v === true ? "" : v);
    }
    for (const kid of kids.flat()) if (kid !== null && kid !== undefined) el.append(kid.nodeType ? kid : document.createTextNode(kid));
    return el;
  }
  function s(tag, attrs, ...kids) {
    const el = document.createElementNS(SVG_NS, tag);
    for (const [k, v] of Object.entries(attrs || {})) {
      if (v === null || v === undefined) continue;
      if (k === "text") el.textContent = v;
      else if (k === "style") el.style.cssText = v;
      else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
      else el.setAttribute(k, v);
    }
    for (const kid of kids.flat()) if (kid) el.append(kid);
    return el;
  }
  function inr(n, d = 0) {
    const neg = n < 0;
    let [w, f] = Math.abs(n).toFixed(d).split(".");
    if (w.length > 3) w = w.slice(0, -3).replace(/\B(?=(\d{2})+(?!\d))/g, ",") + "," + w.slice(-3);
    return (neg ? "-" : "") + "₹" + w + (d ? "." + f : "");
  }
  function short(n) {
    const a = Math.abs(n), sign = n < 0 ? "-" : "";
    if (a >= 1e7) return `${sign}₹${(a / 1e7).toFixed(1)}Cr`;
    if (a >= 1e5) return `${sign}₹${(a / 1e5).toFixed(1)}L`;
    if (a >= 1e3) return `${sign}₹${(a / 1e3).toFixed(1).replace(".0", "")}k`;
    return `${sign}₹${Math.round(a)}`;
  }
  const pctTxt = (p) => (p === null || p === undefined || Math.abs(p) < 1 ? "" : `${p > 0 ? "▲" : "▼"}${Math.abs(p).toFixed(0)}%`);
  // Draw each chart at the width it is shown at, so text keeps its real size on phones.
  const cw = (el) => Math.max(280, Math.min(960, Math.round(el.clientWidth || 440)));
  const hh = (W) => Math.round(Math.max(200, Math.min(300, W * 0.42)));
  const clear = (el) => { el.replaceChildren(); return el; };
  const empty = (el, msg) => clear(el).append(h("div", { class: "empty", text: msg }));
  const niceMax = (v) => {
    if (v <= 0) return 1;
    const p = Math.pow(10, Math.floor(Math.log10(v))), n = v / p;
    const steps = [1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10];
    return steps.find((x) => n <= x + 1e-9) * p;
  };
  const store = {
    get(k) { try { return localStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { localStorage.setItem(k, v); } catch { /* private mode */ } },
  };
  const sess = {
    get(k) { try { return sessionStorage.getItem(k); } catch { return null; } },
    set(k, v) { try { sessionStorage.setItem(k, v); } catch { /* ignore */ } },
  };

  // ------------------------------------------------------------------------------ tooltip
  const tip = $("#tip");
  document.addEventListener("pointermove", (e) => {
    const t = e.target.closest ? e.target.closest("[data-tip]") : null;
    if (!t) { tip.hidden = true; return; }
    tip.textContent = t.getAttribute("data-tip");
    tip.hidden = false;
    const x = Math.min(e.clientX + 14, window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = `${Math.max(8, x)}px`;
    tip.style.top = `${e.clientY + 16 + tip.offsetHeight > window.innerHeight ? e.clientY - tip.offsetHeight - 10 : e.clientY + 16}px`;
  });
  document.addEventListener("pointerleave", () => { tip.hidden = true; });

  // ------------------------------------------------------------------------------ state
  const PERIODS = [["7d", "7D"], ["30d", "30D"], ["90d", "90D"], ["6m", "6M"], ["1y", "1Y"], ["ytd", "YTD"], ["all", "All"], ["custom", "Custom"]];
  const state = { period: "30d", start: "", end: "", categories: [], accounts: [], merchants: [], min: "", max: "", group: "auto", compare: true, page: 1, sort: "date", dir: "desc", q: "", type: "" };
  let TOKEN = null, REPORT = null, loadTimer = null, loadSeq = 0;

  function params(extra = {}) {
    const p = new URLSearchParams();
    if (state.period === "custom" && state.start && state.end) { p.set("start", state.start); p.set("end", state.end); }
    else p.set("period", state.period === "custom" ? "30d" : state.period);
    if (state.categories.length) p.set("categories", state.categories.join(","));
    if (state.accounts.length) p.set("accounts", state.accounts.join(","));
    if (state.merchants.length) p.set("merchants", state.merchants.join(","));
    if (state.min !== "") p.set("min", state.min);
    if (state.max !== "") p.set("max", state.max);
    if (state.group !== "auto") p.set("group", state.group);
    p.set("compare", state.compare ? "1" : "0");
    for (const [k, v] of Object.entries(extra)) p.set(k, v);
    return p;
  }

  async function api(path, extra) {
    const res = await fetch(`${path}?${params(extra)}`, { headers: { "X-Dashboard-Token": TOKEN } });
    if (res.status === 401) throw Object.assign(new Error("expired"), { code: 401 });
    if (!res.ok) {
      let detail = "";
      try { detail = (await res.json()).detail; } catch { /* not json */ }
      throw new Error(detail || `Error ${res.status}`);
    }
    return res;
  }

  function showStatus(msg, isError) {
    const el = $("#status");
    el.hidden = !msg;
    el.className = "status" + (isError ? " error" : "");
    el.textContent = msg || "";
  }

  // ------------------------------------------------------------------------------ loading
  function schedule(delay = 250) {
    clearTimeout(loadTimer);
    loadTimer = setTimeout(load, delay);
  }

  async function load() {
    const seq = ++loadSeq;
    document.body.classList.add("loading");
    try {
      const res = await api("/api/dashboard");
      const rep = await res.json();
      if (seq !== loadSeq) return;
      REPORT = rep;
      showStatus("");
      render(rep);
      state.page = 1;
      loadTx();
    } catch (e) {
      if (e.code === 401) showStatus("This link has expired. Ask the bot for /dashboard to get a new one.", true);
      else showStatus(`Could not load your data: ${e.message}`, true);
    } finally {
      if (seq === loadSeq) document.body.classList.remove("loading");
    }
  }

  // ------------------------------------------------------------------------------ rendering
  function render(rep) {
    renderFilters(rep);
    renderKpis(rep);
    renderInsights(rep);
    const none = !rep.summary.transactions;
    $("#grid").hidden = none;
    if (none) {
      showStatus(rep.query.filtered ? "No transactions match these filters." : "Nothing logged in this period yet. Tell the bot, for example: Spent 450 on dinner via Bank");
      return;
    }
    $("#trendSub").textContent = `${rep.query.describe} · per ${rep.series.group}`;
    drawTrend($("#chTrend"), rep);
    drawCategories($("#chCats"), rep);
    drawIncome($("#chIncome"), rep);
    drawPace($("#chPace"), rep);
    drawStack($("#chStack"), rep);
    drawWeekdays($("#chWeek"), rep);
    drawHeatmap($("#chHeat"), rep);
    drawBudgets($("#chBudgets"), rep);
    drawNet($("#chNet"), rep);
    drawMerchants($("#chMerch"), rep);
    drawSizes($("#chSizes"), rep);
    drawAccounts($("#chAcc"), rep);
    drawLists(rep);
  }

  function renderFilters(rep) {
    const wrap = clear($("#periods"));
    for (const [id, label] of PERIODS) wrap.append(h("button", { class: "pchip", type: "button", "aria-pressed": String(state.period === id), text: label, onclick: () => { state.period = id; if (id === "custom" && !state.start) { const d = new Date(), f = new Date(Date.now() - 29 * 864e5); state.end = d.toISOString().slice(0, 10); state.start = f.toISOString().slice(0, 10); } syncInputs(); if (id !== "custom") schedule(0); else { $("#customRange").hidden = false; schedule(0); } } }));
    $("#customRange").hidden = state.period !== "custom";
    fillMenu($("#catList"), $("#catMenu"), "Categories", rep.options.categories, state.categories, (v) => { state.categories = v; schedule(); });
    fillMenu($("#accList"), $("#accMenu"), "Accounts", rep.options.accounts, state.accounts, (v) => { state.accounts = v; schedule(); });
    const chips = clear($("#activeFilters"));
    const addChip = (label, remove) => chips.append(h("span", { class: "chip" }, label, h("button", { type: "button", "aria-label": `Remove ${label}`, text: "×", onclick: remove })));
    state.categories.forEach((c) => addChip(c, () => { state.categories = state.categories.filter((x) => x !== c); schedule(0); }));
    state.accounts.forEach((c) => addChip(c, () => { state.accounts = state.accounts.filter((x) => x !== c); schedule(0); }));
    state.merchants.forEach((c) => addChip(`Merchant: ${c}`, () => { state.merchants = []; $("#merchant").value = ""; schedule(0); }));
    if (state.min !== "") addChip(`Over ₹${state.min}`, () => { state.min = ""; syncInputs(); schedule(0); });
    if (state.max !== "") addChip(`Under ₹${state.max}`, () => { state.max = ""; syncInputs(); schedule(0); });
  }

  function fillMenu(list, details, title, options, selected, onChange) {
    const summary = $("summary", details);
    summary.textContent = selected.length ? `${title} (${selected.length})` : title;
    clear(list);
    if (!options.length) { list.append(h("div", { class: "empty", text: "None yet" })); return; }
    for (const name of options) {
      const box = h("input", { type: "checkbox", value: name });
      box.checked = selected.includes(name);
      box.addEventListener("change", () => {
        const now = [...list.querySelectorAll("input:checked")].map((i) => i.value);
        onChange(now);
      });
      list.append(h("label", {}, box, name));
    }
  }

  function syncInputs() {
    $("#startDate").value = state.start; $("#endDate").value = state.end;
    $("#minAmt").value = state.min; $("#maxAmt").value = state.max;
    $("#group").value = state.group; $("#compare").checked = state.compare;
    $("#customRange").hidden = state.period !== "custom";
    document.querySelectorAll(".pchip").forEach((b, i) => b.setAttribute("aria-pressed", String(PERIODS[i][0] === state.period)));
  }

  function kpi(label, value, deltaPct, upIsBad, note) {
    const d = pctTxt(deltaPct);
    let cls = "";
    if (d) cls = deltaPct > 0 ? (upIsBad ? "up-bad" : "up-good") : (upIsBad ? "down-good" : "down-bad");
    return h("div", { class: "kpi" }, h("div", { class: "k-label", text: label }), h("div", { class: "k-value", text: value }),
      d ? h("div", { class: `k-delta ${cls}`, text: `${d} vs before` }) : null, note ? h("div", { class: "k-note", text: note }) : null);
  }

  function renderKpis(rep) {
    const s = rep.summary, box = clear($("#kpis"));
    if (rep.query.filtered) {
      box.append(kpi("Spent", inr(s.spent), s.spent_change_pct, true, `${s.expense_count} purchases`), kpi("Average purchase", inr(s.avg_expense), null, false, `median ${inr(s.median_expense)}`),
        kpi("Largest", inr(s.largest_expense)), kpi("Per day", inr(s.avg_per_day), null, false, `${s.active_days} active days`));
      return;
    }
    box.append(kpi("Spent", inr(s.spent), s.spent_change_pct, true, `${s.expense_count} purchases`), kpi("Earned", inr(s.earned), s.earned_change_pct, false),
      kpi("Saved", inr(s.net), null, false, s.savings_rate === null ? "" : `${s.savings_rate.toFixed(0)}% of income`), kpi("Per day", inr(s.avg_per_day), null, false, `${s.no_spend_days} no-spend days`),
      kpi("Money now", inr(rep.balances.total), null, false, rep.runway.months && !rep.runway.low_confidence ? `lasts ~${rep.runway.months} months` : ""));
  }

  function renderInsights(rep) {
    const box = $("#insights");
    box.hidden = !rep.insights.length;
    if (!rep.insights.length) return;
    clear(box).append(h("h2", { text: "Worth knowing" }), h("ul", {}, rep.insights.map((t) => h("li", { text: t }))));
  }

  // ------------------------------------------------------------------------------ axes
  function frame(W, H, m, ymax, fmtY) {
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const ax = s("g", { class: "ax" });
    for (let i = 0; i <= 4; i++) {
      const y = m.t + ((H - m.t - m.b) * i) / 4;
      ax.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y, y2: y }), s("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", text: fmtY(ymax * (1 - i / 4)) }));
    }
    svg.append(ax);
    return svg;
  }
  function xLabels(svg, labels, xs, H, m) {
    const step = Math.max(1, Math.ceil(labels.length / 8));
    const g = s("g", { class: "ax" });
    labels.forEach((l, i) => { if (i % step === 0) g.append(s("text", { x: xs(i), y: H - m.b + 18, "text-anchor": "middle", text: l })); });
    svg.append(g);
  }

  // ------------------------------------------------------------------------------ charts
  function bucketRange(key, group) {
    const a = new Date(key + "T00:00:00");
    const b = new Date(a);
    if (group === "week") b.setDate(b.getDate() + 6);
    else if (group === "month") { b.setMonth(b.getMonth() + 1); b.setDate(0); }
    const iso = (d) => `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}-${String(d.getDate()).padStart(2, "0")}`;
    return [iso(a), iso(b)];
  }

  function drawTrend(el, rep) {
    const pts = rep.series.points, group = rep.series.group;
    if (!pts.some((p) => p.spent)) return empty(clear(el), "No spending in this period");
    const W = cw(el), H = hh(W), m = { l: 54, r: 12, t: 14, b: 34 };
    const ymax = niceMax(Math.max(...pts.map((p) => Math.max(p.spent, p.avg))));
    const svg = frame(W, H, m, ymax, short);
    const bw = (W - m.l - m.r) / pts.length, xs = (i) => m.l + bw * (i + 0.5), ys = (v) => m.t + (H - m.t - m.b) * (1 - v / ymax);
    pts.forEach((p, i) => {
      const hgt = Math.max(p.spent ? 2 : 0, (H - m.t - m.b) * (p.spent / ymax));
      const [a, b] = bucketRange(p.key, group);
      svg.append(s("rect", { class: "bar", x: xs(i) - bw * 0.36, width: bw * 0.72, y: H - m.b - hgt, height: hgt, rx: 3, fill: "var(--primary)", "data-tip": `${p.label}: ${inr(p.spent)}`, onclick: () => { state.period = "custom"; state.start = a; state.end = b; syncInputs(); schedule(0); } }));
    });
    if (pts.length > 3) svg.append(s("path", { d: pts.map((p, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(p.avg).toFixed(1)}`).join(" "), fill: "none", stroke: "#8a8a8a", "stroke-dasharray": "6 4", "stroke-width": 2.5, "stroke-linejoin": "round" }));
    xLabels(svg, pts.map((p) => p.label), xs, H, m);
    clear(el).append(svg);
  }

  function drawCategories(el, rep) {
    const cats = rep.categories;
    if (!cats.length) return empty(clear(el), "No spending in this period");
    const total = cats.reduce((a, c) => a + c.amount, 0), R = 84, r = 54, C = 100;
    const svg = s("svg", { viewBox: "0 0 200 200", role: "img", style: "max-width:220px;margin:auto" });
    let angle = -Math.PI / 2;
    cats.forEach((c, i) => {
      const frac = c.amount / total, a2 = angle + frac * Math.PI * 2 - (cats.length > 1 ? 0.012 : 0), big = a2 - angle > Math.PI ? 1 : 0;
      const p = (rad, a) => [C + rad * Math.cos(a), C + rad * Math.sin(a)];
      const [x1, y1] = p(R, angle), [x2, y2] = p(R, a2), [x3, y3] = p(r, a2), [x4, y4] = p(r, angle);
      const d = frac >= 0.9999 ? `M${C},${C - R} A${R},${R} 0 1 1 ${C - 0.01},${C - R} L${C - 0.01},${C - r} A${r},${r} 0 1 0 ${C},${C - r} Z`
        : `M${x1},${y1} A${R},${R} 0 ${big} 1 ${x2},${y2} L${x3},${y3} A${r},${r} 0 ${big} 0 ${x4},${y4} Z`;
      svg.append(s("path", { class: "slice", d, fill: catColor(c.category, i), "data-tip": `${c.category}: ${inr(c.amount)} (${c.share_pct.toFixed(0)}%)`, onclick: () => canFilter(c.category) && toggleFilter("categories", c.category) }));
      angle += frac * Math.PI * 2;
    });
    svg.append(s("text", { x: C, y: C - 2, "text-anchor": "middle", "font-size": 12, fill: "var(--on-surface-variant)", text: "Spent" }), s("text", { x: C, y: C + 18, "text-anchor": "middle", "font-size": 19, "font-weight": 700, fill: "var(--on-surface)", text: short(total) }));
    const legend = h("div", { class: "legend" }, cats.map((c, i) => {
      const ch = pctTxt(c.change_pct);
      return h("button", { type: "button", onclick: () => canFilter(c.category) && toggleFilter("categories", c.category) },
        h("i", { style: `background:${catColor(c.category, i)}` }), h("span", { text: c.category }), h("span", { class: "amt", text: short(c.amount) }),
        h("span", { class: `chg ${ch.startsWith("▲") ? "up-bad" : "down-good"}`, text: ch || `${c.share_pct.toFixed(0)}%` }));
    }));
    clear(el).append(svg, legend);
  }

  function groupedBars(el, labels, a, b, nameA, nameB, colorA, colorB) {
    const W = cw(el), H = hh(W), m = { l: 50, r: 8, t: 24, b: 34 };
    const ymax = niceMax(Math.max(...a, ...b, 1));
    const svg = frame(W, H, m, ymax, short);
    const bw = (W - m.l - m.r) / labels.length, xs = (i) => m.l + bw * (i + 0.5);
    labels.forEach((l, i) => {
      for (const [k, v, col, nm] of [[-1, a[i], colorA, nameA], [1, b[i], colorB, nameB]]) {
        const hgt = (H - m.t - m.b) * (v / ymax);
        svg.append(s("rect", { class: "bar", x: xs(i) + (k < 0 ? -bw * 0.34 : 0), width: bw * 0.34, y: H - m.b - hgt, height: hgt, rx: 2, fill: col, "data-tip": `${l} · ${nm}: ${inr(v)}` }));
      }
    });
    xLabels(svg, labels, xs, H, m);
    svg.append(s("g", { class: "ax" }, s("rect", { x: m.l, y: 4, width: 10, height: 10, rx: 2, fill: colorA }), s("text", { x: m.l + 15, y: 13, text: nameA }), s("rect", { x: m.l + 80, y: 4, width: 10, height: 10, rx: 2, fill: colorB }), s("text", { x: m.l + 95, y: 13, text: nameB })));
    clear(el).append(svg);
  }
  function drawIncome(el, rep) {
    const p = rep.series.points;
    if (!p.some((x) => x.spent || x.earned)) return empty(clear(el), "Nothing to compare yet");
    groupedBars(el, p.map((x) => x.label), p.map((x) => x.earned), p.map((x) => x.spent), "Earned", "Spent", "var(--c2)", "var(--primary)");
  }
  function drawAccounts(el, rep) {
    const a = rep.accounts;
    if (!a.length) return empty(clear(el), "No account activity");
    groupedBars(el, a.map((x) => x.account.slice(0, 14)), a.map((x) => x.earned), a.map((x) => x.spent), "In", "Out", "var(--c2)", "var(--primary)");
  }

  function lineChart(el, series, W, H, m, xlabels, yfmt, extras) {
    const ymax = niceMax(Math.max(1, ...series.flatMap((x) => x.values)));
    const n = Math.max(...series.map((x) => x.values.length));
    const svg = frame(W, H, m, ymax, yfmt);
    const xs = (i) => m.l + ((W - m.l - m.r) * i) / Math.max(1, n - 1), ys = (v) => m.t + (H - m.t - m.b) * (1 - v / ymax);
    for (const sr of series) {
      const d = sr.values.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
      if (sr.area) svg.append(s("path", { d: `${d} L${xs(sr.values.length - 1)},${H - m.b} L${xs(0)},${H - m.b} Z`, fill: sr.color, "fill-opacity": 0.12 }));
      svg.append(s("path", { d, fill: "none", stroke: sr.color, "stroke-width": sr.width || 2.5, "stroke-dasharray": sr.dash || null, "stroke-linejoin": "round" }));
    }
    if (extras) extras(svg, xs, ys);
    if (xlabels) xLabels(svg, xlabels, xs, H, m);
    clear(el).append(svg);
    return { xs, ys };
  }

  function drawPace(el, rep) {
    const p = rep.pace;
    $("#paceSub").textContent = `Spent so far ${inr(p.spent)} · projected ${inr(p.projected)} (${inr(p.regular_daily)}/day${p.bills_due ? ` + ${inr(p.bills_due)} bills due` : ""})`;
    if (!p.this_month.length && !p.last_month.length) return empty(clear(el), "No spending this month or last month");
    const series = [{ values: p.last_month, color: "var(--outline)", dash: "6 5", width: 2 }, { values: p.this_month, color: "var(--primary)" }];
    const days = Math.max(p.days_in_month, p.last_month.length);
    series[0].values = p.last_month.concat(Array(Math.max(0, days - p.last_month.length)).fill(p.last_month.at(-1) || 0)).slice(0, days);
    series[1].values = p.this_month;
    const ymax = Math.max(p.projected, ...p.last_month, 1);
    lineChart(el, [{ values: [ymax], color: "transparent" }, ...series], cw(el), hh(cw(el)), { l: 50, r: 12, t: 16, b: 30 }, Array.from({ length: days }, (_, i) => String(i + 1)), short, (svg, xs, ys) => {
      svg.append(s("path", { d: `M${xs(p.day - 1)},${ys(p.spent)} L${xs(p.days_in_month - 1)},${ys(p.projected)}`, fill: "none", stroke: "var(--primary)", "stroke-width": 2, "stroke-dasharray": "2 5" }),
        s("text", { x: xs(p.days_in_month - 1), y: ys(p.projected) - 8, "text-anchor": "end", "font-size": 11, fill: "var(--link)", text: short(p.projected) }));
    });
  }

  function drawStack(el, rep) {
    const st = rep.stack;
    if (!st.periods.length) return empty(clear(el), "No spending in this period");
    const W = cw(el), H = hh(W), m = { l: 54, r: 12, t: 14, b: 34 };
    const totals = st.periods.map((p) => st.categories.reduce((a, c) => a + (p.values[c] || 0), 0));
    const ymax = niceMax(Math.max(...totals, 1));
    const svg = frame(W, H, m, ymax, short);
    const bw = (W - m.l - m.r) / st.periods.length, xs = (i) => m.l + bw * (i + 0.5);
    st.periods.forEach((p, i) => {
      let y = H - m.b;
      st.categories.forEach((c, ci) => {
        const v = p.values[c] || 0, hgt = (H - m.t - m.b) * (v / ymax);
        if (hgt > 0) svg.append(s("rect", { class: "bar", x: xs(i) - bw * 0.36, width: bw * 0.72, y: y - hgt, height: hgt, fill: catColor(c, ci), "data-tip": `${p.label} · ${c}: ${inr(v)}` }));
        y -= hgt;
      });
    });
    xLabels(svg, st.periods.map((p) => p.label), xs, H, m);
    clear(el).append(svg, h("div", { class: "stack-legend" }, st.categories.map((c, i) => h("span", {}, h("i", { style: `background:${catColor(c, i)}` }), c))));
  }

  function drawWeekdays(el, rep) {
    const w = rep.weekdays, top = Math.max(...w.map((x) => x.avg));
    if (!top) return empty(clear(el), "Not enough spending for a weekly pattern");
    const W = cw(el), H = hh(W), m = { l: 54, r: 10, t: 16, b: 30 }, ymax = niceMax(top), svg = frame(W, H, m, ymax, short);
    const bw = (W - m.l - m.r) / 7, xs = (i) => m.l + bw * (i + 0.5);
    w.forEach((d, i) => {
      const hgt = (H - m.t - m.b) * (d.avg / ymax);
      svg.append(s("rect", { class: "bar", x: xs(i) - bw * 0.3, width: bw * 0.6, y: H - m.b - hgt, height: hgt, rx: 3, fill: "var(--primary)", "fill-opacity": d.avg === top ? 1 : 0.4, "data-tip": `${d.day}: ${inr(d.avg)} on average` }));
    });
    xLabels(svg, w.map((d) => d.day), xs, H, m);
    clear(el).append(svg);
  }

  function drawHeatmap(el, rep) {
    const hm = rep.heatmap;
    if (!hm.cells.length || !hm.max) return empty(clear(el), "No spending in the last 12 weeks");
    const size = 15, gap = 3, left = 30, W = left + hm.weeks * (size + gap), H = 7 * (size + gap) + 10;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    ["Mon", "", "Wed", "", "Fri", "", "Sun"].forEach((d, i) => d && svg.append(s("text", { x: 0, y: i * (size + gap) + size - 2, "font-size": 10, fill: "var(--on-surface-variant)", text: d })));
    for (const c of hm.cells) {
      const ratio = c.spent / hm.max;
      svg.append(s("rect", { class: "hm", x: left + c.week * (size + gap), y: c.dow * (size + gap), width: size, height: size, rx: 4, fill: c.spent ? "var(--primary)" : "var(--surface-variant)", "fill-opacity": c.spent ? 0.18 + 0.82 * ratio : 1, "data-tip": `${c.date}: ${inr(c.spent)}`, onclick: () => { state.period = "custom"; state.start = c.date; state.end = c.date; syncInputs(); schedule(0); } }));
    }
    clear(el).append(svg);
  }

  function drawBudgets(el, rep) {
    const b = rep.budgets;
    if (!b.length) return empty(clear(el), "No budgets yet. In the chat: /budget food 6000");
    clear(el).append(...b.map((x) => h("div", { class: "budget" }, h("div", { class: "b-top" }, h("strong", { text: x.category }), h("span", { text: `${short(x.spent)} of ${short(x.budget)} · ${x.percent.toFixed(0)}%` })),
      h("div", { class: "track" }, h("div", { class: `fill ${x.level === "ok" ? "good" : x.level}`, style: `width:${Math.min(100, x.percent)}%` })))));
  }

  function drawNet(el, rep) {
    const n = rep.net_worth;
    if (n.length < 2) return empty(clear(el), "Not enough history yet");
    const vals = n.map((p) => p.total), lo = Math.min(...vals), hi = Math.max(...vals), pad = (hi - lo) * 0.15 || Math.max(Math.abs(hi) * 0.05, 1);
    $("#nwSub").textContent = `Now ${inr(vals.at(-1))} · ${vals.at(-1) - vals[0] >= 0 ? "+" : "-"}${inr(Math.abs(vals.at(-1) - vals[0]))} over the range`;
    const W = cw(el), H = hh(W), m = { l: 58, r: 12, t: 14, b: 30 }, base = lo - pad, span = hi + pad - base;
    const shifted = vals.map((v) => v - base);
    const ymax = span;
    const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img" });
    const ax = s("g", { class: "ax" });
    for (let i = 0; i <= 4; i++) {
      const y = m.t + ((H - m.t - m.b) * i) / 4;
      ax.append(s("line", { class: "gridline", x1: m.l, x2: W - m.r, y1: y, y2: y }), s("text", { x: m.l - 8, y: y + 4, "text-anchor": "end", text: short(base + ymax * (1 - i / 4)) }));
    }
    svg.append(ax);
    const xs = (i) => m.l + ((W - m.l - m.r) * i) / (vals.length - 1), ys = (v) => m.t + (H - m.t - m.b) * (1 - v / ymax);
    const d = shifted.map((v, i) => `${i ? "L" : "M"}${xs(i).toFixed(1)},${ys(v).toFixed(1)}`).join(" ");
    svg.append(s("path", { d: `${d} L${xs(vals.length - 1)},${H - m.b} L${xs(0)},${H - m.b} Z`, fill: "var(--primary)", "fill-opacity": 0.12 }), s("path", { d, fill: "none", stroke: "var(--primary)", "stroke-width": 2.5, "stroke-linejoin": "round" }));
    n.forEach((p, i) => svg.append(s("circle", { cx: xs(i), cy: ys(shifted[i]), r: 9, fill: "transparent", "data-tip": `${p.label}: ${inr(p.total)}` })));
    xLabels(svg, n.map((p) => p.label), xs, H, m);
    clear(el).append(svg);
  }

  function drawMerchants(el, rep) {
    const ms = rep.merchants;
    if (!ms.length) return empty(clear(el), "No merchants recorded in this period");
    const top = ms[0].amount || 1;
    clear(el).append(...ms.map((x) => h("div", { class: "budget clickable", onclick: () => toggleFilter("merchants", x.merchant) }, h("div", { class: "b-top" }, h("strong", { text: x.merchant }), h("span", { text: `${short(x.amount)} · ${x.count}×` })),
      h("div", { class: "track" }, h("div", { class: "fill brand", style: `width:${(x.amount / top) * 100}%` })))));
  }

  function drawSizes(el, rep) {
    const z = rep.sizes;
    if (!z.some((x) => x.count)) return empty(clear(el), "No expenses in this period");
    const W = cw(el), H = hh(W), m = { l: 40, r: 10, t: 16, b: 30 }, ymax = Math.max(4, Math.ceil(niceMax(Math.max(...z.map((x) => x.count))) / 4) * 4), svg = frame(W, H, m, ymax, (v) => String(Math.round(v)));
    const bw = (W - m.l - m.r) / z.length, xs = (i) => m.l + bw * (i + 0.5);
    z.forEach((b, i) => { const hgt = (H - m.t - m.b) * (b.count / ymax); svg.append(s("rect", { class: "bar", x: xs(i) - bw * 0.3, width: bw * 0.6, y: H - m.b - hgt, height: hgt, rx: 3, fill: "var(--c1)", "data-tip": `${b.band}: ${b.count} purchases, ${inr(b.total)}` })); });
    xLabels(svg, z.map((b) => b.band), xs, H, m);
    clear(el).append(svg);
  }

  function row(title, sub, amount) {
    return h("div", { class: "row" }, h("div", {}, h("div", { text: title }), sub ? h("div", { class: "sub2", text: sub }) : null), h("div", { class: "amt", text: amount }));
  }
  function drawLists(rep) {
    const rec = rep.recurring, un = rep.unusual, big = rep.biggest;
    const fill = (id, rows, msg) => { const el = clear($(id)); rows.length ? el.append(...rows) : el.append(h("div", { class: "empty", text: msg })); };
    fill("#listRecurring", rec.map((r) => row(r.merchant, `${r.category} · ${r.times}× · ${inr(r.yearly)}/year`, `${inr(r.amount)}/mo`)), "No repeating payments found yet");
    fill("#listUnusual", un.map((u) => row(u.merchant || u.category, `${u.date} · usual ${inr(u.usual)}`, inr(u.amount))), "Nothing unusual. Nice.");
    fill("#listBiggest", big.map((b) => row(b.merchant || b.category, `${b.date} · ${b.category}`, inr(b.amount))), "No expenses in this period");
  }

  function toggleFilter(kind, value) {
    const list = state[kind];
    state[kind] = list.includes(value) ? list.filter((x) => x !== value) : [...list, value];
    if (kind === "merchants") $("#merchant").value = state.merchants.join(", ");
    schedule(0);
  }

  // ------------------------------------------------------------------------------ transactions
  async function loadTx() {
    try {
      const res = await api("/api/transactions", { page: state.page, per_page: 15, sort: state.sort, dir: state.dir, q: state.q, type: state.type });
      const d = await res.json(), body = clear($("#txBody"));
      if (!d.rows.length) body.append(h("tr", {}, h("td", { colspan: 5, class: "empty", text: "No transactions match" })));
      for (const r of d.rows) {
        const sign = r.type === "expense" ? "−" : r.type === "income" ? "+" : "⇄", cls = r.type === "expense" ? "t-exp" : r.type === "income" ? "t-inc" : "t-tr";
        body.append(h("tr", {}, h("td", {}, r.date, r.time ? h("div", { class: "meta", text: r.time }) : null),
          h("td", {}, r.merchant || r.note || "—", r.merchant && r.note ? h("div", { class: "meta", text: r.note }) : null),
          h("td", { text: r.category }), h("td", { class: "hide-sm", text: r.account }), h("td", { class: `num ${cls}`, text: `${sign}${inr(r.amount, 2)}` })));
      }
      $("#txInfo").textContent = `${d.total} transactions · page ${d.page} of ${d.pages} · spent ${inr(d.sum_expense)} · earned ${inr(d.sum_income)}`;
      $("#txPrev").disabled = d.page <= 1; $("#txNext").disabled = d.page >= d.pages;
      document.querySelectorAll("#txTable th button[data-sort]").forEach((b) => { if (b.dataset.sort === state.sort) b.setAttribute("aria-sort", state.dir === "asc" ? "ascending" : "descending"); else b.removeAttribute("aria-sort"); });
    } catch (e) { if (e.code !== 401) showStatus(`Could not load transactions: ${e.message}`, true); }
  }

  // ------------------------------------------------------------------------------ wiring
  function wire() {
    $("#startDate").addEventListener("change", (e) => { state.start = e.target.value; if (state.start && state.end) schedule(0); });
    $("#endDate").addEventListener("change", (e) => { state.end = e.target.value; if (state.start && state.end) schedule(0); });
    $("#merchant").addEventListener("input", (e) => { state.merchants = e.target.value.split(",").map((x) => x.trim()).filter(Boolean); schedule(400); });
    $("#minAmt").addEventListener("input", (e) => { state.min = e.target.value; schedule(400); });
    $("#maxAmt").addEventListener("input", (e) => { state.max = e.target.value; schedule(400); });
    $("#group").addEventListener("change", (e) => { state.group = e.target.value; schedule(0); });
    $("#compare").addEventListener("change", (e) => { state.compare = e.target.checked; schedule(0); });
    $("#reset").addEventListener("click", () => { Object.assign(state, { period: "30d", start: "", end: "", categories: [], accounts: [], merchants: [], min: "", max: "", group: "auto", compare: true }); $("#merchant").value = ""; syncInputs(); schedule(0); });
    $("#txSearch").addEventListener("input", (e) => { state.q = e.target.value; state.page = 1; clearTimeout(wire.t); wire.t = setTimeout(loadTx, 300); });
    $("#txType").addEventListener("change", (e) => { state.type = e.target.value; state.page = 1; loadTx(); });
    $("#txPrev").addEventListener("click", () => { state.page = Math.max(1, state.page - 1); loadTx(); });
    $("#txNext").addEventListener("click", () => { state.page += 1; loadTx(); });
    document.querySelectorAll("#txTable th button[data-sort]").forEach((b) => b.addEventListener("click", () => { const k = b.dataset.sort; state.dir = state.sort === k && state.dir === "desc" ? "asc" : "desc"; state.sort = k; state.page = 1; loadTx(); }));
    $("#exportCsv").addEventListener("click", async () => {
      try {
        const res = await api("/api/export.csv");
        const url = URL.createObjectURL(await res.blob()), a = h("a", { href: url, download: "walletledger.csv" });
        document.body.append(a); a.click(); a.remove(); setTimeout(() => URL.revokeObjectURL(url), 2000);
      } catch (e) { showStatus(`Export failed: ${e.message}`, true); }
    });
    $("#theme").addEventListener("click", () => { const dark = document.documentElement.dataset.theme !== "dark"; setTheme(dark ? "dark" : "light"); store.set("wl-theme", dark ? "dark" : "light"); if (REPORT) render(REPORT); });
    document.addEventListener("click", (e) => { document.querySelectorAll("details.menu[open]").forEach((d) => { if (!d.contains(e.target)) d.open = false; }); });
  }

  function setTheme(t) {
    document.documentElement.dataset.theme = t;
    const meta = document.querySelector('meta[name="theme-color"]');
    if (meta) meta.content = t === "dark" ? "#121214" : "#ffffff";
  }

  let lastW = window.innerWidth;
  window.addEventListener("resize", () => { if (Math.abs(window.innerWidth - lastW) < 40) return; lastW = window.innerWidth; clearTimeout(wire.r); wire.r = setTimeout(() => REPORT && render(REPORT), 200); });

  // ------------------------------------------------------------------------------ start
  async function signInWithTelegram(tg) {
    tg.ready(); tg.expand();
    if (tg.colorScheme === "dark" && !store.get("wl-theme")) setTheme("dark");
    const res = await fetch("/api/webapp/auth", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ initData: tg.initData }) });
    if (!res.ok) throw new Error("Telegram sign-in failed");
    return (await res.json()).token;
  }

  async function start() {
    setTheme(store.get("wl-theme") || "light");
    wire();
    const fromPath = location.pathname.startsWith("/d/") ? location.pathname.split("/")[2] : null;
    if (fromPath) { sess.set("wl-token", fromPath); history.replaceState(null, "", "/d"); }
    TOKEN = fromPath || sess.get("wl-token");
    if (!TOKEN && (location.hash.includes("tgWebAppData") || /Telegram/i.test(navigator.userAgent))) {
      try {
        await new Promise((ok, fail) => { const sc = document.createElement("script"); sc.src = "https://telegram.org/js/telegram-web-app.js"; sc.onload = ok; sc.onerror = fail; document.head.append(sc); });
        if (window.Telegram && Telegram.WebApp && Telegram.WebApp.initData) TOKEN = await signInWithTelegram(Telegram.WebApp);
      } catch (e) { showStatus(String(e.message || e), true); return; }
    }
    if (!TOKEN) { showStatus("Open this page from your bot: send /dashboard in Telegram.", true); return; }
    syncInputs();
    load();
  }
  start();
})();
