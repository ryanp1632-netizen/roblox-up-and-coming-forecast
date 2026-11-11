// Walk-forward backtest with the contest's exact scoring rule.
//
//   node tools/backtest.ts <history.ndjson> [--params '{"bwRank":0.3}'] [--from 2026-11-01T00:00:00Z] [--min-span 48]
//
// Every read at or after --from whose following reads reach at least --min-span hours ahead is an origin: the
// forecaster sees the reads up to and including it, and is scored on the reads in the 72 hours after it (one per
// UTC hour: the first read that started in it), as the sealed scorer scores the window. Prints the mean log score
// per origin for the forecaster and the house persistence baseline, then their means.

import { readFileSync } from 'node:fs';
import { parseHistory, forecast, DEFAULTS, PLACES } from '../main.ts';

const FLOOR = 1e-4;

/** The log score of one read: the sum over places of ln max(P, 1e-4), the unnamed share spread as the scorer does. */
export function readScore(rows, chart) {
  const named = Array.from({ length: PLACES }, () => new Map());
  for (const r of rows) named[r.place - 1].set(r.universe_id, r.p);
  let total = 0;
  const n = Math.min(PLACES, chart.length);
  for (let k = 0; k < n; k++) {
    const at = named[k];
    const id = chart[k];
    let p;
    if (at.has(id)) p = at.get(id);
    else {
      let sum = 0;
      for (const v of at.values()) sum += v;
      let unnamed = 0;
      for (const c of new Set(chart)) if (!at.has(c)) unnamed += 1;
      p = unnamed === 0 ? 0 : Math.max(0, 1 - sum) / unnamed;
    }
    total += Math.log(Math.max(p, FLOOR));
  }
  return total;
}

/** The house persistence baseline's rows for a chart (starter/main.ts). */
export function baselineRows(chart) {
  const rows = [];
  for (let place = 1; place <= PLACES && place <= chart.length; place++) {
    rows.push({ universe_id: chart[place - 1], place, p: 0.5 });
    if (place === 1) { if (chart[1] !== undefined) rows.push({ universe_id: chart[1], place, p: 0.3 }); continue; }
    rows.push({ universe_id: chart[place - 2], place, p: 0.15 });
    if (chart[place] !== undefined) rows.push({ universe_id: chart[place], place, p: 0.15 });
  }
  return rows;
}

/** One target read per UTC hour in (t0, t0 + 72]: the first read that started in it. */
export function windowTargets(reads, i) {
  const t0 = reads[i].t;
  const seen = new Set();
  const out = [];
  for (let j = i + 1; j < reads.length; j++) {
    const h = reads[j].t - t0;
    if (h > 72) break;
    const slot = Math.floor(reads[j].t);
    if (seen.has(slot)) continue;
    seen.add(slot);
    out.push(reads[j]);
  }
  return out;
}

export function backtest(reads, params, opts = {}) {
  const from = opts.from ? Date.parse(opts.from) / 3600000 : -Infinity;
  const to = opts.to ? Date.parse(opts.to) / 3600000 : Infinity;
  const minSpan = opts.minSpan ?? 48;
  const tEnd = reads[reads.length - 1].t;
  const res = [];
  for (let i = 1; i < reads.length - 1; i++) {
    if (reads[i].t < from || reads[i].t >= to || tEnd - reads[i].t < minSpan) continue;
    const T = windowTargets(reads, i);
    if (T.length === 0) continue;
    const hist = reads.slice(0, i + 1);
    const f = forecast(hist, params);
    const base = baselineRows(reads[i].ids);
    let s = 0;
    let sb = 0;
    for (const tr of T) { s += readScore(f.probabilities, tr.ids); sb += readScore(base, tr.ids); }
    res.push({ at: reads[i].at, n: T.length, score: s / T.length, base: sb / T.length });
  }
  const mean = (a) => a.reduce((x, y) => x + y, 0) / Math.max(1, a.length);
  return { rows: res, score: mean(res.map((r) => r.score)), base: mean(res.map((r) => r.base)) };
}

if (import.meta.url.endsWith('/tools/backtest.ts') && process.argv[1] && process.argv[1].replace(/\\/g, '/').endsWith('tools/backtest.ts')) {
  const args = process.argv.slice(2);
  const file = args[0];
  const get = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined; };
  const params = { ...DEFAULTS, ...(get('--params') ? JSON.parse(get('--params')) : {}) };
  const reads = parseHistory(readFileSync(file, 'utf8'));
  const r = backtest(reads, params, { from: get('--from'), to: get('--to'), minSpan: get('--min-span') ? Number(get('--min-span')) : undefined });
  if (!args.includes('--quiet')) for (const x of r.rows) console.log(x.at, x.n, x.score.toFixed(3), x.base.toFixed(3));
  console.log(`origins ${r.rows.length}  model ${r.score.toFixed(4)}  baseline ${r.base.toFixed(4)}`);
}
