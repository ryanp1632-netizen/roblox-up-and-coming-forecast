// Roblox Up-and-Coming top-10 forecaster (Node 22, standard library only, no network, deterministic).
//
// It reads the chart's history (/input/history.ndjson) and writes one static forecast for the scored window
// (/output/forecast.json): for each of the top 10 places, a probability for every game on the chart's last read.
//
// Method: a conditional, horizon-averaged nearest-neighbour kernel learned from the history itself.
//   1. Every past read i is an "origin". For each game on the chart at i and each horizon bin b (0-6 h, 6-12 h, ...,
//      60-72 h), the observation is the share of the reads in bin b that put the game at place k (k = 1..10).
//   2. A game on the last read is described by features: log rank now, the rank's change over ~24 h and ~6 h, and
//      its player growth over ~24 h. Its probability at place k in bin b is the kernel-weighted mean of the
//      observations of past games with similar features (recent origins weigh a bit more).
//   3. The share of place k that goes to games absent from the origin's chart (new entrants, returners) is learned
//      the same way, per bin, and left unnamed: the scorer spreads it over the games we did not name.
//   4. Bins are averaged with their length as weight (each hour of the window counts the same), each place is
//      normalised, and every probability is floored so no plausible outcome is given a tiny probability.

import { readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';
import { pathToFileURL } from 'node:url';

export const PLACES = 10;
const OFF_RANK = 130; // the rank a game off the chart counts as

export const DEFAULTS = {
  bins: [0, 6, 12, 24, 36, 48, 60, 72],
  bwRank: 0.35, // kernel bandwidth on ln(rank)
  bwV24: 0.6, // on the ~24 h log-rank change
  bwV6: 0.5, // on the ~6 h log-rank change
  bwG24: 0.8, // on the ~24 h log player growth
  recencyHalfLife: 120, // hours: an origin this old weighs half
  originGapCap: 6, // hours: an origin weighs its gap to the previous origin, capped here
  priorWeight: 0.5, // pseudo-weight of the rank-only prior in each kernel estimate
  floor: 0.0008, // least probability of a named game at a place
  poolFloor: 0.02, // least unnamed share of a place
  poolScale: 1.0, // multiplier on the learned unnamed share
  temp: 1.0, // power applied to the named probabilities before normalising (1 = none)
};

/** Parses history.ndjson text into reads sorted by time: {t (hours), ids (in rank order), rank Map, playing Map}. */
export function parseHistory(text) {
  const reads = [];
  for (const line of text.split('\n')) {
    if (line.trim() === '') continue;
    let o;
    try { o = JSON.parse(line); } catch { continue; }
    if (!o || o.ok === false || !Array.isArray(o.up_and_coming) || o.up_and_coming.length === 0) continue;
    const rows = [...o.up_and_coming].filter((r) => r && Number.isFinite(r.rank) && Number.isFinite(Number(r.universe_id)));
    rows.sort((a, b) => a.rank - b.rank);
    const ids = [];
    const rank = new Map();
    const playing = new Map();
    for (const r of rows) {
      const id = Number(r.universe_id);
      if (rank.has(id)) continue;
      ids.push(id);
      rank.set(id, ids.length);
      if (Number.isFinite(r.playing) && r.playing > 0) playing.set(id, r.playing);
    }
    reads.push({ t: Date.parse(o.at) / 3600000, at: o.at, ids, rank, playing });
  }
  reads.sort((a, b) => a.t - b.t);
  return reads;
}

function lnRank(read, id) {
  const r = read.rank.get(id);
  return Math.log(r === undefined ? OFF_RANK : r);
}

/** Index of the read before i closest to t_i - lag, within [lag - tol, lag + tol] hours back; -1 if none. */
function lagIndex(reads, i, lag, tol) {
  let best = -1;
  let bestD = Infinity;
  const t = reads[i].t - lag;
  for (let j = i - 1; j >= 0; j--) {
    const d = Math.abs(reads[j].t - t);
    if (reads[i].t - reads[j].t > lag + tol) break;
    if (d <= tol && d < bestD) { best = j; bestD = d; }
  }
  return best;
}

/** Feature vectors of every game on read i: Map id -> [lnRank, v24, v6, g24]. */
export function featuresAt(reads, i) {
  const r = reads[i];
  const j24 = lagIndex(reads, i, 24, 10);
  const j6 = lagIndex(reads, i, 6, 3.5);
  const out = new Map();
  for (const id of r.ids) {
    const lr = lnRank(r, id);
    const v24 = j24 >= 0 ? lnRank(reads[j24], id) - lr : 0;
    const v6 = j6 >= 0 ? lnRank(reads[j6], id) - lr : v24 / 4;
    let g24 = 0;
    if (j24 >= 0) {
      const a = r.playing.get(id);
      const b = reads[j24].playing.get(id);
      if (a && b) g24 = Math.log(a / b);
    }
    out.set(id, [lr, v24, v6, g24]);
  }
  return out;
}

/** Observations from every origin with at least one later read in a bin: per bin, a list of {x, y, w, origin}. */
export function buildObservations(reads, params) {
  const bins = params.bins;
  const nb = bins.length - 1;
  const obs = Array.from({ length: nb }, () => []);
  const pool = Array.from({ length: nb }, () => ({ w: 0, y: new Float64Array(PLACES) }));
  const tLast = reads[reads.length - 1].t;
  for (let i = 0; i < reads.length - 1; i++) {
    const o = reads[i];
    const gap = i === 0 ? params.originGapCap : Math.min(params.originGapCap, Math.max(0.25, o.t - reads[i - 1].t));
    const rec = Math.pow(0.5, (tLast - o.t) / params.recencyHalfLife);
    const wo = gap * rec;
    // targets per bin
    const tb = Array.from({ length: nb }, () => []);
    for (let j = i + 1; j < reads.length; j++) {
      const h = reads[j].t - o.t;
      if (h > bins[nb]) break;
      for (let b = 0; b < nb; b++) if (h > bins[b] && h <= bins[b + 1]) { tb[b].push(reads[j]); break; }
    }
    let feats = null;
    for (let b = 0; b < nb; b++) {
      const T = tb[b];
      if (T.length === 0) continue;
      if (!feats) feats = featuresAt(reads, i);
      const inv = 1 / T.length;
      const ys = new Map();
      const py = pool[b].y;
      for (const tr of T) {
        const n = Math.min(PLACES, tr.ids.length);
        for (let k = 0; k < n; k++) {
          const id = tr.ids[k];
          if (o.rank.has(id)) {
            let y = ys.get(id);
            if (!y) { y = new Float64Array(PLACES); ys.set(id, y); }
            y[k] += inv;
          } else py[k] += inv * wo;
        }
      }
      pool[b].w += wo;
      for (const id of o.ids) obs[b].push({ x: feats.get(id), y: ys.get(id) || null, w: wo, r: o.rank.get(id) });
    }
  }
  return { obs, pool };
}

/** The forecast for the window after the last read: {probabilities, top10}. */
export function forecast(reads, params = DEFAULTS) {
  const P = { ...DEFAULTS, ...params };
  const last = reads.length - 1;
  const origin = reads[last];
  const ids = origin.ids.slice(0, 100);
  const nb = P.bins.length - 1;
  const { obs, pool } = buildObservations(reads, P);
  const feats = featuresAt(reads, last);
  const ib = [1 / P.bwRank, 1 / P.bwV24, 1 / P.bwV6, 1 / P.bwG24];
  const prob = new Map(ids.map((id) => [id, new Float64Array(PLACES)]));
  const poolP = new Float64Array(PLACES);
  let wbTotal = 0;
  for (let b = 0; b < nb; b++) {
    const O = obs[b];
    if (O.length === 0 || pool[b].w <= 0) continue;
    const wb = P.bins[b + 1] - P.bins[b];
    wbTotal += wb;
    // rank-only prior for this bin: mean y by origin rank bucket over all observations
    const prior = rankPrior(O);
    for (const id of ids) {
      const q = feats.get(id);
      const acc = new Float64Array(PLACES);
      let wsum = 0;
      for (const ob of O) {
        const x = ob.x;
        const d0 = (x[0] - q[0]) * ib[0];
        if (d0 > 3 || d0 < -3) continue;
        const d1 = (x[1] - q[1]) * ib[1];
        const d2 = (x[2] - q[2]) * ib[2];
        const d3 = (x[3] - q[3]) * ib[3];
        const w = ob.w * Math.exp(-0.5 * (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3));
        if (w < 1e-12) continue;
        wsum += w;
        if (ob.y) for (let k = 0; k < PLACES; k++) acc[k] += w * ob.y[k];
      }
      const pr = prior(origin.rank.get(id));
      const pw = P.priorWeight;
      const out = prob.get(id);
      for (let k = 0; k < PLACES; k++) out[k] += wb * (acc[k] + pw * pr[k]) / (wsum + pw);
    }
    for (let k = 0; k < PLACES; k++) poolP[k] += wb * pool[b].y[k] / pool[b].w;
  }
  if (wbTotal > 0) {
    for (const v of prob.values()) for (let k = 0; k < PLACES; k++) v[k] /= wbTotal;
    for (let k = 0; k < PLACES; k++) poolP[k] /= wbTotal;
  }
  return finish(ids, prob, poolP, P);
}

/** Mean place shares by origin-rank bucket (1, 2, ..., 10, 11-15, 16-25, 26-50, 51+), smoothed toward uniform. */
function rankPrior(O) {
  const bucket = (r) => (r <= 10 ? r - 1 : r <= 15 ? 10 : r <= 25 ? 11 : r <= 50 ? 12 : 13);
  const acc = Array.from({ length: 14 }, () => new Float64Array(PLACES));
  const ws = new Float64Array(14);
  for (const ob of O) {
    const b = bucket(ob.r);
    ws[b] += ob.w;
    if (ob.y) for (let k = 0; k < PLACES; k++) acc[b][k] += ob.w * ob.y[k];
  }
  const res = acc.map((a, b) => a.map((v) => (v + 0.01) / (ws[b] + 1)));
  return (r) => res[bucket(r ?? 99)];
}

/** Normalises each place, floors it, and writes the rows and the top-10 guess. */
export function finish(ids, prob, poolP, P) {
  const probabilities = [];
  const table = new Map(ids.map((id) => [id, new Float64Array(PLACES)]));
  for (let k = 0; k < PLACES; k++) {
    let pool = Math.max(P.poolFloor, Math.min(0.9, poolP[k] * P.poolScale));
    let raw = ids.map((id) => Math.pow(Math.max(prob.get(id)[k], 1e-9), P.temp));
    let s = raw.reduce((a, b) => a + b, 0);
    const named = 1 - pool;
    // floor every named game, then scale the rest to fill the named mass
    const fl = Math.min(P.floor, named / ids.length / 2);
    const free = named - fl * ids.length;
    for (let n = 0; n < ids.length; n++) {
      const p = fl + free * (raw[n] / s);
      table.get(ids[n])[k] = p;
    }
  }
  for (const id of ids) {
    const v = table.get(id);
    for (let k = 0; k < PLACES; k++) {
      const p = Math.floor(v[k] * 1e8) / 1e8; // round down so a place never sums above 1
      if (p > 0) probabilities.push({ universe_id: id, place: k + 1, p });
    }
  }
  // top 10: greedy by probability, each place its likeliest unused game
  const top10 = [];
  const used = new Set();
  for (let k = 0; k < PLACES; k++) {
    let best = null;
    let bp = -1;
    for (const id of ids) if (!used.has(id) && table.get(id)[k] > bp) { bp = table.get(id)[k]; best = id; }
    if (best === null) break;
    used.add(best);
    top10.push(best);
  }
  return { probabilities, top10, table };
}

function main() {
  const inDir = process.env.MAKEFUN_INPUT_DIR ?? '/input';
  const outDir = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';
  let task = { places: PLACES, output: 'forecast.json' };
  try { task = { ...task, ...JSON.parse(readFileSync(join(inDir, 'task.json'), 'utf8')) }; } catch {}
  const reads = parseHistory(readFileSync(join(inDir, 'history.ndjson'), 'utf8'));
  const { probabilities, top10 } = forecast(reads, DEFAULTS);
  writeFileSync(join(outDir, task.output || 'forecast.json'), JSON.stringify({ probabilities, top10 }) + '\n');
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) main();
