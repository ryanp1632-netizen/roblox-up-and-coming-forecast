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
  bwRank: 0.8, // kernel bandwidth on ln(rank)
  bwV24: 0.6, // on the ~24 h log-rank change
  bwV6: 0.5, // on the ~6 h log-rank change
  bwG24: 0.5, // on the ~24 h log player growth
  bwAge: 50, // on ln(1 + hours since the game first showed on the chart)
  bwPeak: 0.4, // on how far (log rank) the game sits below its best rank so far
  recencyHalfLife: 1000, // hours: an origin this old weighs half
  originGapCap: 6, // hours: an origin weighs its gap to the previous origin, capped here
  priorWeight: 2, // pseudo-weight of the rank-only prior in each kernel estimate
  floor: 0.0002, // least probability of a named game at a place
  poolFloor: 0.04, // least unnamed share of a place
  poolScale: 1.3, // multiplier on the learned unnamed share
  temp: 1.0, // power applied to the named probabilities before normalising (1 = none)
  mixDiffusion: 0.3, // weight of the parametric log-rank diffusion component
  dDrift: 0.02, // log-rank drift per hour (games sink as they age)
  dMeanRev: 0.4, // pull toward rank 15 over a day
  dMom24: 0.5, // share of the last 24 h log-rank change that carries on
  dMomHours: 24, // for this many hours
  dMom6: 0.2, // share of the last 6 h change that carries on (over 6 h)
  dS0: 0.25, // log-rank spread at horizon 0
  dS1: 0.12, // and its growth per sqrt(hour)
  mixReg: 0.5, // weight of the regression-fitted drift component
  regMaxRank: 100, // games ranked below this are left out of the drift fit
  regRidge: 0.001, // ridge penalty, per unit of weight
  regS0: 0.05, // log-rank spread at horizon 0
  regSdScale: 0.9, // multiplier on the fitted spreads
  regTail: 0.2, // share of a 2.5x wider Gaussian, for heavy tails
  capFrom: 0.12, // a place's share above this is shrunk...
  capSlope: 0.5, // ...to this fraction of its excess (1 = no cap)
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

/** First-seen time and best log rank so far of every game, read by read: computed once per history. */
function lifeTable(reads) {
  if (reads._life) return reads._life;
  const first = new Map();
  const peak = new Map();
  const life = [];
  for (const r of reads) {
    for (const id of r.ids) {
      if (!first.has(id)) first.set(id, r.t);
      const lr = Math.log(r.rank.get(id));
      if (!peak.has(id) || lr < peak.get(id)) peak.set(id, lr);
    }
    const snap = new Map();
    for (const id of r.ids) snap.set(id, [r.t - first.get(id), peak.get(id)]);
    life.push(snap);
  }
  reads._life = life;
  return life;
}

/** Feature vectors of every game on read i: Map id -> [lnRank, v24, v6, g24, lnChartAge, belowPeak]. */
export function featuresAt(reads, i) {
  const r = reads[i];
  const j24 = lagIndex(reads, i, 24, 10);
  const j6 = lagIndex(reads, i, 6, 3.5);
  const life = lifeTable(reads)[i];
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
    const [age, pk] = life.get(id);
    out.set(id, [lr, v24, v6, g24, Math.log(1 + age), lr - pk]);
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
  const ib = [1 / P.bwRank, 1 / P.bwV24, 1 / P.bwV6, 1 / P.bwG24, 1 / P.bwAge, 1 / P.bwPeak];
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
        const d4 = (x[4] - q[4]) * ib[4];
        const d5 = (x[5] - q[5]) * ib[5];
        const w = ob.w * Math.exp(-0.5 * (d0 * d0 + d1 * d1 + d2 * d2 + d3 * d3 + d4 * d4 + d5 * d5));
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
  if (P.mixDiffusion > 0) {
    const dif = diffusion(ids, feats, P);
    for (const id of ids) {
      const a = prob.get(id);
      const d = dif.get(id);
      for (let k = 0; k < PLACES; k++) a[k] = (1 - P.mixDiffusion) * a[k] + P.mixDiffusion * d[k] * (1 - poolP[k]);
    }
  }
  if (P.mixReg > 0) {
    const reg = regressionDiffusion(reads, ids, feats, P);
    for (const id of ids) {
      const a = prob.get(id);
      const d = reg.get(id);
      for (let k = 0; k < PLACES; k++) a[k] = (1 - P.mixReg) * a[k] + P.mixReg * d[k] * (1 - poolP[k]);
    }
  }
  return finish(ids, prob, poolP, P);
}

const REG_HORIZONS = [3, 12, 24, 48, 72];

/** Solves (A + lambda I) x = b by Gauss-Jordan elimination (A symmetric positive semi-definite, small). */
function ridgeSolve(A, b, lambda) {
  const n = b.length;
  const M = A.map((row, i) => [...row.map((v, j) => v + (i === j && i > 0 ? lambda : 0)), b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = 0; r < n; r++) {
      if (r === c) continue;
      const f = M[r][c] / d;
      if (f !== 0) for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  return M.map((row, i) => row[n] / (row[i] || 1e-12));
}

function regX(f) {
  return [1, f[0] - Math.log(10), f[1], f[2], f[3], f[4] - 4, f[5]];
}

/**
 * Fits, from the history itself, how a game's log rank moves over 3, 12, 24, 48 and 72 hours: a ridge regression of
 * the change on its features (rank, rank velocity, player growth, chart age, distance below peak), and the spread of
 * what it leaves unexplained. Games that left the chart count as rank 130.
 */
export function fitDrift(reads, P) {
  const fits = [];
  const featCache = new Map();
  const featOf = (i) => { if (!featCache.has(i)) featCache.set(i, featuresAt(reads, i)); return featCache.get(i); };
  const maxLr = Math.log(P.regMaxRank);
  for (const H of REG_HORIZONS) {
    const tol = Math.max(1.5, 0.2 * H);
    const A = Array.from({ length: 7 }, () => new Array(7).fill(0));
    const b = new Array(7).fill(0);
    const rows = [];
    for (let i = 0; i < reads.length - 1; i++) {
      let j = -1;
      let bd = Infinity;
      for (let k = i + 1; k < reads.length; k++) {
        const d = reads[k].t - reads[i].t - H;
        if (d > tol) break;
        if (Math.abs(d) <= tol && Math.abs(d) < bd) { bd = Math.abs(d); j = k; }
      }
      if (j < 0) continue;
      const w = i === 0 ? P.originGapCap : Math.min(P.originGapCap, Math.max(0.25, reads[i].t - reads[i - 1].t));
      const F = featOf(i);
      for (const id of reads[i].ids) {
        const f = F.get(id);
        if (f[0] > maxLr) continue;
        const x = regX(f);
        const y = lnRank(reads[j], id) - f[0];
        rows.push([x, y, w]);
        for (let a = 0; a < 7; a++) { b[a] += w * x[a] * y; for (let c = 0; c < 7; c++) A[a][c] += w * x[a] * x[c]; }
      }
    }
    if (rows.length < 60) { fits.push(null); continue; }
    const wsum = rows.reduce((s, r) => s + r[2], 0);
    const beta = ridgeSolve(A, b, P.regRidge * wsum);
    let ad = 0;
    for (const [x, y, w] of rows) { let yh = 0; for (let a = 0; a < 7; a++) yh += x[a] * beta[a]; ad += w * Math.abs(y - yh); }
    fits.push({ H, beta, sd: 1.2533 * ad / wsum });
  }
  return fits;
}

/** The regression's place probabilities for each game, averaged over the window's hours (two-Gaussian tails). */
function regressionDiffusion(reads, ids, feats, P) {
  const fits = fitDrift(reads, P);
  // nodes: horizon 0 (no move, a small spread) and every horizon with a fit
  const nodes = [{ H: 0, beta: null, sd: P.regS0 }];
  for (const f of fits) if (f) nodes.push(f);
  const out = new Map();
  for (const id of ids) {
    const x = regX(feats.get(id));
    const lr = feats.get(id)[0];
    const mu = nodes.map((n) => (n.beta ? n.beta.reduce((s, v, a) => s + v * x[a], 0) : 0));
    const acc = new Float64Array(PLACES);
    let cnt = 0;
    for (let h = 1.5; h < 72; h += 3) {
      let a = 0;
      while (a < nodes.length - 2 && nodes[a + 1].H < h) a++;
      const n0 = nodes[a];
      const n1 = nodes[Math.min(a + 1, nodes.length - 1)];
      const u = n1.H > n0.H ? Math.min(1, Math.max(0, (h - n0.H) / (n1.H - n0.H))) : 0;
      const m = lr + (1 - u) * mu[a] + u * mu[Math.min(a + 1, nodes.length - 1)];
      const s = P.regSdScale * ((1 - u) * n0.sd + u * n1.sd);
      let lo = 0;
      for (let k = 0; k < PLACES; k++) {
        const z = Math.log(k + 1.5) - m;
        const hi = (1 - P.regTail) * phi(z / s) + P.regTail * phi(z / (2.5 * s));
        acc[k] += hi - lo;
        lo = hi;
      }
      cnt++;
    }
    for (let k = 0; k < PLACES; k++) acc[k] /= cnt;
    out.set(id, acc);
  }
  return out;
}

/** Standard normal CDF (Abramowitz-Stegun 7.1.26 on erf; error below 1.5e-7). */
function phi(x) {
  const z = Math.abs(x) / Math.SQRT2;
  const t = 1 / (1 + 0.3275911 * z);
  const e = 1 - ((((1.061405429 * t - 1.453152027) * t + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-z * z);
  return x >= 0 ? 0.5 * (1 + e) : 0.5 * (1 - e);
}

/**
 * A parametric second opinion: each game's log rank drifts (games sink as they age, recent climbers keep climbing
 * for a while) and spreads with the horizon; its place probabilities are the Gaussian mass on each place's log-rank
 * interval, averaged over the window's hours. Places are re-normalised later, with the other component.
 */
function diffusion(ids, feats, P) {
  const out = new Map();
  for (const id of ids) {
    const [lr, v24, v6] = feats.get(id);
    const acc = new Float64Array(PLACES);
    let n = 0;
    for (let h = 1.5; h < 72; h += 3) {
      const m = lr + P.dDrift * h + P.dMeanRev * (1 - Math.exp(-h / 24)) * (Math.log(15) - lr)
        - P.dMom24 * v24 * Math.min(h, P.dMomHours) / 24 - P.dMom6 * v6 * Math.min(h, 6) / 6;
      const s = Math.sqrt(P.dS0 * P.dS0 + P.dS1 * P.dS1 * h);
      let lo = 0;
      for (let k = 0; k < PLACES; k++) {
        const hi = phi((Math.log(k + 1.5) - m) / s);
        acc[k] += hi - lo;
        lo = hi;
      }
      n++;
    }
    for (let k = 0; k < PLACES; k++) acc[k] /= n;
    out.set(id, acc);
  }
  return out;
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
    // soft cap: a share above capFrom keeps only capSlope of its excess, the rest goes to the other games
    if (P.capSlope < 1) {
      let cut = 0;
      let restSum = 0;
      for (let n = 0; n < raw.length; n++) {
        const q = (raw[n] / s) * named;
        if (q > P.capFrom) {
          const q2 = P.capFrom + (q - P.capFrom) * P.capSlope;
          cut += q - q2;
          raw[n] = (q2 / named) * s;
        } else restSum += raw[n];
      }
      if (cut > 0 && restSum > 0) {
        const scale = 1 + (cut / named) * s / restSum;
        for (let n = 0; n < raw.length; n++) if ((raw[n] / s) * named <= P.capFrom) raw[n] *= scale;
      }
      s = raw.reduce((a, b) => a + b, 0);
    }
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
