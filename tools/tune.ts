// Coordinate search of the forecaster's parameters on the walk-forward backtest.
//   node tools/tune.ts <history.ndjson> [--from ISO] [--rounds 2] [--start '{...}']

import { readFileSync } from 'node:fs';
import { parseHistory, DEFAULTS } from '../main.ts';
import { backtest } from './backtest.ts';

const args = process.argv.slice(2);
const get = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined; };
const reads = parseHistory(readFileSync(args[0], 'utf8'));
const from = get('--from') ?? '2026-11-02T00:00:00Z';
const rounds = Number(get('--rounds') ?? 2);
let best = { ...DEFAULTS, ...(get('--start') ? JSON.parse(get('--start')) : {}) };
const evalP = (p) => backtest(reads, p, { from }).score;
let bestS = evalP(best);
console.log('start', bestS.toFixed(4));
const grid = {
  bwRank: [0.15, 0.25, 0.35, 0.5, 0.8],
  bwV24: [0.3, 0.6, 1.0, 2.0, 50],
  bwV6: [0.3, 0.5, 1.0, 2.0, 50],
  bwG24: [0.3, 0.5, 0.8, 1.5, 50],
  bwAge: [0.3, 0.6, 1.0, 2.0, 50],
  bwPeak: [0.2, 0.4, 0.8, 1.5, 50],
  recencyHalfLife: [48, 96, 120, 240, 1000],
  priorWeight: [0.1, 0.5, 1, 2, 4],
  floor: [0.0002, 0.0005, 0.0008, 0.0015, 0.003],
  poolFloor: [0.01, 0.02, 0.04],
  poolScale: [0.7, 0.85, 1.0, 1.15, 1.3],
  temp: [0.7, 0.85, 1.0, 1.15],
  mixDiffusion: [0, 0.15, 0.3, 0.5, 0.7],
  dDrift: [0, 0.01, 0.02, 0.03, 0.045],
  dMeanRev: [0, 0.2, 0.4, 0.6, 0.8],
  dMom24: [0, 0.15, 0.3, 0.5, 0.8],
  dMomHours: [12, 24, 48],
  dMom6: [0, 0.2, 0.5],
  dS0: [0.1, 0.25, 0.4, 0.6],
  dS1: [0.06, 0.09, 0.12, 0.18, 0.25],
  mixReg: [0, 0.2, 0.35, 0.5, 0.65],
  regMaxRank: [30, 60, 100],
  regRidge: [0.001, 0.01, 0.05],
  regS0: [0.05, 0.15, 0.3],
  regSdScale: [0.8, 0.9, 1.0, 1.1],
  regTail: [0, 0.1, 0.2, 0.35],
};
const only = get('--only') ? get('--only').split(',') : null;
for (let r = 0; r < rounds; r++) {
  for (const [k, vals] of Object.entries(grid)) {
    if (only && !only.includes(k)) continue;
    for (const v of vals) {
      if (v === best[k]) continue;
      const p = { ...best, [k]: v };
      const s = evalP(p);
      if (s > bestS + 1e-4) { bestS = s; best = p; console.log(`  ${k}=${v} -> ${s.toFixed(4)}`); }
    }
  }
  console.log('round', r + 1, bestS.toFixed(4), JSON.stringify(Object.fromEntries(Object.keys(grid).map((k) => [k, best[k]]))));
}
