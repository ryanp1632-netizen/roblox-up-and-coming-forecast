// Where the log score is lost: for each scored place, the actual game's origin class (its rank at the origin, or
// absent), the mean ln p it got, and how often each class occurs.
//   node tools/diagnose.ts <history.ndjson> [--params '{...}'] [--from ISO]

import { readFileSync } from 'node:fs';
import { parseHistory, forecast, DEFAULTS, PLACES } from '../main.ts';
import { windowTargets } from './backtest.ts';

const args = process.argv.slice(2);
const get = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined; };
const params = { ...DEFAULTS, ...(get('--params') ? JSON.parse(get('--params')) : {}) };
const reads = parseHistory(readFileSync(args[0], 'utf8'));
const from = get('--from') ? Date.parse(get('--from')) / 3600000 : -Infinity;
const tEnd = reads[reads.length - 1].t;
const cls = (r) => (r === undefined ? 'absent' : r <= 3 ? '1-3' : r <= 10 ? '4-10' : r <= 20 ? '11-20' : r <= 40 ? '21-40' : '41+');
const agg = new Map();
let nReads = 0;
for (let i = 1; i < reads.length - 1; i++) {
  if (reads[i].t < from || tEnd - reads[i].t < 48) continue;
  const f = forecast(reads.slice(0, i + 1), params);
  const named = Array.from({ length: PLACES }, () => new Map());
  for (const r of f.probabilities) named[r.place - 1].set(r.universe_id, r.p);
  for (const tr of windowTargets(reads, i)) {
    nReads++;
    for (let k = 0; k < PLACES; k++) {
      const id = tr.ids[k];
      const at = named[k];
      let p = at.get(id);
      if (p === undefined) {
        let s = 0;
        for (const v of at.values()) s += v;
        let u = 0;
        for (const c of tr.ids) if (!at.has(c)) u++;
        p = Math.max(0, 1 - s) / u;
      }
      const c = cls(reads[i].rank.get(id));
      const a = agg.get(c) || { n: 0, ll: 0 };
      a.n++;
      a.ll += Math.log(Math.max(p, 1e-4));
      agg.set(c, a);
    }
  }
}
for (const [c, a] of [...agg].sort()) console.log(c.padEnd(7), 'share', (a.n / nReads / PLACES).toFixed(3), 'mean ln p', (a.ll / a.n).toFixed(3), 'loss/read', (a.ll / nReads).toFixed(3));
