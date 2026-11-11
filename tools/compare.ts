// Compares parameter variants on two disjoint halves of the origins, so a change is kept only when it helps on both.
//   node tools/compare.ts <history.ndjson> --split 2026-11-06T00:00:00Z [--from 2026-11-02T00:00:00Z] '{"temp":1.1}' ...
// The first variant is always the defaults ({}).

import { readFileSync } from 'node:fs';
import { parseHistory, DEFAULTS } from '../main.ts';
import { backtest } from './backtest.ts';

const args = process.argv.slice(2);
const get = (k) => { const i = args.indexOf(k); return i >= 0 ? args[i + 1] : undefined; };
const reads = parseHistory(readFileSync(args[0], 'utf8'));
const from = get('--from') ?? '2026-11-02T00:00:00Z';
const split = get('--split') ?? '2026-11-06T00:00:00Z';
const variants = ['{}', ...args.slice(1).filter((a, i, all) => !a.startsWith('--') && !(all[i - 1] ?? '').startsWith('--'))];
let base = null;
for (const v of variants) {
  const p = { ...DEFAULTS, ...JSON.parse(v) };
  const a = backtest(reads, p, { from, to: split });
  const b = backtest(reads, p, { from: split });
  const row = [a.score, b.score, (a.score * a.rows.length + b.score * b.rows.length) / (a.rows.length + b.rows.length)];
  if (!base) base = row;
  const d = row.map((x, i) => x - base[i]);
  console.log(`A ${row[0].toFixed(3)} (${d[0] >= 0 ? '+' : ''}${d[0].toFixed(3)})  B ${row[1].toFixed(3)} (${d[1] >= 0 ? '+' : ''}${d[1].toFixed(3)})  all ${row[2].toFixed(3)}  [${a.rows.length}/${b.rows.length}]  ${v}`);
}
