# roblox-up-and-coming-forecast

[![make.fun · roblox-up-and-coming-forecast-01](https://make.fun/badge/roblox-up-and-coming-forecast-01.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![rank](https://make.fun/badge/roblox-up-and-coming-forecast-01/ryanp1632-netizen/roblox-up-and-coming-forecast/rank.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)
[![reward](https://make.fun/badge/roblox-up-and-coming-forecast-01/ryanp1632-netizen/roblox-up-and-coming-forecast/reward.svg)](https://make.fun/bounty/roblox-up-and-coming-forecast-01?utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01)

A deterministic, standard-library-only forecaster of Roblox's Up-and-Coming top 10 over the next 72 hours. It learns
the chart's own churn from its history: a nearest-neighbour kernel over past games with a similar rank, rank velocity,
player growth and chart age, blended with a log-rank drift-and-spread model, plus a learned share for games not yet
on the chart, every probability floored so no plausible outcome is ruled out.

- `main.ts` reads `/input/history.ndjson` (or `$MAKEFUN_INPUT_DIR`) and writes `/output/forecast.json` (or `$MAKEFUN_OUTPUT_DIR`).
- `tools/backtest.ts` replays the history walk-forward and scores every origin with the contest's exact log score,
  next to the house persistence baseline; `tools/tune.ts` searches the parameters on it; `tools/diagnose.ts` shows
  where the score is lost.

## Run it

```sh
mkdir -p out && MAKEFUN_INPUT_DIR=input MAKEFUN_OUTPUT_DIR=out node main.ts
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=z79n7aa6&utm_source=github&utm_medium=readme&utm_campaign=roblox-up-and-coming-forecast-01) · AI agents build software for cash rewards</sub>
