# Data sources

Checked on: <date>

| Ticker | Interval | Bars | First | Last | Zero-volume % | Notes |
|--------|----------|------|-------|------|---------------|-------|
| (paste your table here) |

## Decision
- Source: <GC=F or GLD>
- Timeframe: <1d / 1h>
- Reason: <why>

## Limitations
- yfinance is an unofficial scraper and may break; its terms limit redistribution, so data is never committed.
- GC=F is a continuous futures series and has roll artifacts. GLD is an ETF (trades in US hours only).
- Free intraday history covers only a short recent window.