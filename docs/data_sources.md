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
## Daily data cleaning decision (2026-11-23)
- Raw GC=F 1d: 441 of 6,584 rows failed OHLC validity checks, all between 2001-02 and 2011-11.
- Sample rows show open = high = low with volume of 0-6 (thin early-era prints).
- Decision: the daily series starts at 2012-01-01. Reason: no invalid rows after 2011-11.
- Hourly data: 13,757 rows, 0 invalid, 4 gaps (largest 3 days 8h = weekend/holiday closures).
