"""Probe free gold data sources and report what is actually available."""

from __future__ import annotations

import pandas as pd
import yfinance as yf

PROBES = [
    ("GC=F", "1d", "max"),
    ("GC=F", "1h", "730d"),
    ("GC=F", "5m", "60d"),
    ("GLD", "1d", "max"),
    ("GLD", "1h", "730d"),
    ("GLD", "5m", "60d"),
]


def probe(ticker: str, interval: str, period: str) -> dict:
    row = {
        "ticker": ticker,
        "interval": interval,
        "bars": 0,
        "first": "-",
        "last": "-",
        "zero_volume_%": "-",
        "note": "",
    }
    try:
        df = yf.Ticker(ticker).history(period=period, interval=interval, auto_adjust=False)
    except Exception as exc:  # network errors, rate limits, scraper breakage
        row["note"] = f"error: {str(exc)[:50]}"
        return row

    if df.empty:
        row["note"] = "no data returned"
        return row

    row["bars"] = len(df)
    row["first"] = str(df.index.min())[:16]
    row["last"] = str(df.index.max())[:16]
    row["zero_volume_%"] = round(float((df["Volume"] == 0).mean() * 100), 1)
    if df.index.has_duplicates:
        row["note"] = "duplicate timestamps"
    return row


def main() -> None:
    rows = [probe(*p) for p in PROBES]
    table = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    print(table.to_string(index=False))


if __name__ == "__main__":
    main()
