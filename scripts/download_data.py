"""Download gold data from yfinance, clean it, and save it with a hash."""

from __future__ import annotations

import argparse

import pandas as pd
import yfinance as yf

from stratiq.data.clean import clean
from stratiq.data.store import save_dataset


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ticker", default="GC=F")
    p.add_argument("--interval", default="1h")
    p.add_argument("--period", default="730d")
    p.add_argument("--name", default="gc_1h")
    p.add_argument("--out", default="data/parquet")
    p.add_argument("--start", default=None, help="drop rows before this date, e.g. 2012-01-01")
    args = p.parse_args()

    raw = yf.Ticker(args.ticker).history(
        period=args.period, interval=args.interval, auto_adjust=False
    )
    if raw.empty:
        raise SystemExit("no data returned")
    if args.start:
        raw = raw[raw.index >= pd.Timestamp(args.start, tz=raw.index.tz)]

    daily = args.interval in ("1d", "1wk")
    df, report = clean(raw, daily=daily)
    info = save_dataset(
        df,
        args.name,
        args.out,
        source="yfinance",
        ticker=args.ticker,
        interval=args.interval,
        start=args.start,
        quality=report.to_dict(),
    )
    print("quality report:")
    for k, v in report.to_dict().items():
        print(f"  {k}: {v}")
    print(f"saved {args.out}/{args.name}.parquet  sha256={info['sha256'][:16]}...")


if __name__ == "__main__":
    main()
