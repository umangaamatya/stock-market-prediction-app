#!/usr/bin/env python3
"""
Run the ETL pipeline for the Stock Market Prediction App.
Usage:
    python scripts/run_etl.py [--mode full|incremental] [--period 1y|2y|6m] [--no-news]

Defaults:
    --mode full
    --period 1y
    include news by default
"""
import argparse
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from etl.etl_pipeline import ETLPipeline


def main():
    parser = argparse.ArgumentParser(description="Run ETL Pipeline")
    parser.add_argument(
        "--mode",
        choices=["full", "incremental"],
        default="full",
        help="Pipeline mode",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Historical period for full pipeline (e.g., 6m, 1y, 2y)",
    )
    parser.add_argument(
        "--no-news",
        action="store_true",
        help="Disable news extraction during full pipeline",
    )

    args = parser.parse_args()

    etl = ETLPipeline()

    if args.mode == "full":
        result = etl.run_full_pipeline(period=args.period, include_news=not args.no_news)
    else:
        result = etl.run_incremental_update()

    # Print concise result summary
    success = result.get("success")
    errors = result.get("errors", [])
    summary = result.get("data_summary", {})

    print(f"success={success}")
    if summary:
        print(f"data_summary={summary}")
    if errors:
        print("errors:")
        for e in errors:
            print(f" - {e}")


if __name__ == "__main__":
    main()

