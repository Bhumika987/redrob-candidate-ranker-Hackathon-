"""
csv_to_xlsx.py
Convert submission.csv to submission.xlsx with no value changes.

Usage
-----
    python redrob-ranker/csv_to_xlsx.py
    python redrob-ranker/csv_to_xlsx.py --in ./submission.csv --out ./submission.xlsx
"""

import argparse
from pathlib import Path

import pandas as pd

def main() -> None:
    ap = argparse.ArgumentParser(description="Convert submission CSV to XLSX.")
    ap.add_argument("--in",  dest="csv_path",  default="./submission.csv",  metavar="PATH")
    ap.add_argument("--out", dest="xlsx_path", default="./submission.xlsx", metavar="PATH")
    args = ap.parse_args()

    csv_path  = Path(args.csv_path)
    xlsx_path = Path(args.xlsx_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"{csv_path} not found")

    df = pd.read_csv(
        csv_path,
        dtype={"candidate_id": str, "rank": int, "score": str, "reasoning": str},
        keep_default_na=False,
    )

    expected = ["candidate_id", "rank", "score", "reasoning"]
    if list(df.columns) != expected:
        raise ValueError(f"Unexpected columns: {list(df.columns)}")

    df.to_excel(xlsx_path, index=False, engine="openpyxl")
    print(f"Written: {xlsx_path}  ({len(df)} rows)")


if __name__ == "__main__":
    main()
