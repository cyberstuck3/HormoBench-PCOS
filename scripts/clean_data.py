"""
Stage 1 — Clean the raw PCOS xlsx into a benchmark-ready CSV.

Deterministic: no imputation, no target-aware operations. Imputation happens
later, fit only on the training fold, per the anti-leakage architecture in
docs/METHODOLOGY.md.
"""
import sys
from pathlib import Path

# Make src/hormobench importable when running the script directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from hormobench import DATA_RAW, CLEAN_CSV

RAW_XLSX = DATA_RAW / "PCOS_data_without_infertility.xlsx"


def clean() -> pd.DataFrame:
    df = pd.ExcelFile(RAW_XLSX).parse("Full_new")
    print(f"Loaded raw data: {df.shape[0]} rows, {df.shape[1]} columns")

    # 1. Strip whitespace from column names and collapse internal doubles
    df.columns = [" ".join(c.split()).strip() for c in df.columns]

    # 2. Drop junk column + non-feature identifiers
    drop_cols = ["Unnamed: 44", "Sl. No", "Patient File No."]
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # 3. Fix two known typos in object columns; coerce to numeric
    col_beta = "II beta-HCG(mIU/mL)"
    col_amh  = "AMH(ng/mL)"
    df[col_beta] = pd.to_numeric(df[col_beta].replace("1.99.", "1.99"), errors="coerce")
    df[col_amh]  = pd.to_numeric(df[col_amh],  errors="coerce")   # 'a' → NaN

    # 4. Assert all columns numeric
    non_num = df.select_dtypes(exclude="number").columns.tolist()
    assert not non_num, f"Non-numeric columns remain: {non_num}"

    print(f"\nCleaned shape       : {df.shape}")
    print(f"PCOS prevalence     : {df['PCOS (Y/N)'].mean()*100:.1f}%")
    print(f"Remaining NaN cells : {int(df.isna().sum().sum())}  "
          "(handled later inside the training Pipeline)")
    return df


def main():
    df = clean()
    CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEAN_CSV, index=False)
    print(f"\n✓ Written: {CLEAN_CSV}")


if __name__ == "__main__":
    main()
