"""
Metabolomics track, Stage 1: clean Table-S1 (clinical) + Table-S2 (follicular-fluid
amino acids) into one benchmark-ready CSV.

The cohort (37 controls, 35 PCOS) is separate from the Kerala cohort and has
almost no feature overlap with it, so it gets its own CSV, split and results.

Dropped on purpose:
  * Estrous cyclicity, Ovarian morphology: Rotterdam diagnostic criteria that
    separate the classes perfectly (label leakage).
  * L-Glycine, L-Alanine, L-Cysteine: not measured for any participant ("/").
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd
from hormobench import META_RAW_DIR, META_CLEAN_CSV

TARGET = "PCOS (Y/N)"
LEAKY = ["Estrous cyclicity", "Ovarian morphology"]


def _clinical(sheet: str) -> pd.DataFrame:
    df = pd.read_excel(META_RAW_DIR / "Table-S1.xlsx", sheet_name=sheet, header=0)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])
    df.columns = [" ".join(str(c).split()) for c in df.columns]
    df = df.rename(columns={"Case No.": "Case ID"}).dropna(subset=["Case ID"])
    return df.drop(columns=LEAKY)


def _amino_acids(sheet: str) -> pd.DataFrame:
    raw = pd.read_excel(META_RAW_DIR / "Table-S2.xlsx", sheet_name=sheet, header=None)
    names = raw.iloc[0, 1:].tolist()
    body = raw.iloc[5:].dropna(subset=[0])   # rows 1-4 are formula/mass/RT/CAS
    df = pd.DataFrame(body.iloc[:, 1:].values, columns=names)
    df = df.replace("/", pd.NA).apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")        # unmeasured compounds
    df.insert(0, "Case ID", body.iloc[:, 0].astype(str).values)
    return df


def clean() -> pd.DataFrame:
    parts = []
    for sheet, label in [("CON", 0), ("PCOS", 1)]:
        clin, aa = _clinical(sheet), _amino_acids(sheet)
        assert clin["Case ID"].tolist() == aa["Case ID"].tolist(), f"ID mismatch in {sheet}"
        merged = clin.merge(aa, on="Case ID", validate="one_to_one")
        merged.insert(0, TARGET, label)
        parts.append(merged)

    df = pd.concat(parts, ignore_index=True).drop(columns=["Case ID"])
    df = df.apply(pd.to_numeric, errors="raise")

    print(f"Cleaned shape       : {df.shape}")
    print(f"PCOS prevalence     : {df[TARGET].mean()*100:.1f}%")
    print(f"Remaining NaN cells : {int(df.isna().sum().sum())}")
    return df


def main():
    df = clean()
    META_CLEAN_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(META_CLEAN_CSV, index=False)
    print(f"\n✓ Written: {META_CLEAN_CSV}")


if __name__ == "__main__":
    main()
