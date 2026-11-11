import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import clean_metabolomics


def test_clean_metabolomics_shape_and_no_leakage():
    df = clean_metabolomics.clean()
    assert df.shape == (72, 23)
    assert df["PCOS (Y/N)"].sum() == 35
    assert not set(clean_metabolomics.LEAKY) & set(df.columns)
    assert not {"L-Glycine", "L-Alanine", "L-Cysteine"} & set(df.columns)
    assert df.isna().sum().sum() == 0
