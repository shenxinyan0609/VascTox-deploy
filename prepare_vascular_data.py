# -*- coding: utf-8 -*-

import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent

train_path = BASE_DIR / "train_vascular.csv"
test_path = BASE_DIR / "test_vascular.csv"

out_train = BASE_DIR / "train.xlsx"
out_test = BASE_DIR / "test.xlsx"

print("Current directory:", BASE_DIR)

if not train_path.exists():
    raise FileNotFoundError(f"Missing file: {train_path}")

if not test_path.exists():
    raise FileNotFoundError(f"Missing file: {test_path}")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

required_cols = ["canonical_smiles", "Label"]

for name, df in [("train", train_df), ("test", test_df)]:
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{name} file lacks required column: {col}")

def convert(df):
    out = df.copy()
    out["SMILES"] = out["canonical_smiles"]

    keep_cols = [
        "SMILES",
        "Label",
        "original_smiles",
        "canonical_smiles",
        "cc50_uM",
        "inchikey",
        "split"
    ]

    return out[keep_cols]

train_out = convert(train_df)
test_out = convert(test_df)

train_out.to_excel(out_train, index=False)
test_out.to_excel(out_test, index=False)

print("Done.")
print(f"train.xlsx saved to: {out_train}")
print(f"test.xlsx saved to: {out_test}")

print("\nTrain shape:", train_out.shape)
print("Test shape:", test_out.shape)

print("\nTrain label distribution:")
print(train_out["Label"].value_counts())

print("\nTest label distribution:")
print(test_out["Label"].value_counts())
