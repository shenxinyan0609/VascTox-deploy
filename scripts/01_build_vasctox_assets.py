# -*- coding: utf-8 -*-

import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors
from rdkit import RDLogger

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")


# =========================================================
# 1. Project paths
# =========================================================
PROJECT_DIR = Path(__file__).resolve().parents[1]

TRAIN_CSV = PROJECT_DIR / "train_vascular.csv"
TEST_CSV = PROJECT_DIR / "test_vascular.csv"

TRAIN_XLSX = PROJECT_DIR / "train.xlsx"
TEST_XLSX = PROJECT_DIR / "test.xlsx"

DATA_DIR = PROJECT_DIR / "data"
MODEL_DIR = PROJECT_DIR / "models"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

MODEL_PATH = MODEL_DIR / "best_model.pkl"
MODEL_INFO_PATH = MODEL_DIR / "best_model_info.json"

AD_OUTPUT_PATH = DATA_DIR / "AD_train_test_result_checked.xlsx"
TOP20_PATH = DATA_DIR / "top20_toxic.csv"

RANDOM_STATE = 42
AD_PERCENTILE = 0.05


# =========================================================
# 2. Basic functions
# =========================================================
def standardize_smiles(smiles):
    if pd.isna(smiles):
        return None, None, "Empty SMILES"

    smi = str(smiles).strip()

    if smi == "":
        return None, None, "Empty SMILES"

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None, None, "Invalid SMILES"

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    return canonical_smiles, mol, "Valid"


try:
    from rdkit.Chem import rdFingerprintGenerator

    MORGAN_GENERATOR = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=2048
    )

    def mol_to_fp(mol):
        return MORGAN_GENERATOR.GetFingerprint(mol)

except ImportError:
    from rdkit.Chem import AllChem

    def mol_to_fp(mol):
        return AllChem.GetMorganFingerprintAsBitVect(
            mol,
            radius=2,
            nBits=2048
        )


def get_descriptor_names():
    return [name for name, func in Descriptors._descList]


def calc_descriptor_df(mols, desc_names):
    desc_func_dict = dict(Descriptors._descList)
    records = []

    for mol in mols:
        values = []

        for name in desc_names:
            func = desc_func_dict.get(name)

            if func is None:
                values.append(np.nan)
                continue

            try:
                value = func(mol)
                if value is None or isinstance(value, str):
                    value = np.nan
                else:
                    value = float(value)
            except Exception:
                value = np.nan

            values.append(value)

        records.append(values)

    desc_df = pd.DataFrame(records, columns=desc_names, dtype=np.float64)
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)

    return desc_df


def preprocess_train_descriptors(desc_df):
    all_nan_cols = desc_df.columns[desc_df.isna().all()].tolist()
    desc_df = desc_df.drop(columns=all_nan_cols, errors="ignore")

    medians = desc_df.median(numeric_only=True)
    desc_df = desc_df.fillna(medians)

    finite_mask = np.isfinite(desc_df.values).all(axis=0)
    finite_cols = desc_df.columns[finite_mask].tolist()
    desc_df = desc_df[finite_cols]

    max_abs = desc_df.abs().max(axis=0)
    huge_cols = max_abs[max_abs > 1e10].index.tolist()
    desc_df = desc_df.drop(columns=huge_cols, errors="ignore")

    nunique = desc_df.nunique(dropna=False)
    keep_cols = nunique[nunique > 1].index.tolist()
    desc_df = desc_df[keep_cols]

    X = desc_df.values.astype(np.float64)

    prep = {
        "all_nan_cols": all_nan_cols,
        "medians": medians.to_dict(),
        "finite_cols": finite_cols,
        "huge_cols": huge_cols,
        "keep_cols": keep_cols,
    }

    return X, prep


def preprocess_test_descriptors(desc_df, prep):
    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.drop(columns=prep["all_nan_cols"], errors="ignore")
    desc_df = desc_df.fillna(prep["medians"])
    desc_df = desc_df.reindex(columns=prep["finite_cols"])
    desc_df = desc_df.drop(columns=prep["huge_cols"], errors="ignore")
    desc_df = desc_df.reindex(columns=prep["keep_cols"])

    try:
        desc_df = desc_df.fillna(
            pd.Series(prep["medians"]).reindex(prep["keep_cols"])
        )
    except Exception:
        pass

    desc_df = desc_df.fillna(0)

    return desc_df.values.astype(np.float64)


def evaluate_binary_model(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1]
    ).ravel()

    se = recall_score(y_true, y_pred, zero_division=0)
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan
    acc = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    ba = balanced_accuracy_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = np.nan

    return {
        "SE": float(se),
        "SP": float(sp),
        "ACC": float(acc),
        "P": float(precision),
        "F1": float(f1),
        "BA": float(ba),
        "MCC": float(mcc),
        "AUC": float(auc),
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
    }


# =========================================================
# 3. Convert CSV to website train/test Excel files
# =========================================================
print("Project directory:", PROJECT_DIR)

if not TRAIN_CSV.exists():
    raise FileNotFoundError(f"Missing file: {TRAIN_CSV}")

if not TEST_CSV.exists():
    raise FileNotFoundError(f"Missing file: {TEST_CSV}")

train_raw_csv = pd.read_csv(TRAIN_CSV)
test_raw_csv = pd.read_csv(TEST_CSV)

required_cols = ["canonical_smiles", "Label"]

for name, df in [("train", train_raw_csv), ("test", test_raw_csv)]:
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"{name} CSV lacks required column: {col}")


def convert_to_website_format(df):
    out = df.copy()
    out["SMILES"] = out["canonical_smiles"]

    keep_cols = [
        "SMILES",
        "Label",
        "original_smiles",
        "canonical_smiles",
        "cc50_uM",
        "inchikey",
        "split",
    ]

    keep_cols = [col for col in keep_cols if col in out.columns]

    return out[keep_cols]


train_excel = convert_to_website_format(train_raw_csv)
test_excel = convert_to_website_format(test_raw_csv)

train_excel.to_excel(TRAIN_XLSX, index=False)
test_excel.to_excel(TEST_XLSX, index=False)

print("\nSaved website train/test files:")
print(TRAIN_XLSX)
print(TEST_XLSX)

print("\nTrain shape:", train_excel.shape)
print("Test shape:", test_excel.shape)

print("\nTrain label distribution:")
print(train_excel["Label"].value_counts())

print("\nTest label distribution:")
print(test_excel["Label"].value_counts())


# =========================================================
# 4. Load train/test and standardize molecules
# =========================================================
train_raw = pd.read_excel(TRAIN_XLSX)
test_raw = pd.read_excel(TEST_XLSX)

for name, df in [("train", train_raw), ("test", test_raw)]:
    for col in ["SMILES", "Label"]:
        if col not in df.columns:
            raise ValueError(f"{name} file lacks required column: {col}")


def prepare_molecule_table(df, name):
    rows = []
    invalid_count = 0

    for _, row in df.iterrows():
        canonical_smiles, mol, status = standardize_smiles(row["SMILES"])

        if status != "Valid":
            invalid_count += 1
            continue

        new_row = row.to_dict()
        new_row["Canonical_SMILES_for_model"] = canonical_smiles
        new_row["valid_smiles"] = True
        new_row["Mol"] = mol
        rows.append(new_row)

    out = pd.DataFrame(rows)

    print(f"\n{name}: raw rows = {len(df)}")
    print(f"{name}: valid molecules = {len(out)}")
    print(f"{name}: invalid SMILES = {invalid_count}")

    return out, invalid_count


train_df, train_invalid = prepare_molecule_table(train_raw, "train")
test_df, test_invalid = prepare_molecule_table(test_raw, "test")

y_train = train_df["Label"].astype(int).values
y_test = test_df["Label"].astype(int).values

train_mols = train_df["Mol"].tolist()
test_mols = test_df["Mol"].tolist()


# =========================================================
# 5. Train Descriptors-RF model
# =========================================================
print("\nCalculating RDKit descriptors...")

desc_names = get_descriptor_names()

train_desc = calc_descriptor_df(train_mols, desc_names)
test_desc = calc_descriptor_df(test_mols, desc_names)

X_train, prep = preprocess_train_descriptors(train_desc)
X_test = preprocess_test_descriptors(test_desc, prep)

print("X_train:", X_train.shape)
print("X_test:", X_test.shape)

print("\nTraining Descriptors-RF model...")

model = RandomForestClassifier(
    n_estimators=500,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    max_features="sqrt"
)

model.fit(X_train, y_train)

train_prob = model.predict_proba(X_train)[:, 1]
test_prob = model.predict_proba(X_test)[:, 1]

train_metrics = evaluate_binary_model(y_train, train_prob, threshold=0.5)
test_metrics = evaluate_binary_model(y_test, test_prob, threshold=0.5)

print("\n========== Train Metrics ==========")
for k, v in train_metrics.items():
    print(f"{k}: {v}")

print("\n========== Test Metrics ==========")
for k, v in test_metrics.items():
    print(f"{k}: {v}")


# =========================================================
# 6. Save model package
# =========================================================
model_package = {
    "model_type": "Descriptors-RF",
    "model": model,
    "desc_names": desc_names,
    "all_nan_cols": prep["all_nan_cols"],
    "medians": prep["medians"],
    "finite_cols": prep["finite_cols"],
    "huge_cols": prep["huge_cols"],
    "keep_cols": prep["keep_cols"],
    "classification_threshold": 0.5,
    "task": "vascular_toxicity_prediction",
    "label_definition": "Label=1 indicates vascular toxicity under the single 10 uM criterion",
    "random_state": RANDOM_STATE,
}

joblib.dump(model_package, MODEL_PATH)

model_info = {
    "model_type": "Descriptors-RF",
    "task": "vascular_toxicity_prediction",
    "train_valid_molecules": int(len(train_df)),
    "test_valid_molecules": int(len(test_df)),
    "n_features": int(X_train.shape[1]),
    "classification_threshold": 0.5,
    "train_metrics": train_metrics,
    "test_metrics": test_metrics,
}

with open(MODEL_INFO_PATH, "w", encoding="utf-8") as f:
    json.dump(model_info, f, indent=2, ensure_ascii=False)

print("\nSaved model:")
print(MODEL_PATH)

print("Saved model info:")
print(MODEL_INFO_PATH)


# =========================================================
# 7. Applicability Domain analysis
# =========================================================
print("\nRunning applicability domain analysis...")

train_fps = [mol_to_fp(mol) for mol in train_mols]
test_fps = [mol_to_fp(mol) for mol in test_mols]

train_nn_sims = []

for i, fp in enumerate(train_fps):
    sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    sims[i] = -1.0
    train_nn_sims.append(max(sims))

ad_threshold = float(pd.Series(train_nn_sims).quantile(AD_PERCENTILE))

print(f"AD threshold = {ad_threshold:.6f}")
print(f"AD percentile = {AD_PERCENTILE * 100:.1f}th percentile")

max_sims = []
nearest_smiles = []
nearest_labels = []
nearest_idx_list = []

for fp in test_fps:
    sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    nearest_idx = int(np.argmax(sims))
    max_sim = float(sims[nearest_idx])

    max_sims.append(max_sim)
    nearest_idx_list.append(nearest_idx)
    nearest_smiles.append(train_df.iloc[nearest_idx]["SMILES"])
    nearest_labels.append(train_df.iloc[nearest_idx]["Label"])

test_output = test_df.drop(columns=["Mol"], errors="ignore").copy()
test_output["Pred_probability"] = test_prob
test_output["Pred_label"] = (test_prob >= 0.5).astype(int)
test_output["AD_max_Tanimoto"] = max_sims
test_output["AD_status"] = np.where(
    test_output["AD_max_Tanimoto"] >= ad_threshold,
    "Inside AD",
    "Outside AD"
)
test_output["nearest_train_SMILES"] = nearest_smiles
test_output["nearest_train_Label"] = nearest_labels
test_output["nearest_train_index"] = nearest_idx_list

inside_count = int((test_output["AD_status"] == "Inside AD").sum())
outside_count = int((test_output["AD_status"] == "Outside AD").sum())

summary_df = pd.DataFrame({
    "raw_train_rows": [len(train_raw)],
    "raw_test_rows": [len(test_raw)],
    "train_valid_molecules": [len(train_df)],
    "test_valid_molecules": [len(test_df)],
    "train_invalid_SMILES": [train_invalid],
    "test_invalid_SMILES": [test_invalid],
    "AD_percentile": [AD_PERCENTILE],
    "AD_threshold": [ad_threshold],
    "inside_AD_count": [inside_count],
    "outside_AD_count": [outside_count],
    "inside_AD_ratio": [inside_count / len(test_output)],
    "outside_AD_ratio": [outside_count / len(test_output)],
    "model_type": ["Descriptors-RF"],
    "test_AUC": [test_metrics["AUC"]],
    "test_ACC": [test_metrics["ACC"]],
    "test_MCC": [test_metrics["MCC"]],
})

train_nn_df = pd.DataFrame({
    "train_nearest_neighbor_similarity": train_nn_sims
})

metrics_df = pd.DataFrame([test_metrics])
metrics_df.insert(0, "Model", "Descriptors-RF")

with pd.ExcelWriter(AD_OUTPUT_PATH, engine="openpyxl") as writer:
    test_output.to_excel(writer, sheet_name="test_AD_result", index=False)
    summary_df.to_excel(writer, sheet_name="AD_summary", index=False)
    train_nn_df.to_excel(writer, sheet_name="train_NN_similarity", index=False)
    metrics_df.to_excel(writer, sheet_name="model_metrics", index=False)

print("\nSaved AD result:")
print(AD_OUTPUT_PATH)


# =========================================================
# 8. Save Top 20 high-risk molecules
# =========================================================
top20_df = test_output.sort_values("Pred_probability", ascending=False).head(20).copy()

top20_df = top20_df.rename(columns={
    "Pred_probability": "pred_prob",
    "Pred_label": "pred_label",
})

top20_cols = []

for col in [
    "SMILES",
    "Label",
    "pred_prob",
    "pred_label",
    "cc50_uM",
    "AD_status",
    "AD_max_Tanimoto",
    "nearest_train_SMILES",
    "nearest_train_Label",
    "valid_smiles",
]:
    if col in top20_df.columns:
        top20_cols.append(col)

top20_df[top20_cols].to_csv(
    TOP20_PATH,
    index=False,
    encoding="utf-8-sig"
)

print("\nSaved Top 20 toxic molecules:")
print(TOP20_PATH)

print("\nAll done.")