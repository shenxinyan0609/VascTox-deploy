# app.py
# VascTox: Vascular Toxicity Prediction Platform
# Final deployed model: MorganFP-RF
# Prediction feature: Morgan fingerprint, radius=2, nBits=1024
# AD feature: Morgan fingerprint, radius=2, nBits=2048; cutoff=0.3750

import os
import io
import warnings
import textwrap
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import streamlit as st

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, Descriptors, Draw

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    accuracy_score,
    matthews_corrcoef,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)

import joblib


# ============================================================
# 1. Page config
# ============================================================

st.set_page_config(
    page_title="VascTox",
    page_icon="🧪",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# 2. Basic configuration
# ============================================================

APP_TITLE = "VascTox"
APP_VERSION = "VascTox v1.1 (final-model aligned)"

APP_DIR = Path(__file__).resolve().parent

# 你的网站文件夹里现在有 train_vascular.csv / test_vascular.csv
# 如果以后你把数据移到 data 文件夹，这里也能自动识别
TRAIN_FILE_ROOT = APP_DIR / "train_vascular.csv"
TEST_FILE_ROOT = APP_DIR / "test_vascular.csv"

TRAIN_FILE_DATA = APP_DIR / "data" / "train_vascular.csv"
TEST_FILE_DATA = APP_DIR / "data" / "test_vascular.csv"

TRAIN_FILE = TRAIN_FILE_DATA if TRAIN_FILE_DATA.exists() else TRAIN_FILE_ROOT
TEST_FILE = TEST_FILE_DATA if TEST_FILE_DATA.exists() else TEST_FILE_ROOT

MODEL_DIR = APP_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_FILE = MODEL_DIR / "morganfp1024_rf_model_final.jobli"
METRICS_FILE = MODEL_DIR / "morganfp_rf_test_metrics.csv"
TEST_PRED_FILE = MODEL_DIR / "morganfp_rf_test_predictions.csv"
AD_SUMMARY_FILE = MODEL_DIR / "morgan_tanimoto_ad_summary.csv"

RANDOM_STATE = 42
MORGAN_RADIUS = 2
MORGAN_NBITS = 1024
AD_NBITS = 2048
AD_CUTOFF = 0.3750
PREDICTION_THRESHOLD = 0.50

# 如果你想强制重新训练模型，把这里改成 True
FORCE_RETRAIN = False


# ============================================================
# 3. UI styling
# ============================================================

def inject_custom_css():
    st.markdown(
        """
<style>
/* Global */
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Helvetica Neue", Arial, sans-serif;
}

.block-container {
    padding-top: 2.0rem;
    padding-bottom: 3.0rem;
    max-width: 1320px;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #eef6ff 0%, #f8fbff 100%);
    border-right: 1px solid #dbeafe;
}

section[data-testid="stSidebar"] h1,
section[data-testid="stSidebar"] h2,
section[data-testid="stSidebar"] h3 {
    color: #0f172a;
}

/* Hero */
.hero-card {
    padding: 30px 36px;
    border-radius: 24px;
    background: linear-gradient(135deg, #0f766e 0%, #0f4c81 55%, #111827 100%);
    color: white;
    box-shadow: 0 20px 50px rgba(15, 76, 129, 0.20);
    margin-bottom: 28px;
}

.hero-card h1 {
    font-size: 3.15rem;
    font-weight: 850;
    margin: 0 0 8px 0;
    letter-spacing: -0.045em;
}

.hero-card h2 {
    font-size: 1.48rem;
    font-weight: 650;
    margin: 0 0 18px 0;
    opacity: 0.96;
}

.hero-card p {
    font-size: 1.02rem;
    line-height: 1.65;
    max-width: 1080px;
    margin: 0;
    color: rgba(255, 255, 255, 0.88);
}

.tag-row {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 20px;
}

.tag {
    display: inline-block;
    padding: 7px 12px;
    border-radius: 999px;
    background: rgba(255,255,255,0.14);
    border: 1px solid rgba(255,255,255,0.25);
    color: white;
    font-size: 0.86rem;
}

/* Metric cards */
.metric-card {
    padding: 22px 22px 18px 22px;
    border-radius: 21px;
    background: #ffffff;
    border: 1px solid #e5edf5;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.06);
    height: 142px;
    min-height: 142px;
    max-height: 142px;
    margin-bottom: 12px;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    overflow: hidden;
}

.metric-card .label {
    color: #64748b;
    font-size: 0.88rem;
    font-weight: 650;
    margin-bottom: 4px;
    white-space: normal;
}

.metric-card .value {
    color: #0f172a;
    font-size: 2.0rem;
    font-weight: 800;
    line-height: 1.05;
    letter-spacing: -0.035em;
}

.metric-card .note {
    margin-top: 6px;
    color: #64748b;
    font-size: 0.80rem;
    line-height: 1.28;
    display: -webkit-box;
    -webkit-line-clamp: 2;
    -webkit-box-orient: vertical;
    overflow: hidden;
}

.metric-card.blue {
    border: 1px solid #bfdbfe;
    background: linear-gradient(180deg, #eff6ff 0%, #ffffff 100%);
}

.metric-card.green {
    border: 1px solid #bbf7d0;
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 100%);
}

.metric-card.amber {
    border: 1px solid #fde68a;
    background: linear-gradient(180deg, #fffbeb 0%, #ffffff 100%);
}

.metric-card.red {
    border: 1px solid #fecaca;
    background: linear-gradient(180deg, #fef2f2 0%, #ffffff 100%);
}

/* Section titles */
.section-title {
    margin-top: 30px;
    margin-bottom: 14px;
}

.section-title h3 {
    font-size: 1.42rem;
    font-weight: 780;
    color: #0f172a;
    margin-bottom: 4px;
}

.section-title p {
    color: #64748b;
    margin-top: 0;
    font-size: 0.95rem;
}

/* Workflow */
.workflow-box {
    padding: 22px;
    border-radius: 21px;
    background: #ffffff;
    border: 1px solid #e5edf5;
    box-shadow: 0 10px 28px rgba(15, 23, 42, 0.05);
}

.workflow-line {
    font-weight: 700;
    color: #075985;
    font-size: 0.98rem;
    line-height: 1.75;
}

/* Info panels */
.info-panel {
    padding: 18px 20px;
    border-radius: 18px;
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    color: #334155;
    line-height: 1.65;
    margin-bottom: 14px;
}

.success-panel {
    padding: 18px 20px;
    border-radius: 18px;
    background: #f0fdf4;
    border: 1px solid #bbf7d0;
    color: #14532d;
    line-height: 1.65;
    margin-bottom: 14px;
}

.warning-panel {
    padding: 18px 20px;
    border-radius: 18px;
    background: #fffbeb;
    border: 1px solid #fde68a;
    color: #78350f;
    line-height: 1.65;
    margin-bottom: 14px;
}

.danger-panel {
    padding: 18px 20px;
    border-radius: 18px;
    background: #fef2f2;
    border: 1px solid #fecaca;
    color: #7f1d1d;
    line-height: 1.65;
    margin-bottom: 14px;
}

/* Tables */
[data-testid="stDataFrame"] {
    border-radius: 16px;
    overflow: hidden;
}

/* Buttons */
.stButton > button {
    border-radius: 999px;
    padding: 0.55rem 1.3rem;
    font-weight: 700;
    border: 1px solid #0f766e;
    background: linear-gradient(135deg, #0f766e 0%, #0f4c81 100%);
    color: white;
}

.stDownloadButton > button {
    border-radius: 999px;
    padding: 0.55rem 1.2rem;
    font-weight: 700;
}

/* Hide Streamlit default footer */
footer {
    visibility: hidden;
}
</style>
        """,
        unsafe_allow_html=True
    )


def section_title(title, subtitle=None):
    subtitle_html = f"<p>{subtitle}</p>" if subtitle else ""
    st.markdown(
        f"""
<div class="section-title">
    <h3>{title}</h3>
    {subtitle_html}
</div>
        """,
        unsafe_allow_html=True
    )


def metric_card(label, value, note="", style=""):
    st.markdown(
        f"""
<div class="metric-card {style}">
    <div class="label">{label}</div>
    <div class="value">{value}</div>
    <div class="note">{note}</div>
</div>
        """,
        unsafe_allow_html=True
    )


def info_panel(text, panel_type="info"):
    cls = {
        "info": "info-panel",
        "success": "success-panel",
        "warning": "warning-panel",
        "danger": "danger-panel",
    }.get(panel_type, "info-panel")

    st.markdown(
        f"""
<div class="{cls}">
{text}
</div>
        """,
        unsafe_allow_html=True
    )


inject_custom_css()


# ============================================================
# 4. Data utilities
# ============================================================

def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".csv":
        return pd.read_csv(path)
    elif suffix in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    else:
        raise ValueError(f"Unsupported file type: {path}")


def auto_find_columns(df: pd.DataFrame, require_label=True):
    smiles_candidates = [
        "SMILES", "smiles", "Smiles", "canonical_smiles", "Canonical_SMILES",
        "CanonicalSmiles", "canonical_smile", "structure", "Structure"
    ]

    label_candidates = [
        "Label", "label", "Y", "y", "Class", "class", "Toxic", "toxic",
        "Toxicity", "toxicity", "Activity", "activity"
    ]

    smiles_col = None
    label_col = None

    for c in smiles_candidates:
        if c in df.columns:
            smiles_col = c
            break

    for c in label_candidates:
        if c in df.columns:
            label_col = c
            break

    if smiles_col is None:
        raise ValueError(
            "Cannot identify SMILES column. "
            f"Available columns: {list(df.columns)}"
        )

    if require_label and label_col is None:
        raise ValueError(
            "Cannot identify label column. "
            f"Available columns: {list(df.columns)}"
        )

    return smiles_col, label_col


def canonicalize_smiles(smiles):
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    smiles_col, label_col = auto_find_columns(df, require_label=True)

    out = df.copy()
    out["SMILES"] = out[smiles_col].apply(canonicalize_smiles)
    out = out.dropna(subset=["SMILES"])

    out["Label"] = out[label_col].astype(int)
    out = out[out["Label"].isin([0, 1])]

    conflict = out.groupby("SMILES")["Label"].nunique()
    conflict_smiles = conflict[conflict > 1].index.tolist()

    if len(conflict_smiles) > 0:
        out = out[~out["SMILES"].isin(conflict_smiles)]

    out = out.drop_duplicates(subset=["SMILES", "Label"]).reset_index(drop=True)

    return out[["SMILES", "Label"]]


# ============================================================
# 5. Morgan fingerprint and molecule utilities
# ============================================================

def mol_from_smiles(smiles):
    try:
        return Chem.MolFromSmiles(str(smiles))
    except Exception:
        return None


def smiles_to_morgan_fp_bitvect(smiles, radius=MORGAN_RADIUS, nbits=MORGAN_NBITS):
    mol = mol_from_smiles(smiles)

    if mol is None:
        return None

    fp = AllChem.GetMorganFingerprintAsBitVect(
        mol,
        radius,
        nBits=nbits
    )

    return fp


def bitvect_to_numpy(fp, nbits=MORGAN_NBITS):
    arr = np.zeros((nbits,), dtype=np.int8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def smiles_to_morgan_array(smiles):
    fp = smiles_to_morgan_fp_bitvect(smiles)

    if fp is None:
        return np.zeros((MORGAN_NBITS,), dtype=np.int8)

    return bitvect_to_numpy(fp)


def build_morgan_matrix(smiles_list):
    return np.array([smiles_to_morgan_array(smi) for smi in smiles_list])


def compute_basic_descriptors(smiles):
    mol = mol_from_smiles(smiles)

    if mol is None:
        return None

    return {
        "Molecular weight": Descriptors.MolWt(mol),
        "LogP": Descriptors.MolLogP(mol),
        "TPSA": Descriptors.TPSA(mol),
        "H-bond acceptors": Descriptors.NumHAcceptors(mol),
        "H-bond donors": Descriptors.NumHDonors(mol),
        "Rotatable bonds": Descriptors.NumRotatableBonds(mol),
        "Ring count": Descriptors.RingCount(mol),
        "Heavy atoms": Descriptors.HeavyAtomCount(mol),
        "Fraction Csp3": Descriptors.FractionCSP3(mol),
    }


# ============================================================
# 6. Applicability domain
# ============================================================

def compute_train_nn_distribution(train_fps):
    nn_sims = []

    for i, fp in enumerate(train_fps):
        sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
        sims[i] = -1.0
        nn_sims.append(max(sims))

    return np.array(nn_sims)


def compute_ad_cutoff(train_fps):
    train_nn = compute_train_nn_distribution(train_fps)
    return AD_CUTOFF, train_nn


def assess_ad(smiles, train_fps, cutoff):
    fp = smiles_to_morgan_fp_bitvect(
        smiles,
        radius=MORGAN_RADIUS,
        nbits=AD_NBITS
    )

    if fp is None:
        return {
            "Nearest_Tanimoto": np.nan,
            "Inside_AD": False,
            "AD_Status": "Invalid SMILES"
        }

    sims = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
    nearest = float(max(sims)) if len(sims) > 0 else np.nan
    inside = bool(nearest >= cutoff)

    return {
        "Nearest_Tanimoto": nearest,
        "Inside_AD": inside,
        "AD_Status": "Inside AD" if inside else "Outside AD"
    }


# ============================================================
# 7. Model and metrics
# ============================================================

def train_morgan_rf(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=200,
        criterion="entropy",
        max_depth=None,
        min_samples_leaf=1,
        max_features="log2",
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    return model

def calculate_metrics(y_true, y_prob, threshold=PREDICTION_THRESHOLD):
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    se = tp / (tp + fn) if (tp + fn) > 0 else np.nan
    sp = tn / (tn + fp) if (tn + fp) > 0 else np.nan

    metrics = {
        "ROC-AUC": roc_auc_score(y_true, y_prob),
        "PR-AUC": average_precision_score(y_true, y_prob),
        "ACC": accuracy_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "SE": se,
        "SP": sp,
        "TN": int(tn),
        "FP": int(fp),
        "FN": int(fn),
        "TP": int(tp),
        "Threshold": threshold,
    }

    return metrics, y_pred


@st.cache_resource
def load_all_resources():
    train_raw = read_table(TRAIN_FILE)
    test_raw = read_table(TEST_FILE)

    train_df = clean_dataset(train_raw)
    test_df = clean_dataset(test_raw)

    X_train = build_morgan_matrix(train_df["SMILES"].tolist())
    y_train = train_df["Label"].values

    X_test = build_morgan_matrix(test_df["SMILES"].tolist())
    y_test = test_df["Label"].values

    train_fps = [
    smiles_to_morgan_fp_bitvect(
        smi,
        radius=MORGAN_RADIUS,
        nbits=AD_NBITS
    )
    for smi in train_df["SMILES"].tolist()
]
    train_fps = [fp for fp in train_fps if fp is not None]

    if FORCE_RETRAIN or (not MODEL_FILE.exists()):
        model = train_morgan_rf(X_train, y_train)
        joblib.dump(model, MODEL_FILE)
    else:
        model = joblib.load(MODEL_FILE)

    test_prob = model.predict_proba(X_test)[:, 1]
    metrics, test_pred = calculate_metrics(y_test, test_prob)

    metrics_df = pd.DataFrame([metrics])
    metrics_df.to_csv(METRICS_FILE, index=False, encoding="utf-8-sig")

    ad_cutoff, train_nn = compute_ad_cutoff(train_fps)

    ad_records = []
    for smi in test_df["SMILES"]:
        ad_records.append(assess_ad(smi, train_fps, ad_cutoff))

    ad_df = pd.DataFrame(ad_records)

    test_result = test_df.copy()
    test_result["Toxicity_Risk_Score"] = test_prob
    test_result["Predicted_Label"] = test_pred
    test_result["Predicted_Class"] = np.where(test_pred == 1, "Toxic", "Non-toxic")
    test_result["Nearest_Tanimoto"] = ad_df["Nearest_Tanimoto"]
    test_result["Inside_AD"] = ad_df["Inside_AD"]
    test_result["AD_Status"] = ad_df["AD_Status"]

    test_result.to_csv(TEST_PRED_FILE, index=False, encoding="utf-8-sig")

    ad_summary = pd.DataFrame({
        "AD_cutoff": [ad_cutoff],
        "Test_inside_AD_ratio": [test_result["Inside_AD"].mean()],
        "Train_molecules": [len(train_df)],
        "Test_molecules": [len(test_df)]
    })

    ad_summary.to_csv(AD_SUMMARY_FILE, index=False, encoding="utf-8-sig")

    return {
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "model": model,
        "train_fps": train_fps,
        "train_nn": train_nn,
        "ad_cutoff": ad_cutoff,
        "metrics": metrics,
        "test_result": test_result,
        "test_prob": test_prob,
    }


def predict_one_smiles(smiles, model, train_fps, ad_cutoff):
    can_smi = canonicalize_smiles(smiles)

    if can_smi is None:
        return None

    x = smiles_to_morgan_array(can_smi).reshape(1, -1)
    prob = float(model.predict_proba(x)[0, 1])
    pred = int(prob >= PREDICTION_THRESHOLD)

    ad_info = assess_ad(can_smi, train_fps, ad_cutoff)
    desc = compute_basic_descriptors(can_smi)

    result = {
        "SMILES": can_smi,
        "Toxicity_Risk_Score": prob,
        "Predicted_Label": pred,
        "Predicted_Class": "Toxic" if pred == 1 else "Non-toxic",
        "Threshold": PREDICTION_THRESHOLD,
        **ad_info
    }

    if desc is not None:
        result.update(desc)

    return result


# ============================================================
# 8. Plotting
# ============================================================

def plot_roc(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(fpr, tpr, linewidth=2.3, label=f"ROC-AUC = {auc_value:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC curve")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_pr(y_true, y_prob):
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    ap_value = average_precision_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(5.6, 4.5))
    ax.plot(recall, precision, linewidth=2.3, label=f"PR-AUC = {ap_value:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall curve")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_cm(metrics):
    cm = np.array([
        [metrics["TN"], metrics["FP"]],
        [metrics["FN"], metrics["TP"]]
    ])

    fig, ax = plt.subplots(figsize=(5.2, 4.5))
    ax.imshow(cm)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Pred 0\nNon-toxic", "Pred 1\nToxic"])
    ax.set_yticklabels(["True 0\nNon-toxic", "True 1\nToxic"])

    total = cm.sum()

    for i in range(2):
        for j in range(2):
            value = cm[i, j]
            percent = value / total * 100
            ax.text(
                j,
                i,
                f"{value}\n{percent:.1f}%",
                ha="center",
                va="center",
                fontsize=12
            )

    ax.set_title("Confusion matrix")
    fig.tight_layout()

    return fig


def plot_ad_distribution(train_nn, ad_cutoff):
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(train_nn, bins=40, alpha=0.85)
    ax.axvline(
        ad_cutoff,
        linestyle="--",
        linewidth=2,
        label=f"AD cutoff = {ad_cutoff:.4f}"
    )
    ax.set_xlabel("Training nearest-neighbor Tanimoto similarity")
    ax.set_ylabel("Count")
    ax.set_title("Morgan-Tanimoto AD cutoff")
    ax.legend()
    fig.tight_layout()

    return fig


def plot_risk_histogram(test_result):
    fig, ax = plt.subplots(figsize=(6, 4.5))

    toxic_scores = test_result.loc[test_result["Label"] == 1, "Toxicity_Risk_Score"]
    nontoxic_scores = test_result.loc[test_result["Label"] == 0, "Toxicity_Risk_Score"]

    ax.hist(nontoxic_scores, bins=35, alpha=0.65, label="Non-toxic")
    ax.hist(toxic_scores, bins=35, alpha=0.65, label="Toxic")
    ax.axvline(PREDICTION_THRESHOLD, linestyle="--", linewidth=2, label="Threshold = 0.50")

    ax.set_xlabel("Toxicity risk score")
    ax.set_ylabel("Count")
    ax.set_title("Risk score distribution")
    ax.legend()
    fig.tight_layout()

    return fig


def dataframe_to_csv_download(df):
    return df.to_csv(index=False, encoding="utf-8-sig").encode("utf-8-sig")


# ============================================================
# 9. Load resources
# ============================================================

try:
    resources = load_all_resources()
except Exception as e:
    st.error("Failed to load the VascTox resources.")
    st.exception(e)
    st.stop()

train_df = resources["train_df"]
test_df = resources["test_df"]
model = resources["model"]
train_fps = resources["train_fps"]
train_nn = resources["train_nn"]
ad_cutoff = resources["ad_cutoff"]
metrics = resources["metrics"]
test_result = resources["test_result"]
y_test = resources["y_test"]
test_prob = resources["test_prob"]


# ============================================================
# 10. Sidebar
# ============================================================

st.sidebar.title(APP_TITLE)
st.sidebar.caption("Vascular Toxicity Prediction and Applicability Domain Assessment")

page = st.sidebar.radio(
    "Select page",
    [
        "Overview",
        "Single Prediction",
        "Batch Prediction",
        "Applicability Domain",
        "High-Risk Molecules",
        "Model Details",
        "About"
    ]
)

st.sidebar.markdown("---")
st.sidebar.subheader("Model version")
st.sidebar.info(APP_VERSION)

st.sidebar.subheader("Current model")
st.sidebar.markdown(
    f"""
**Prediction model:** MorganFP-RF

**Input features:** Morgan fingerprints  
radius = {MORGAN_RADIUS}, nBits = {MORGAN_NBITS}

**AD fingerprint:** Morgan radius = {MORGAN_RADIUS}, nBits = {AD_NBITS}  
**AD method:** Morgan-Tanimoto nearest-neighbor similarity  
**AD cutoff:** {AD_CUTOFF:.4f}
"""
)

st.sidebar.subheader("Use note")
st.sidebar.caption(
    "For research use only. The predicted risk score is not clinical, "
    "regulatory, or experimental safety evidence. Predictions outside the "
    "applicability domain should be interpreted cautiously."
)


# ============================================================
# 11. Page: Overview
# ============================================================

if page == "Overview":
    st.markdown(
        """
<div class="hero-card">
    <h1>VascTox</h1>
    <h2>Vascular Toxicity Prediction Platform</h2>
    <p>
        VascTox is a machine-learning platform for small-molecule vascular toxicity
        risk prediction. The current version uses a MorganFP-RF model and integrates
        single-molecule prediction, batch SMILES screening, Morgan-Tanimoto applicability
        domain assessment, and high-risk compound prioritization.
    </p>
    <div class="tag-row">
        <span class="tag">MorganFP-RF</span>
        <span class="tag">Binary vascular toxicity</span>
        <span class="tag">Applicability domain</span>
        <span class="tag">Batch screening</span>
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

    section_title(
        "Dataset and Applicability Domain",
        "Core information for the fixed 1:1 vascular toxicity dataset and Morgan-Tanimoto AD definition."
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Training molecules", f"{len(train_df)}", "Model training set.", "blue")
    with c2:
        metric_card("Test molecules", f"{len(test_df)}", "Held-out evaluation set.", "blue")
    with c3:
        metric_card("AD cutoff", f"{ad_cutoff:.4f}", "Morgan-Tanimoto threshold.", "amber")
    with c4:
        metric_card("Test inside AD", f"{test_result['Inside_AD'].mean() * 100:.1f}%", "Coverage of test molecules.", "green")

    section_title(
        "Held-out Test Performance",
        "Four core metrics are shown on the homepage. Detailed metrics are provided in the Model Details page."
    )

    m1, m2, m3, m4 = st.columns(4)
    with m1:
       metric_card("ROC-AUC", f"{metrics['ROC-AUC']:.4f}", "Overall discrimination.", "green")
    with m2:
        metric_card("PR-AUC", f"{metrics['PR-AUC']:.4f}", "Toxic-class performance.", "green")
    with m3:
        metric_card("MCC", f"{metrics['MCC']:.4f}", "Balanced classification.", "green")
    with m4:
        metric_card("F1 score", f"{metrics['F1']:.4f}", "Precision-recall balance.", "blue")

    with st.expander("Show detailed metrics"):
        detail_df = pd.DataFrame([
            {
                "Metric": "Accuracy",
                "Value": metrics["ACC"],
                "Meaning": "Overall fraction of correct predictions"
            },
            {
                "Metric": "Sensitivity / Recall",
                "Value": metrics["SE"],
                "Meaning": "Fraction of toxic molecules correctly identified"
            },
            {
                "Metric": "Specificity",
                "Value": metrics["SP"],
                "Meaning": "Fraction of non-toxic molecules correctly identified"
            },
            {
                "Metric": "Precision",
                "Value": metrics["Precision"],
                "Meaning": "Fraction of predicted toxic molecules that are truly toxic"
            },
            {
                "Metric": "Decision threshold",
                "Value": metrics["Threshold"],
                "Meaning": "Risk score threshold used for binary classification"
            },
        ])
        st.dataframe(detail_df, use_container_width=True)

    section_title(
        "Workflow",
        "The prediction pipeline used by the deployed VascTox model."
    )

    st.markdown(
        """
<div class="workflow-box">
    <div class="workflow-line">
        SMILES input → Molecular standardization → Morgan fingerprint calculation →
        MorganFP-RF prediction → Morgan-Tanimoto AD assessment → Result interpretation
    </div>
</div>
        """,
        unsafe_allow_html=True
    )

   

    section_title("Performance Visualization")

    p1, p2, p3 = st.columns(3)
    with p1:
        st.pyplot(plot_roc(y_test, test_prob))
    with p2:
        st.pyplot(plot_pr(y_test, test_prob))
    with p3:
        st.pyplot(plot_cm(metrics))


# ============================================================
# 12. Page: Single Prediction
# ============================================================

elif page == "Single Prediction":
    st.title("Single-Molecule Prediction")

    info_panel(
        """
        Enter one SMILES string. VascTox will standardize the molecule, calculate Morgan
        fingerprints, predict vascular toxicity risk using MorganFP-RF, and assess whether
        the molecule is inside the Morgan-Tanimoto applicability domain.
        """,
        "info"
    )

    example = "CC(=O)OC1=CC=CC=C1C(=O)O"

    smiles_input = st.text_input(
        "Input SMILES",
        value=example,
        help="Example shown: aspirin"
    )

    run_button = st.button("Predict molecule")

    if run_button:
        result = predict_one_smiles(smiles_input, model, train_fps, ad_cutoff)

        if result is None:
            st.error("Invalid SMILES. Please check the input.")
        else:
            section_title("Prediction Result")

            p1, p2, p3, p4 = st.columns(4)

            pred_style = "red" if result["Predicted_Label"] == 1 else "green"
            ad_style = "green" if result["Inside_AD"] else "amber"

            with p1:
                metric_card("Predicted class", result["Predicted_Class"], "Binary prediction at threshold 0.50.", pred_style)
            with p2:
                metric_card("Risk score", f"{result['Toxicity_Risk_Score']:.4f}", "Model-derived toxicity risk score.", pred_style)
            with p3:
                metric_card("Nearest Tanimoto", f"{result['Nearest_Tanimoto']:.4f}", "Similarity to the nearest training molecule.", ad_style)
            with p4:
                metric_card("AD status", result["AD_Status"], "Inside or outside the Morgan-Tanimoto AD.", ad_style)

            if result["Predicted_Label"] == 1 and result["Inside_AD"]:
                info_panel(
                    """
                    The molecule is predicted as toxic and is inside the
                    applicability domain. This result may be prioritized for further
                    experimental or literature validation.
                    """,
                    "warning"
                )
            elif result["Predicted_Label"] == 1 and not result["Inside_AD"]:
                info_panel(
                    """
                    The molecule is predicted as toxic, but it is outside
                    the applicability domain. Interpret the risk score cautiously.
                    """,
                    "warning"
                )
            elif result["Predicted_Label"] == 0 and result["Inside_AD"]:
                info_panel(
                    """
                    The molecule is predicted as non-toxic and is inside
                    the applicability domain.
                    """,
                    "success"
                )
            else:
                info_panel(
                    """
                    The molecule is predicted as non-toxic, but it is outside
                    the applicability domain. Interpret the prediction cautiously.
                    """,
                    "info"
                )

            section_title("Molecular Structure")

            mol = Chem.MolFromSmiles(result["SMILES"])
            if mol is not None:
                img = Draw.MolToImage(mol, size=(480, 330))
                st.image(img)

            section_title("Basic Molecular Descriptors")

            desc_keys = [
                "Molecular weight",
                "LogP",
                "TPSA",
                "H-bond acceptors",
                "H-bond donors",
                "Rotatable bonds",
                "Ring count",
                "Heavy atoms",
                "Fraction Csp3"
            ]

            desc_df = pd.DataFrame(
                [{"Descriptor": k, "Value": result.get(k, np.nan)} for k in desc_keys]
            )

            st.dataframe(desc_df, use_container_width=True)

            section_title("Full Output")

            output_df = pd.DataFrame([result])
            st.dataframe(output_df, use_container_width=True)

            st.download_button(
                label="Download single prediction result",
                data=dataframe_to_csv_download(output_df),
                file_name="single_prediction_result.csv",
                mime="text/csv"
            )


# ============================================================
# 13. Page: Batch Prediction
# ============================================================

elif page == "Batch Prediction":
    st.title("Batch SMILES Prediction")

    info_panel(
        """
        Upload a CSV file containing a SMILES column, or paste multiple SMILES strings
        with one molecule per line. VascTox will return toxicity risk scores, predicted
        classes, nearest Tanimoto similarity, and applicability domain status.
        """,
        "info"
    )

    mode = st.radio("Input mode", ["Paste SMILES", "Upload CSV"], horizontal=True)

    input_df = None

    if mode == "Paste SMILES":
        text = st.text_area(
            "Paste SMILES, one per line",
            value="CC(=O)OC1=CC=CC=C1C(=O)O\nCCO\nc1ccccc1",
            height=160
        )

        if st.button("Run batch prediction"):
            smiles_list = [x.strip() for x in text.splitlines() if x.strip()]
            input_df = pd.DataFrame({"SMILES": smiles_list})

    else:
        uploaded = st.file_uploader("Upload CSV file", type=["csv"])

        if uploaded is not None:
            raw = pd.read_csv(uploaded)

            try:
                smiles_col, _ = auto_find_columns(raw, require_label=False)
                input_df = pd.DataFrame({"SMILES": raw[smiles_col].astype(str).tolist()})
            except Exception:
                st.error("Cannot identify a SMILES column in the uploaded CSV file.")

    if input_df is not None:
        results = []
        progress = st.progress(0)

        for i, smi in enumerate(input_df["SMILES"].tolist()):
            res = predict_one_smiles(smi, model, train_fps, ad_cutoff)

            if res is None:
                results.append({
                    "Input_SMILES": smi,
                    "SMILES": None,
                    "Toxicity_Risk_Score": np.nan,
                    "Predicted_Label": np.nan,
                    "Predicted_Class": "Invalid SMILES",
                    "Nearest_Tanimoto": np.nan,
                    "Inside_AD": False,
                    "AD_Status": "Invalid SMILES"
                })
            else:
                res["Input_SMILES"] = smi
                results.append(res)

            progress.progress((i + 1) / len(input_df))

        result_df = pd.DataFrame(results)

        valid_df = result_df[result_df["Predicted_Class"] != "Invalid SMILES"]

        section_title("Batch Prediction Summary")

        if len(valid_df) > 0:
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                metric_card("Total molecules", len(result_df), "Including invalid SMILES.", "blue")
            with b2:
                metric_card("Valid molecules", len(valid_df), "Successfully parsed by RDKit.", "blue")
            with b3:
                metric_card("Predicted toxic", int((valid_df["Predicted_Label"] == 1).sum()), "Risk score ≥ 0.50.", "amber")
            with b4:
                metric_card("Inside AD", f"{valid_df['Inside_AD'].mean() * 100:.1f}%", "Molecules inside Morgan-Tanimoto AD.", "green")

        section_title("Batch Prediction Results")
        st.dataframe(result_df, use_container_width=True)

        st.download_button(
            label="Download batch prediction results",
            data=dataframe_to_csv_download(result_df),
            file_name="batch_prediction_results.csv",
            mime="text/csv"
        )


# ============================================================
# 14. Page: Applicability Domain
# ============================================================

elif page == "Applicability Domain":
    st.title("Applicability Domain Analysis")

    info_panel(
        f"""
        The applicability domain is defined using 2048-bit Morgan fingerprint-based
        Tanimoto nearest-neighbor similarity. This deployed version uses the final
        study cutoff of {ad_cutoff:.4f}, consistent with the manuscript analysis.
        """,
        "info"
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("AD cutoff", f"{ad_cutoff:.4f}", "Morgan-Tanimoto similarity cutoff.", "amber")
    with c2:
        metric_card("Test inside AD", f"{test_result['Inside_AD'].mean() * 100:.1f}%", "Coverage of the held-out test set.", "green")
    with c3:
        metric_card("Inside AD molecules", int(test_result["Inside_AD"].sum()), "Test molecules inside AD.", "green")
    with c4:
        metric_card("Outside AD molecules", int((~test_result["Inside_AD"]).sum()), "Test molecules outside AD.", "amber")

    section_title("AD Cutoff Distribution")

    left, right = st.columns([1.15, 1.0])

    with left:
        st.pyplot(plot_ad_distribution(train_nn, ad_cutoff))

    with right:
        st.pyplot(plot_risk_histogram(test_result))

    section_title("Performance by AD Subset")

    inside = test_result[test_result["Inside_AD"] == True]
    outside = test_result[test_result["Inside_AD"] == False]

    perf_rows = []

    for name, sub in [
        ("All test set", test_result),
        ("Inside AD", inside),
        ("Outside AD", outside),
    ]:
        if len(sub) > 2 and sub["Label"].nunique() == 2:
            sub_metrics, _ = calculate_metrics(
                sub["Label"].values,
                sub["Toxicity_Risk_Score"].values
            )

            perf_rows.append({
                "Subset": name,
                "N": len(sub),
                "ROC-AUC": sub_metrics["ROC-AUC"],
                "PR-AUC": sub_metrics["PR-AUC"],
                "ACC": sub_metrics["ACC"],
                "MCC": sub_metrics["MCC"],
                "F1": sub_metrics["F1"],
                "Sensitivity": sub_metrics["SE"],
                "Specificity": sub_metrics["SP"],
            })
        else:
            perf_rows.append({
                "Subset": name,
                "N": len(sub),
                "ROC-AUC": np.nan,
                "PR-AUC": np.nan,
                "ACC": np.nan,
                "MCC": np.nan,
                "F1": np.nan,
                "Sensitivity": np.nan,
                "Specificity": np.nan,
            })

    perf_df = pd.DataFrame(perf_rows)
    st.dataframe(perf_df, use_container_width=True)

    st.download_button(
        label="Download AD performance table",
        data=dataframe_to_csv_download(perf_df),
        file_name="ad_subset_performance.csv",
        mime="text/csv"
    )

    section_title("Test-set AD Results")

    show_cols = [
        "SMILES",
        "Label",
        "Toxicity_Risk_Score",
        "Predicted_Class",
        "Nearest_Tanimoto",
        "Inside_AD",
        "AD_Status",
    ]

    st.dataframe(test_result[show_cols], use_container_width=True)

    st.download_button(
        label="Download test-set AD predictions",
        data=dataframe_to_csv_download(test_result),
        file_name="test_set_ad_predictions.csv",
        mime="text/csv"
    )


# ============================================================
# 15. Page: High-Risk Molecules
# ============================================================

elif page == "High-Risk Molecules":
    st.title("Representative High-Risk Molecules")

    info_panel(
        """
        Molecules are ranked by MorganFP-RF toxicity risk score. By default, this page
        prioritizes molecules predicted as toxic and located inside the applicability
        domain.
        """,
        "info"
    )

    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        only_inside = st.checkbox("Inside AD only", value=True)

    with c2:
        only_pred_toxic = st.checkbox("Predicted toxic only", value=True)

    with c3:
        top_n = st.slider("Number of molecules to show", min_value=5, max_value=100, value=20, step=5)

    df = test_result.copy()

    if only_inside:
        df = df[df["Inside_AD"] == True]

    if only_pred_toxic:
        df = df[df["Predicted_Label"] == 1]

    df = df.sort_values("Toxicity_Risk_Score", ascending=False).head(top_n)

    section_title("High-Risk Molecule Table")

    show_cols = [
        "SMILES",
        "Label",
        "Toxicity_Risk_Score",
        "Predicted_Class",
        "Nearest_Tanimoto",
        "AD_Status"
    ]

    st.dataframe(df[show_cols], use_container_width=True)

    st.download_button(
        label="Download high-risk molecule table",
        data=dataframe_to_csv_download(df),
        file_name="high_risk_molecules.csv",
        mime="text/csv"
    )

    section_title("Molecular Structures")

    if len(df) == 0:
        st.info("No molecules match the selected filters.")
    else:
        cols = st.columns(4)

        for i, (_, row) in enumerate(df.head(12).iterrows()):
            mol = Chem.MolFromSmiles(row["SMILES"])

            if mol is None:
                continue

            img = Draw.MolToImage(mol, size=(260, 200))

            with cols[i % 4]:
                st.image(img)
                st.caption(
                    f"Risk score: {row['Toxicity_Risk_Score']:.3f} | "
                    f"AD: {row['AD_Status']}"
                )


# ============================================================
# 16. Page: Model Details
# ============================================================

elif page == "Model Details":
    st.title("Model Details")

    section_title("Model Configuration")

    config_df = pd.DataFrame([
        {"Item": "Prediction model", "Value": "MorganFP-RF"},
        {"Item": "Molecular representation", "Value": f"Morgan fingerprint, radius={MORGAN_RADIUS}, nBits={MORGAN_NBITS}"},
        {"Item": "Classifier", "Value": "Random Forest"},
        {"Item": "Number of trees", "Value": "200"},
        {"Item": "Class weighting", "Value": "balanced"},
        {"Item": "Decision threshold", "Value": f"{PREDICTION_THRESHOLD:.2f}"},
        {"Item": "AD method", "Value": "Morgan-Tanimoto nearest-neighbor similarity"},
        {"Item": "AD cutoff", "Value": f"{ad_cutoff:.4f}"},
    ])

    st.dataframe(config_df, use_container_width=True)

    section_title("Complete Test Metrics")

    metrics_df = pd.DataFrame([
        {"Metric": "ROC-AUC", "Value": metrics["ROC-AUC"]},
        {"Metric": "PR-AUC", "Value": metrics["PR-AUC"]},
        {"Metric": "Accuracy", "Value": metrics["ACC"]},
        {"Metric": "MCC", "Value": metrics["MCC"]},
        {"Metric": "F1", "Value": metrics["F1"]},
        {"Metric": "Precision", "Value": metrics["Precision"]},
        {"Metric": "Sensitivity / Recall", "Value": metrics["SE"]},
        {"Metric": "Specificity", "Value": metrics["SP"]},
        {"Metric": "Threshold", "Value": metrics["Threshold"]},
        {"Metric": "TN", "Value": metrics["TN"]},
        {"Metric": "FP", "Value": metrics["FP"]},
        {"Metric": "FN", "Value": metrics["FN"]},
        {"Metric": "TP", "Value": metrics["TP"]},
    ])

    st.dataframe(metrics_df, use_container_width=True)

    st.download_button(
        label="Download complete model metrics",
        data=dataframe_to_csv_download(metrics_df),
        file_name="morganfp_rf_model_metrics.csv",
        mime="text/csv"
    )

    section_title("Performance Figures")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.pyplot(plot_roc(y_test, test_prob))
    with f2:
        st.pyplot(plot_pr(y_test, test_prob))
    with f3:
        st.pyplot(plot_cm(metrics))

    

# ============================================================
# 17. Page: About
# ============================================================

elif page == "About":
    st.title("About VascTox")

    info_panel(
        """
        VascTox is intended for research-stage small-molecule vascular toxicity risk
        prediction and compound prioritization. It is not a substitute for experimental,
        clinical, or regulatory safety evaluation.
        """,
        "info"
    )

    section_title("Current Version")

    st.markdown(
        f"""
**{APP_VERSION}** uses a **MorganFP-RF** classifier.

- Molecular representation: Morgan fingerprint
- Radius: {MORGAN_RADIUS}
- Fingerprint length: {MORGAN_NBITS}
- Classifier: Random Forest
- Decision threshold: {PREDICTION_THRESHOLD:.2f}
- Applicability domain: Morgan-Tanimoto nearest-neighbor similarity
- Training molecules: {len(train_df)}
- Test molecules: {len(test_df)}
"""
    )

    section_title("Output Interpretation")

    st.markdown(
        """
The platform reports a **toxicity risk score**. This value should be interpreted as
a model-derived risk score rather than direct experimental toxicity evidence.

Predictions should be interpreted together with the applicability domain status:

- **Inside AD**: the molecule is sufficiently similar to at least one training molecule
  according to Morgan-Tanimoto similarity.
- **Outside AD**: the molecule is less well covered by the training chemical space;
  prediction reliability may be lower.
"""
    )

    section_title("Limitations")

    info_panel(
        """
        The model is optimized for research screening and prioritization. It should not
        be used as clinical, regulatory, or experimental safety evidence. Predictions
        outside the applicability domain should be interpreted cautiously.
        """,
        "warning"
    )