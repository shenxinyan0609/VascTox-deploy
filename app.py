# -*- coding: utf-8 -*-

import io
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from rdkit import Chem, DataStructs
from rdkit.Chem import Draw, Descriptors, Crippen, rdMolDescriptors
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")


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


# =========================================================
# 1. Path settings
# =========================================================
BASE_DIR = Path(__file__).resolve().parent

TRAIN_PATH = BASE_DIR / "train.xlsx"
TEST_PATH = BASE_DIR / "test.xlsx"

PROJECT_DIR = BASE_DIR
MODEL_PATH = PROJECT_DIR / "models" / "best_model.pkl"
MODEL_INFO_PATH = PROJECT_DIR / "models" / "best_model_info.json"

AD_RESULT_PATH = PROJECT_DIR / "data" / "AD_train_test_result_checked.xlsx"
TOP20_PATH = PROJECT_DIR / "data" / "top20_toxic.csv"


# =========================================================
# 2. Page configuration
# =========================================================
st.set_page_config(
    page_title="VascTox",
    page_icon="🧬",
    layout="wide"
)


# =========================================================
# 3. Basic functions
# =========================================================
def standardize_smiles(smiles):
    if pd.isna(smiles):
        return None, None, None, "Empty SMILES"

    smi = str(smiles).strip()

    if smi == "":
        return None, None, None, "Empty SMILES"

    mol = Chem.MolFromSmiles(smi)

    if mol is None:
        return None, None, None, "Invalid SMILES"

    canonical_smiles = Chem.MolToSmiles(
        mol,
        canonical=True,
        isomericSmiles=True
    )

    fp = mol_to_fp(mol)

    return canonical_smiles, mol, fp, "Valid"


def calc_basic_descriptors(mol):
    return {
        "MolWt": round(Descriptors.MolWt(mol), 3),
        "LogP": round(Crippen.MolLogP(mol), 3),
        "TPSA": round(rdMolDescriptors.CalcTPSA(mol), 3),
        "NumHDonors": rdMolDescriptors.CalcNumHBD(mol),
        "NumHAcceptors": rdMolDescriptors.CalcNumHBA(mol),
        "NumRotatableBonds": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "RingCount": rdMolDescriptors.CalcNumRings(mol),
        "HeavyAtomCount": mol.GetNumHeavyAtoms()
    }


def get_risk_level(prob):
    if prob is None:
        return "Not available"

    if prob >= 0.80:
        return "High risk"
    elif prob >= 0.50:
        return "Moderate risk"
    else:
        return "Low risk"


@st.cache_data
def load_ad_summary():
    if not AD_RESULT_PATH.exists():
        return None

    try:
        summary_df = pd.read_excel(AD_RESULT_PATH, sheet_name="AD_summary")
        return summary_df
    except Exception:
        return None


def get_ad_threshold():
    summary_df = load_ad_summary()

    if summary_df is not None and "AD_threshold" in summary_df.columns:
        try:
            return float(summary_df["AD_threshold"].iloc[0])
        except Exception:
            pass

    return 0.375000


def get_metric_from_summary(column_name, default_value="NA"):
    summary_df = load_ad_summary()

    if summary_df is not None and column_name in summary_df.columns:
        try:
            value = summary_df[column_name].iloc[0]
            return value
        except Exception:
            return default_value

    return default_value


@st.cache_resource
def load_train_reference():
    if not TRAIN_PATH.exists():
        return None, None, None, "Training file not found"

    train_df = pd.read_excel(TRAIN_PATH)

    if "SMILES" not in train_df.columns:
        return None, None, None, "The training set does not contain a SMILES column"

    if "Label" not in train_df.columns:
        train_df["Label"] = np.nan

    canonical_list = []
    label_list = []
    fp_list = []

    for _, row in train_df.iterrows():
        canonical_smiles, mol, fp, status = standardize_smiles(row["SMILES"])

        if status == "Valid":
            canonical_list.append(canonical_smiles)
            label_list.append(row["Label"])
            fp_list.append(fp)

    if len(fp_list) == 0:
        return None, None, None, "No valid SMILES found in the training set"

    return canonical_list, label_list, fp_list, "OK"


@st.cache_resource
def load_prediction_model():
    if MODEL_PATH.exists():
        try:
            model_package = joblib.load(MODEL_PATH)
            return model_package, "OK"
        except Exception as e:
            return None, f"Failed to load model: {e}"

    return None, "Model file not found"


def calculate_ad(fp):
    train_smiles, train_labels, train_fps, status = load_train_reference()
    ad_threshold = get_ad_threshold()

    if status != "OK":
        return {
            "AD_max_Tanimoto": None,
            "AD_status": "Not available",
            "nearest_train_SMILES": None,
            "nearest_train_Label": None,
            "AD_message": status
        }

    sims = list(DataStructs.BulkTanimotoSimilarity(fp, train_fps))
    nearest_idx = int(np.argmax(sims))
    max_sim = float(sims[nearest_idx])

    ad_status = "Inside AD" if max_sim >= ad_threshold else "Outside AD"

    return {
        "AD_max_Tanimoto": max_sim,
        "AD_status": ad_status,
        "nearest_train_SMILES": train_smiles[nearest_idx],
        "nearest_train_Label": train_labels[nearest_idx],
        "AD_message": "OK"
    }


def calc_descriptors_for_model(mol, model_package):
    desc_names = model_package["desc_names"]
    all_nan_cols = model_package["all_nan_cols"]
    medians = model_package["medians"]
    finite_cols = model_package["finite_cols"]
    huge_cols = model_package["huge_cols"]
    keep_cols = model_package["keep_cols"]

    desc_func_dict = dict(Descriptors._descList)

    values = []

    for name in desc_names:
        func = desc_func_dict.get(name)

        if func is None:
            values.append(np.nan)
            continue

        try:
            v = func(mol)
            if v is None or isinstance(v, str):
                v = np.nan
            else:
                v = float(v)
        except Exception:
            v = np.nan

        values.append(v)

    desc_df = pd.DataFrame([values], columns=desc_names, dtype=np.float64)

    desc_df = desc_df.replace([np.inf, -np.inf], np.nan)
    desc_df = desc_df.drop(columns=all_nan_cols, errors="ignore")
    desc_df = desc_df.fillna(medians)
    desc_df = desc_df.reindex(columns=finite_cols)
    desc_df = desc_df.drop(columns=huge_cols, errors="ignore")
    desc_df = desc_df.reindex(columns=keep_cols)

    try:
        desc_df = desc_df.fillna(pd.Series(medians).reindex(keep_cols))
    except Exception:
        pass

    desc_df = desc_df.fillna(0)

    X = desc_df.values.astype(np.float64)

    return X


def predict_toxicity(mol):
    model_package, status = load_prediction_model()

    if status != "OK":
        return {
            "Pred_probability": None,
            "Pred_label": "Model not loaded",
            "Risk_level": "Not available",
            "Pred_message": status
        }

    try:
        if not isinstance(model_package, dict):
            return {
                "Pred_probability": None,
                "Pred_label": "Unsupported model",
                "Risk_level": "Not available",
                "Pred_message": "best_model.pkl is not a Descriptors-RF model package."
            }

        if model_package.get("model_type") != "Descriptors-RF":
            return {
                "Pred_probability": None,
                "Pred_label": "Unsupported model",
                "Risk_level": "Not available",
                "Pred_message": f"Current model type is {model_package.get('model_type')}, not Descriptors-RF."
            }

        model = model_package["model"]
        X = calc_descriptors_for_model(mol, model_package)

        prob = float(model.predict_proba(X)[0][1])

        threshold = model_package.get("classification_threshold", 0.5)

        pred_label = "Vascular toxicant" if prob >= threshold else "Non-toxic"
        risk_level = get_risk_level(prob)

        return {
            "Pred_probability": prob,
            "Pred_label": pred_label,
            "Risk_level": risk_level,
            "Pred_message": "OK"
        }

    except Exception as e:
        return {
            "Pred_probability": None,
            "Pred_label": "Prediction failed",
            "Risk_level": "Not available",
            "Pred_message": str(e)
        }


def analyze_one_smiles(smiles):
    canonical_smiles, mol, fp, status = standardize_smiles(smiles)

    result = {
        "Input_SMILES": smiles,
        "Canonical_SMILES": canonical_smiles,
        "SMILES_status": status
    }

    if status != "Valid":
        result.update({
            "Pred_probability": None,
            "Pred_label": "Invalid",
            "Risk_level": "Not available",
            "Pred_message": "Invalid SMILES",
            "AD_max_Tanimoto": None,
            "AD_status": "Not available",
            "nearest_train_SMILES": None,
            "nearest_train_Label": None,
            "AD_message": "Invalid SMILES"
        })
        return result, mol

    pred_result = predict_toxicity(mol)
    ad_result = calculate_ad(fp)

    result.update(pred_result)
    result.update(ad_result)

    return result, mol


def display_molecule(mol, title="Molecular structure"):
    if mol is None:
        st.warning("Unable to display molecular structure.")
        return

    img = Draw.MolToImage(mol, size=(420, 300))
    st.image(img, caption=title)


# =========================================================
# 4. Sidebar
# =========================================================
st.sidebar.title("VascTox")
st.sidebar.caption("Vascular Toxicity Prediction and Applicability Domain Assessment")

page = st.sidebar.radio(
    "Select page",
    [
        "Overview",
        "Single Prediction",
        "Batch Prediction",
        "Applicability Domain Analysis",
        "High-Risk Molecules",
        "About"
    ]
)

st.sidebar.divider()
st.sidebar.write("Current project files:")
st.sidebar.code(f"Training set: {TRAIN_PATH}")
st.sidebar.code(f"AD result: {AD_RESULT_PATH}")
st.sidebar.code(f"Model: {MODEL_PATH}")


# =========================================================
# 5. Overview
# =========================================================
if page == "Overview":

    st.title("VascTox")
    st.subheader("Vascular Toxicity Prediction Platform")

    st.markdown(
        """
        This platform supports small-molecule vascular toxicity prediction, batch SMILES analysis,
        applicability domain assessment, and visualization of representative high-risk molecules.
        """
    )

    ad_threshold = get_ad_threshold()

    train_count = get_metric_from_summary("train_valid_molecules", 3616)
    test_count = get_metric_from_summary("test_valid_molecules", 904)
    inside_ratio = get_metric_from_summary("inside_AD_ratio", None)
    test_auc = get_metric_from_summary("test_AUC", None)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Training molecules", f"{int(train_count)}")
    col2.metric("Test molecules", f"{int(test_count)}")
    col3.metric("AD cutoff", f"{ad_threshold:.4f}")

    if inside_ratio is not None and not pd.isna(inside_ratio):
        col4.metric("Inside AD ratio", f"{float(inside_ratio) * 100:.1f}%")
    else:
        col4.metric("Inside AD ratio", "NA")

    st.divider()

    st.markdown("### Model Summary")

    c1, c2, c3 = st.columns(3)

    if test_auc is not None and not pd.isna(test_auc):
        c1.metric("Test AUC", f"{float(test_auc):.4f}")
    else:
        c1.metric("Test AUC", "NA")

    test_acc = get_metric_from_summary("test_ACC", None)
    test_mcc = get_metric_from_summary("test_MCC", None)

    if test_acc is not None and not pd.isna(test_acc):
        c2.metric("Test ACC", f"{float(test_acc):.4f}")
    else:
        c2.metric("Test ACC", "NA")

    if test_mcc is not None and not pd.isna(test_mcc):
        c3.metric("Test MCC", f"{float(test_mcc):.4f}")
    else:
        c3.metric("Test MCC", "NA")

    st.markdown("### Workflow")

    st.markdown(
        """
        **SMILES input → Molecular standardization → RDKit descriptor calculation → Descriptors-RF prediction → Morgan-Tanimoto AD assessment → Result interpretation**
        """
    )

    st.info(
        "The current version uses a Descriptors-RF classifier trained on the fixed 1:1 vascular toxicity dataset under the single 10 µM criterion."
    )


# =========================================================
# 6. Single Prediction
# =========================================================
elif page == "Single Prediction":

    st.title("Single-Molecule Vascular Toxicity Prediction")

    example_smiles = "CC(=O)Oc1ccccc1C(=O)O"

    smiles = st.text_area(
        "Enter a SMILES:",
        value=example_smiles,
        height=100
    )

    run_button = st.button("Run Analysis", type="primary")

    if run_button:

        result, mol = analyze_one_smiles(smiles)

        st.divider()

        left, right = st.columns([1.1, 1.2])

        with left:
            display_molecule(mol)

            st.markdown("### Basic Molecular Properties")

            if mol is not None:
                desc = calc_basic_descriptors(mol)
                st.dataframe(
                    pd.DataFrame([desc]).T.rename(columns={0: "Value"}),
                    use_container_width=True
                )

        with right:
            st.markdown("### Prediction and AD Results")

            st.write(f"**Canonical SMILES:** `{result['Canonical_SMILES']}`")
            st.write(f"**SMILES status:** {result['SMILES_status']}")

            pred_prob = result.get("Pred_probability")

            if pred_prob is not None:
                st.metric("Predicted Vascular Toxicity Probability", f"{pred_prob:.4f}")
                st.write(f"**Predicted class:** {result['Pred_label']}")
                st.write(f"**Risk level:** {result['Risk_level']}")

                if result["Pred_label"] == "Vascular toxicant":
                    st.error("The model predicts that this molecule has vascular toxicity risk.")
                else:
                    st.success("The model predicts that this molecule is non-toxic under the current criterion.")

            else:
                st.warning(
                    f"Model prediction failed: {result.get('Pred_message', 'Unknown error')}"
                )

            ad_sim = result.get("AD_max_Tanimoto")

            if ad_sim is not None:
                st.metric("AD Max Tanimoto Similarity", f"{ad_sim:.4f}")
                st.write(f"**AD status:** {result['AD_status']}")
                st.write(f"**Nearest training-set molecule:** `{result['nearest_train_SMILES']}`")
                st.write(f"**Nearest training-set molecule label:** {result['nearest_train_Label']}")

                if result["AD_status"] == "Inside AD":
                    st.success(
                        "This molecule is inside the model applicability domain, suggesting relatively higher prediction reliability."
                    )
                else:
                    st.error(
                        "This molecule is outside the model applicability domain; the prediction should be interpreted with caution."
                    )

            else:
                st.warning(
                    f"AD analysis failed: {result.get('AD_message', 'Unknown error')}"
                )


# =========================================================
# 7. Batch Prediction
# =========================================================
elif page == "Batch Prediction":

    st.title("Batch SMILES Prediction")

    st.markdown(
        """
        Upload a CSV or Excel file. The file must contain at least one column named `SMILES`.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload file",
        type=["csv", "xlsx"]
    )

    if uploaded_file is not None:

        try:
            if uploaded_file.name.lower().endswith(".csv"):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            st.write("Uploaded file preview:")
            st.dataframe(df.head(), use_container_width=True)

            if "SMILES" not in df.columns:
                st.error("The uploaded file does not contain a SMILES column.")

            else:
                if st.button("Run Batch Analysis", type="primary"):

                    results = []

                    for _, row in df.iterrows():
                        result, mol = analyze_one_smiles(row["SMILES"])

                        for col in df.columns:
                            if col not in result and col != "SMILES":
                                result[col] = row[col]

                        results.append(result)

                    result_df = pd.DataFrame(results)

                    st.success("Batch analysis completed.")
                    st.dataframe(result_df, use_container_width=True)

                    csv_bytes = result_df.to_csv(index=False).encode("utf-8-sig")

                    st.download_button(
                        label="Download CSV Results",
                        data=csv_bytes,
                        file_name="VascTox_batch_prediction.csv",
                        mime="text/csv"
                    )

                    output = io.BytesIO()

                    with pd.ExcelWriter(output, engine="openpyxl") as writer:
                        result_df.to_excel(
                            writer,
                            index=False,
                            sheet_name="prediction_result"
                        )

                    st.download_button(
                        label="Download Excel Results",
                        data=output.getvalue(),
                        file_name="VascTox_batch_prediction.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                    )

        except Exception as e:
            st.error(f"File reading or analysis failed: {e}")


# =========================================================
# 8. Applicability Domain Analysis
# =========================================================
elif page == "Applicability Domain Analysis":

    st.title("Applicability Domain Analysis")

    st.markdown(
        """
        This module evaluates whether test-set molecules fall within the training-set chemical space based on Morgan fingerprints and Tanimoto similarity.
        """
    )

    ad_threshold = get_ad_threshold()

    test_count = get_metric_from_summary("test_valid_molecules", 904)
    inside_count = get_metric_from_summary("inside_AD_count", "NA")
    outside_count = get_metric_from_summary("outside_AD_count", "NA")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("AD cutoff", f"{ad_threshold:.4f}")
    col2.metric("Valid test molecules", f"{int(test_count)}")
    col3.metric("Inside AD", f"{inside_count}")
    col4.metric("Outside AD", f"{outside_count}")

    st.divider()

    if not AD_RESULT_PATH.exists():
        st.error(f"AD result file not found: {AD_RESULT_PATH}")

    else:
        try:
            summary_df = pd.read_excel(AD_RESULT_PATH, sheet_name="AD_summary")
            test_ad_df = pd.read_excel(AD_RESULT_PATH, sheet_name="test_AD_result")
            train_nn_df = pd.read_excel(AD_RESULT_PATH, sheet_name="train_NN_similarity")

            st.markdown("### AD Summary")
            st.dataframe(summary_df, use_container_width=True)

            st.markdown("### Training-Set Nearest-Neighbor Tanimoto Similarity Distribution")

            if "train_nearest_neighbor_similarity" in train_nn_df.columns:
                fig1 = px.histogram(
                    train_nn_df,
                    x="train_nearest_neighbor_similarity",
                    nbins=40,
                    title="Training-set nearest-neighbor Tanimoto similarity"
                )
                fig1.add_vline(
                    x=ad_threshold,
                    line_dash="dash",
                    annotation_text=f"AD cutoff = {ad_threshold:.4f}",
                    annotation_position="top right"
                )
                st.plotly_chart(fig1, use_container_width=True)

            st.markdown("### Test-Set Inside/Outside AD Distribution")

            if "AD_status" in test_ad_df.columns:
                fig2 = px.histogram(
                    test_ad_df,
                    x="AD_status",
                    title="Test-set AD distribution"
                )
                st.plotly_chart(fig2, use_container_width=True)

            st.markdown("### Ranked AD_max_Tanimoto Values of Test Compounds")

            if "AD_max_Tanimoto" in test_ad_df.columns:
                temp_df = test_ad_df.copy()
                temp_df = temp_df.sort_values("AD_max_Tanimoto").reset_index(drop=True)
                temp_df["Index"] = np.arange(1, len(temp_df) + 1)

                fig3 = px.scatter(
                    temp_df,
                    x="Index",
                    y="AD_max_Tanimoto",
                    color="AD_status",
                    title="AD_max_Tanimoto values of test compounds"
                )
                fig3.add_hline(
                    y=ad_threshold,
                    line_dash="dash",
                    annotation_text=f"AD cutoff = {ad_threshold:.4f}",
                    annotation_position="top left"
                )
                st.plotly_chart(fig3, use_container_width=True)

        except Exception as e:
            st.error(f"Failed to read AD results: {e}")


# =========================================================
# 9. High-Risk Molecules
# =========================================================
elif page == "High-Risk Molecules":

    st.title("High-Risk Molecules")

    st.markdown(
        """
        This page lists representative high-risk vascular toxicants predicted by the model
        and provides molecular structure visualization.
        """
    )

    st.divider()

    st.markdown("### Top 20 Predicted High-Risk Molecules")

    if TOP20_PATH.exists():
        try:
            top20_df = pd.read_csv(TOP20_PATH)

            st.dataframe(top20_df, use_container_width=True)

            if "SMILES" not in top20_df.columns:
                st.error("The file top20_toxic.csv does not contain a SMILES column, so molecular structures cannot be rendered.")

            else:
                st.markdown("### Molecular Structure Preview")

                for i, row in top20_df.head(20).iterrows():

                    canonical_smiles, mol, fp, status = standardize_smiles(row["SMILES"])

                    if status != "Valid":
                        with st.expander(f"{i + 1}. Molecule_{i + 1} | Invalid SMILES"):
                            st.warning(f"Invalid SMILES: {row['SMILES']}")
                        continue

                    name = f"Molecule_{i + 1}"

                    pred_prob = None

                    if "Pred_Prob" in top20_df.columns:
                        pred_prob = row.get("Pred_Prob")
                    elif "Pred_probability" in top20_df.columns:
                        pred_prob = row.get("Pred_probability")
                    elif "pred_prob" in top20_df.columns:
                        pred_prob = row.get("pred_prob")
                    elif "probability" in top20_df.columns:
                        pred_prob = row.get("probability")

                    if pred_prob is None or pd.isna(pred_prob):
                        pred_prob_text = "NA"
                    else:
                        try:
                            pred_prob_text = f"{float(pred_prob):.4f}"
                        except Exception:
                            pred_prob_text = str(pred_prob)

                    expander_title = f"{i + 1}. {name} | Pred_Prob = {pred_prob_text}"

                    with st.expander(expander_title):

                        display_molecule(mol)

                        st.write(f"**Canonical SMILES:** `{canonical_smiles}`")
                        st.write(f"**Input SMILES:** `{row['SMILES']}`")

                        if "Label" in top20_df.columns:
                            st.write(f"**Original label:** {row.get('Label')}")
                        elif "label" in top20_df.columns:
                            st.write(f"**Original label:** {row.get('label')}")

                        if "pred_label" in top20_df.columns:
                            st.write(f"**Predicted label:** {row.get('pred_label')}")
                        elif "Pred_Label" in top20_df.columns:
                            st.write(f"**Predicted label:** {row.get('Pred_Label')}")

                        if pred_prob_text != "NA":
                            st.write(f"**Predicted probability:** {pred_prob_text}")

                        if "cc50_uM" in top20_df.columns:
                            st.write(f"**CC50 / µM:** {row.get('cc50_uM')}")

                        if "AD_status" in top20_df.columns:
                            st.write(f"**AD status:** {row.get('AD_status')}")

                        if "AD_max_Tanimoto" in top20_df.columns:
                            try:
                                st.write(f"**AD max Tanimoto:** {float(row.get('AD_max_Tanimoto')):.4f}")
                            except Exception:
                                st.write(f"**AD max Tanimoto:** {row.get('AD_max_Tanimoto')}")

                        if "nearest_train_SMILES" in top20_df.columns:
                            st.write(f"**Nearest training-set molecule:** `{row.get('nearest_train_SMILES')}`")

                        if "nearest_train_Label" in top20_df.columns:
                            st.write(f"**Nearest training-set label:** {row.get('nearest_train_Label')}")

        except Exception as e:
            st.error(f"Failed to read Top 20 file: {e}")

    else:
        st.info(
            f"Top 20 file was not detected. Please place top20_toxic.csv at: {TOP20_PATH}"
        )


# =========================================================
# 10. About
# =========================================================
elif page == "About":

    st.title("About VascTox")

    st.markdown(
        """
        **VascTox** is a prototype web platform for vascular toxicity prediction.

        The current version includes:

        1. Single-molecule SMILES prediction;
        2. Batch SMILES prediction;
        3. Descriptors-RF vascular toxicity classification;
        4. Morgan-Tanimoto applicability domain assessment;
        5. Molecular structure visualization;
        6. Representative high-risk molecule display.
        """
    )

    st.markdown("### Current Model Settings")

    st.code(
        f"""
Prediction model:
    Descriptors-RF

Input features:
    RDKit molecular descriptors

Model file:
    {MODEL_PATH}

Classification threshold:
    0.5

Positive class:
    Label = 1, vascular toxicant under the single 10 µM criterion
        """
    )

    st.markdown("### Current AD Settings")

    ad_threshold = get_ad_threshold()

    st.code(
        f"""
Morgan fingerprint:
    radius = 2
    nBits = 2048

AD threshold:
    {ad_threshold:.6f}

Inside AD:
    AD_max_Tanimoto >= {ad_threshold:.6f}

Outside AD:
    AD_max_Tanimoto < {ad_threshold:.6f}
        """
    )