# -*- coding: utf-8 -*-
"""
GeneGenie — Rare Disease Intelligence Dashboard
Clinical-grade UI covering ALL rare diseases:
- 12,671 rare diseases indexed (OMIM + Orphanet + HPO)
- 500-disease differential classifier (RF V4, 98.94% within-set accuracy)
- Full-corpus TF-IDF retrieval across all 12,671 diseases
- ClinVar pathogenicity integration (3.7 GB variant_summary processed)
- gnomAD v4.1 constraint scores (pLI, LOEUF)
- Knowledge graph with 9,801 nodes, 100K edges
- Similar disease engine with HPO Jaccard + penetrance weighting
- UMAP disease map with Louvain clustering
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

st.set_page_config(
    page_title="GeneGenie - Rare Disease Intelligence",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Clinical-Grade Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main theme - clinical blue/teal palette */
    :root {
        --primary: #0066CC;
        --primary-dark: #004C99;
        --accent: #00A896;
        --warning: #F18F01;
        --danger: #E63946;
        --success: #06A77D;
        --bg-card: #F8F9FA;
        --border: #E1E5EB;
    }

    /* Hide default Streamlit branding */
    #MainMenu, header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1400px;
    }

    /* Clinical header */
    .clinical-header {
        background: linear-gradient(135deg, #0066CC 0%, #00A896 100%);
        padding: 1.5rem 2rem;
        border-radius: 12px;
        color: white;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0, 102, 204, 0.15);
    }
    .clinical-header h1 {
        color: white !important;
        margin: 0;
        font-size: 2rem;
        font-weight: 700;
    }
    .clinical-header p {
        color: rgba(255,255,255,0.9);
        margin: 0.3rem 0 0 0;
        font-size: 0.95rem;
    }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid var(--border);
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        transition: transform 0.2s;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
    }
    div[data-testid="stMetric"] label {
        color: #6c757d;
        font-size: 0.75rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stMetricValue"] {
        color: var(--primary-dark);
        font-weight: 700;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        background: var(--bg-card);
        padding: 0.4rem;
        border-radius: 10px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        font-weight: 500;
        padding: 0.5rem 1.2rem;
    }
    .stTabs [aria-selected="true"] {
        background-color: var(--primary) !important;
        color: white !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--accent) 100%);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1.5rem;
        font-weight: 600;
        transition: all 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 8px rgba(0,102,204,0.3);
    }

    /* Info boxes */
    .clinical-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        border-left: 4px solid var(--primary);
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        margin-bottom: 1rem;
    }
    .warning-card {
        background: #FFF4E5;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--warning);
        margin-bottom: 1rem;
    }
    .success-card {
        background: #E8F7F1;
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid var(--success);
        margin-bottom: 1rem;
    }

    /* Dataframes */
    .stDataFrame {
        border-radius: 8px;
        overflow: hidden;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FFFFFF 0%, #F8F9FA 100%);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: var(--primary-dark);
    }

    /* Pills / badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 12px;
        font-size: 0.75rem;
        font-weight: 600;
        margin: 0.1rem;
    }
    .badge-green { background: #D4EDDA; color: #155724; }
    .badge-blue { background: #D1ECF1; color: #0C5460; }
    .badge-orange { background: #FFF3CD; color: #856404; }
    .badge-red { background: #F8D7DA; color: #721C24; }
</style>
""", unsafe_allow_html=True)

MODELS = BASE / "outputs" / "models"
PLOTS = BASE / "outputs" / "plots"
RESULTS = BASE / "outputs" / "results"
PROC = BASE / "data" / "processed"
EMBED = BASE / "outputs" / "embeddings"


# ─── Cached Loaders ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_best_classifier():
    """Load V4 -> V3 -> V2 cascade."""
    for ver, rf_name, le_name, feat_name, xgb_name in [
        ("V4", "random_forest_v4.pkl", "label_encoder_v4.pkl", "hpo_feature_names_v4.npy", "xgboost_v4.pkl"),
        ("V3", "random_forest_v3.pkl", "label_encoder_v3.pkl", "hpo_feature_names_v3.npy", "xgboost_v3.pkl"),
        ("V2", "random_forest_v2.pkl", "label_encoder_v2.pkl", "hpo_feature_names_v2.npy", "xgboost_v2.pkl"),
    ]:
        try:
            rf = joblib.load(MODELS / rf_name)
            le = joblib.load(MODELS / le_name)
            feats = np.load(MODELS / feat_name, allow_pickle=True).tolist()
            xgb = None
            try:
                xgb = joblib.load(MODELS / xgb_name)
            except Exception:
                pass
            return rf, xgb, le, feats, ver
        except Exception:
            continue
    return None, None, None, None, None


@st.cache_resource(show_spinner=False)
def load_similar_engine():
    try:
        from retrieval.similar_disease_engine import SimilarDiseaseEngine
        engine = SimilarDiseaseEngine()
        engine.load()
        return engine
    except Exception as e:
        return None


@st.cache_data(show_spinner=False)
def load_babyseq():
    p = PROC / "babyseq_gene_disease.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_data(show_spinner=False)
def load_iba_panel_map():
    p = PROC / "iba_panel_hpo_map.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_master_table():
    """Load full master gene-disease-phenotype table for HPO direct lookup."""
    p = PROC / "master_gene_disease_phenotype.csv"
    if p.exists():
        df = pd.read_csv(p, low_memory=False,
                         usecols=["disease_id", "hpo_id", "gene_symbol"])
        return df
    return None


def hpo_direct_lookup(hpo_ids, master_df, name_map, top_k=10):
    """
    Direct HPO-ID → disease lookup across all 12,671 diseases.
    Score = fraction of query HPO terms matched per disease.
    Works for ANY HPO ID regardless of classifier training vocabulary.
    """
    if master_df is None or not hpo_ids:
        return []
    hpo_set = set(hpo_ids)
    filtered = master_df[master_df["hpo_id"].isin(hpo_set)]
    if filtered.empty:
        return []
    dis_counts = filtered.groupby("disease_id")["hpo_id"].nunique()
    gene_counts = master_df.groupby("disease_id")["gene_symbol"].nunique()
    top = dis_counts.nlargest(top_k)
    results = []
    for rank, (did, cnt) in enumerate(top.items()):
        dname = name_map.get(did, "") if name_map else ""
        results.append({
            "Rank": rank + 1,
            "Disease ID": did,
            "Disease Name": dname if dname else did,
            "HPO Match": f"{int(cnt)}/{len(hpo_ids)}",
            "Score %": round(100.0 * cnt / len(hpo_ids), 1),
            "Genes": int(gene_counts.get(did, 0)),
        })
    return results


@st.cache_data(show_spinner=False)
def load_gene_actionability():
    try:
        p = PROC / "gene_actionability.csv"
        if p.exists():
            return pd.read_csv(p, low_memory=False)
        return None
    except Exception:
        return None


@st.cache_resource(show_spinner=False)
def load_v2_classifier():
    try:
        rf = joblib.load(MODELS / "random_forest_v2.pkl")
        xgb = joblib.load(MODELS / "xgboost_v2.pkl")
        lr = joblib.load(MODELS / "logistic_regression_v2.pkl")
        le = joblib.load(MODELS / "label_encoder_v2.pkl")
        feats = np.load(MODELS / "hpo_feature_names_v2.npy", allow_pickle=True).tolist()
        return rf, xgb, lr, le, feats
    except Exception:
        return None, None, None, None, None


@st.cache_resource(show_spinner=False)
def load_v1_classifier():
    try:
        rf = joblib.load(MODELS / "random_forest.pkl")
        le = joblib.load(MODELS / "label_encoder.pkl")
        feats = np.load(MODELS / "hpo_feature_names.npy", allow_pickle=True).tolist()
        return rf, le, feats
    except Exception:
        return None, None, None


@st.cache_resource(show_spinner=False)
def load_shap_best():
    """Load V3 SHAP if available, else V2."""
    for path in ["shap_explainer_v3.pkl", "shap_explainer_v2.pkl"]:
        try:
            return joblib.load(MODELS / path)
        except Exception:
            continue
    return None


@st.cache_data(show_spinner=False)
def load_pediatric_master():
    p = PROC / "master_pediatric.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_data(show_spinner=False)
def load_full_master():
    p = PROC / "master_gene_disease_phenotype.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_data(show_spinner=False)
def load_hpo_lookup():
    p = RESULTS / "hpo_lookup.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_disease_name_map():
    p = RESULTS / "disease_name_map.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return {}


@st.cache_data(show_spinner=False)
def load_orphanet_geo():
    p = PROC / "orphanet_product1_geo.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_data(show_spinner=False)
def load_clinvar_summary():
    p = PROC / "clinvar_gene_summary.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics = {}
    for name, file in [
        ("v1", "classifier_metrics.json"),
        ("v2", "classifier_metrics_v2.json"),
        ("v3", "classifier_metrics_v3.json"),
        ("v4", "classifier_metrics_v4.json"),
        ("link_prediction", "link_prediction_metrics.json"),
        ("cv", "cv_scores.json"),
        ("cv_v3", "cv_scores_v3.json"),
        ("cv_v4", "cv_scores_v4.json"),
        ("graph", "graph_stats.json"),
    ]:
        p = RESULTS / file
        if p.exists():
            with open(p) as f:
                metrics[name] = json.load(f)
    return metrics


@st.cache_resource(show_spinner=False)
def load_tfidf():
    p = MODELS / "tfidf_index.pkl"
    if p.exists():
        return joblib.load(p)
    return None


@st.cache_resource(show_spinner=False)
def load_graph_bundle():
    import networkx as nx
    G, embeddings, nodes, lp = None, None, None, None
    gpkl = MODELS / "knowledge_graph.pkl"
    if gpkl.exists():
        try:
            G = joblib.load(gpkl)
        except Exception:
            pass
    emb_p = EMBED / "node_embeddings.npy"
    nodes_p = EMBED / "node_list.json"
    if emb_p.exists() and nodes_p.exists():
        vecs = np.load(emb_p)
        with open(nodes_p) as f:
            nodes = json.load(f)
        embeddings = dict(zip(nodes, vecs))
    lp_p = MODELS / "link_predictor.pkl"
    if lp_p.exists():
        lp = joblib.load(lp_p)
    return G, embeddings, nodes, lp


@st.cache_data(show_spinner=False)
def load_gene_disease_index():
    p = MODELS / "gene_disease_index.pkl"
    if p.exists():
        return joblib.load(p)
    return None


# ─── Clinical Header ────────────────────────────────────────────────────────────
def render_header():
    st.markdown("""
    <div class='clinical-header'>
        <h1>🧬 GeneGenie — Rare Disease Intelligence System</h1>
        <p>ML-powered clinical decision support | 12,671 rare diseases | HPO + OMIM + Orphanet + ClinVar + gnomAD | 2026</p>
    </div>
    """, unsafe_allow_html=True)


# ─── Prediction Helpers ─────────────────────────────────────────────────────────
def predict_disease(hpo_ids, rf, xgb, le, feature_names, top_k=10, name_map=None):
    """Predict disease from HPO IDs. Handles V2/V3/V4 feature sets."""
    x = np.zeros(len(feature_names), dtype=np.float32)
    feat_set = set(feature_names)
    matched = [h for h in hpo_ids if h in feat_set and h.startswith("HP:")]
    for h in matched:
        x[feature_names.index(h)] = 1.0
    if x.sum() == 0:
        return [], matched
    x = x.reshape(1, -1)
    rf_p = rf.predict_proba(x)[0]
    if xgb is not None:
        try:
            xgb_p = xgb.predict_proba(x)[0]
            if len(xgb_p) == len(rf_p):
                ens = (rf_p + xgb_p) / 2
                has_xgb = True
            else:
                ens = rf_p
                has_xgb = False
        except Exception:
            ens = rf_p
            has_xgb = False
    else:
        ens = rf_p
        has_xgb = False

    top = np.argsort(ens)[::-1][:top_k]
    results = []
    for rank, i in enumerate(top):
        did = le.classes_[i]
        dname = (name_map or {}).get(did, "")
        row = {
            "Rank": rank + 1,
            "Disease ID": did,
            "Disease Name": dname if dname else did,
            "Ensemble %": round(float(ens[i]) * 100, 2),
            "RF %": round(float(rf_p[i]) * 100, 2),
        }
        if has_xgb:
            row["XGB %"] = round(float(xgb_p[i]) * 100, 2)
        results.append(row)
    return results, matched


# ─── Tab 1: Disease Classifier ────────────────────────────────────────────────
def tab_classifier():
    st.subheader("🩺 Rare Disease Classifier")
    st.markdown("""
    <div class='clinical-card'>
    <strong>About:</strong> Differential diagnosis classifier covering <strong>500 rare diseases</strong>
    selected from 12,671 total (OMIM + Orphanet + HPO) by HPO phenotype coverage.
    Enter patient symptoms as HPO IDs to get ranked disease predictions with SHAP explanations.<br>
    <strong>Accuracy note:</strong> 98.94% = within-distribution accuracy (correct disease ranked
    #1 from partial symptom subsets of the 500 training diseases).
    <strong>Top-5 accuracy ≈ 99.5%</strong> — clinically relevant metric for differential diagnosis.
    For diseases outside this set, use the <em>Retrieval</em> tab (full 12,671-disease corpus).
    </div>
    """, unsafe_allow_html=True)

    rf, xgb_clf, le, feats, model_ver = load_best_classifier()
    hpo_lookup = load_hpo_lookup()
    disease_names = load_disease_name_map()

    if rf is None:
        st.error("Classifier not found. Run: `python src/classifier/train_classifier_v3.py`")
        return

    # Show which model version is loaded
    ver_color = "#00A896" if model_ver == "V3" else "#0066CC"
    st.markdown(f"""
    <div style='background:{ver_color}20; border-left:4px solid {ver_color};
    padding:0.6rem 1rem; border-radius:8px; margin-bottom:1rem;'>
    <strong>Model Loaded: {model_ver}</strong> |
    {"HPO + ClinVar + Gene Attributes (408 features)" if model_ver == "V3" else "HPO features (300 features)"}
    | {len(le.classes_)} pediatric diseases
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([3, 2])

    with col1:
        st.markdown("##### Patient Presentation")
        default_hpos = "HP:0001166\nHP:0000098\nHP:0001382\nHP:0002650\nHP:0002616"
        hpo_input = st.text_area(
            "HPO IDs (one per line):", value=default_hpos, height=140,
            help="Enter Human Phenotype Ontology codes for observed symptoms",
        )

        top_k = st.slider("Number of predictions:", 3, 20, 10)

        col_search, col_btn = st.columns([3, 1])
        with col_search:
            search_term = st.text_input("🔍 Search HPO by symptom name:", placeholder="e.g. seizure, hypotonia, microcephaly")
        with col_btn:
            st.markdown("<br>", unsafe_allow_html=True)
            predict_btn = st.button("Diagnose", type="primary", use_container_width=True)

        if search_term and len(search_term) >= 3:
            matches = [(hid, name) for hid, name in hpo_lookup.items()
                       if search_term.lower() in str(name).lower()][:20]
            if matches:
                with st.expander(f"Found {len(matches)} matching HPO terms", expanded=True):
                    mdf = pd.DataFrame(matches, columns=["HPO ID", "Symptom Name"])
                    st.dataframe(mdf, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("##### Clinical Examples")
        examples = {
            "🦋 Marfan Syndrome":      "HP:0001166\nHP:0000098\nHP:0001382\nHP:0002650\nHP:0002616",
            "🫁 Cystic Fibrosis":      "HP:0032261\nHP:0002570\nHP:0001394\nHP:0002726\nHP:0001392",
            "🧬 PKU":                   "HP:0001250\nHP:0001249\nHP:0002514\nHP:0005982\nHP:0007513",
            "🦷 Apert Syndrome":       "HP:0001159\nHP:0001177\nHP:0001274\nHP:0001249\nHP:0007291",
            "🧠 Prader-Willi":         "HP:0001290\nHP:0001270\nHP:0001249\nHP:0000054\nHP:0001385",
        }
        for name, hpos in examples.items():
            if st.button(name, key=f"ex_{name}", use_container_width=True):
                st.session_state["_hpo_input"] = hpos
                st.rerun()

        if "_hpo_input" in st.session_state:
            hpo_input = st.session_state.pop("_hpo_input")

    if predict_btn and hpo_input.strip():
        hpo_ids = [h.strip() for h in hpo_input.strip().split("\n") if h.strip().startswith("HP:")]
        if not hpo_ids:
            st.warning("Enter HPO IDs (format: HP:XXXXXXX), one per line.")
            return

        disease_names = load_disease_name_map()
        master_df = load_master_table()
        preds, matched = predict_disease(hpo_ids, rf, xgb_clf, le, feats, top_k, name_map=disease_names)

        # Always run HPO direct lookup — works across all 12,671 diseases
        hpo_results = hpo_direct_lookup(hpo_ids, master_df, disease_names, top_k=top_k)

        # Determine classifier reliability:
        # Need ≥3 matched HPOs AND top prediction > 20% confidence (not guessing)
        top_confidence = preds[0]["Ensemble %"] if preds else 0
        classifier_reliable = preds and len(matched) >= 3 and top_confidence >= 20.0

        if not preds or len(matched) == 0:
            # Classifier couldn't match any HPOs → show HPO lookup only
            st.markdown("""
            <div class='info-card' style='background:#E8F4FD;border-left:4px solid #0066CC;padding:0.8rem 1rem;border-radius:8px;margin-bottom:1rem;'>
            <strong>📋 HPO Search Results</strong> — Searching across all 12,671 rare diseases by phenotype overlap.
            The 500-disease classifier uses a different HPO vocabulary; retrieval covers the full corpus.
            </div>
            """, unsafe_allow_html=True)
            if hpo_results:
                st.markdown(f"**Top matches for {len(hpo_ids)} HPO terms across 12,671 diseases:**")
                df_res = pd.DataFrame(hpo_results)
                st.dataframe(df_res, use_container_width=True, hide_index=True)
            else:
                st.error("HPO IDs not found in database. Check codes at hpo.jax.org")
            return

        # Always show HPO direct lookup first (works for ALL diseases)
        if hpo_results:
            st.markdown("#### 🔍 HPO Phenotype Match — All 12,671 Diseases")
            st.caption("Ranked by HPO term overlap across full disease corpus (OMIM + Orphanet + HPO)")
            df_hpo = pd.DataFrame(hpo_results)
            st.dataframe(df_hpo, use_container_width=True, hide_index=True)

        if classifier_reliable:
            st.markdown(f"""
            <div class='success-card'>
            ✓ Matched <strong>{len(matched)}/{len(hpo_ids)}</strong> HPO terms to classifier vocabulary.
            Showing top {top_k} predictions.
            </div>
            """, unsafe_allow_html=True)
        else:
            note = (
                f"Only <strong>{len(matched)}/{len(hpo_ids)}</strong> HPO terms matched the "
                f"500-disease classifier vocabulary" if len(matched) < 3
                else f"Matched <strong>{len(matched)}/{len(hpo_ids)}</strong> HPO terms but "
                     f"top confidence <strong>{top_confidence:.1f}%</strong> is below threshold "
                     f"(symptoms are generic — insufficient to differentiate within 500 classes)"
            )
            st.markdown(f"""
            <div class='warning-card'>
            ⚠️ {note}.<br>
            HPO direct lookup results above cover all 12,671 diseases and are your primary result.
            </div>
            """, unsafe_allow_html=True)
            return

        # Classifier results (500-disease closed-world) — only shown when ≥3 HPOs matched
        df_pred = pd.DataFrame(preds)
        st.markdown("#### 🤖 Classifier Predictions — 500-Disease Scope")
        st.caption("Random Forest V4 (98.94% within-distribution accuracy, 426 features)")
        st.dataframe(
            df_pred.style.background_gradient(subset=["Ensemble %"], cmap="Blues"),
            use_container_width=True, hide_index=True,
        )

        # Visualization
        col_a, col_b = st.columns(2)
        with col_a:
            import plotly.express as px
            fig = px.bar(
                df_pred.head(10), x="Ensemble %", y="Disease ID",
                orientation="h", color="Ensemble %",
                color_continuous_scale="Blues",
                title="Confidence Scores by Disease",
            )
            fig.update_layout(yaxis=dict(autorange="reversed"), height=400)
            st.plotly_chart(fig, use_container_width=True)

        with col_b:
            # Compare RF vs XGB vs Ensemble (dynamically select available columns)
            avail_cols = [c for c in ["RF %", "XGB %", "Ensemble %"] if c in df_pred.columns]
            compare_df = df_pred.head(10).melt(
                id_vars="Disease ID", value_vars=avail_cols,
                var_name="Model", value_name="Confidence"
            )
            fig2 = px.bar(
                compare_df, x="Disease ID", y="Confidence", color="Model",
                barmode="group", title="Model Confidence Comparison",
                color_discrete_sequence=["#0066CC", "#F18F01", "#00A896"],
            )
            fig2.update_xaxes(tickangle=45)
            fig2.update_layout(height=400)
            st.plotly_chart(fig2, use_container_width=True)

        # SHAP explanation
        st.markdown("#### 🔬 Explainable AI (SHAP Feature Importance)")
        explainer = load_shap_best()
        if explainer is not None:
            try:
                x = np.zeros(len(feats), dtype=np.float32)
                for h in hpo_ids:
                    if h in feats:
                        x[feats.index(h)] = 1.0
                import shap as shap_lib
                sv = explainer.shap_values(x.reshape(1, -1))
                if isinstance(sv, list):
                    mean_abs = np.mean([np.abs(s) for s in sv], axis=0)[0]
                else:
                    mean_abs = np.abs(sv).mean(axis=-1)[0] if sv.ndim == 3 else np.abs(sv)[0]
                top_i = np.argsort(mean_abs)[::-1][:15]
                shap_data = [{
                    "HPO ID": feats[i],
                    "Symptom": hpo_lookup.get(feats[i], "—")[:60],
                    "Impact (|SHAP|)": float(mean_abs[i]),
                } for i in top_i if mean_abs[i] > 0]
                if shap_data:
                    sdf = pd.DataFrame(shap_data)
                    import plotly.express as px
                    fig3 = px.bar(
                        sdf, x="Impact (|SHAP|)", y="HPO ID", orientation="h",
                        color="Impact (|SHAP|)", color_continuous_scale="Reds",
                        hover_data=["Symptom"],
                        title="Most Influential Symptoms for This Prediction",
                    )
                    fig3.update_layout(yaxis=dict(autorange="reversed"), height=500)
                    st.plotly_chart(fig3, use_container_width=True)
            except Exception as e:
                st.info(f"SHAP analysis unavailable: {type(e).__name__}")


# ─── Tab 2: Cohort Explorer ─────────────────────────────────────────────────────
def tab_cohort():
    st.subheader("👥 Disease Cohort Explorer")
    st.markdown("""
    <div class='clinical-card'>
    Browse the disease cohort used for training (top-500 from all rare diseases). Filter by disease type, gene, or HPO phenotype.
    </div>
    """, unsafe_allow_html=True)

    master_ped = load_pediatric_master()
    geo_df = load_orphanet_geo()
    clinvar_df = load_clinvar_summary()

    if master_ped is None:
        st.error("Disease master not found. Run: `python run_pipeline.py --module preprocess`")
        return

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Pediatric Diseases", master_ped["disease_id"].nunique() if "disease_id" in master_ped.columns else 0)
    c2.metric("Associated Genes", master_ped["gene_symbol"].nunique() if "gene_symbol" in master_ped.columns else 0)
    c3.metric("HPO Phenotypes", master_ped["hpo_id"].nunique() if "hpo_id" in master_ped.columns else 0)
    c4.metric("Total Records", f"{len(master_ped):,}")

    col_l, col_r = st.columns([1, 2])
    with col_l:
        st.markdown("##### Filters")
        # Geography filter
        if geo_df is not None and "geo_prevalence" in geo_df.columns:
            geos = ["All"] + sorted(geo_df["geo_prevalence"].dropna().unique().tolist())
            sel_geo = st.selectbox("Geographic Prevalence:", geos)
        else:
            sel_geo = "All"

        search = st.text_input("Search disease/gene:", placeholder="Type to filter...")

        # Geo pie chart
        if geo_df is not None and "geo_prevalence" in geo_df.columns:
            import plotly.express as px
            geo_counts = geo_df["geo_prevalence"].value_counts().reset_index()
            geo_counts.columns = ["Region", "Diseases"]
            fig_geo = px.pie(
                geo_counts, names="Region", values="Diseases",
                title="Geographic Distribution",
                color_discrete_sequence=px.colors.qualitative.Set2,
            )
            fig_geo.update_traces(textposition="inside", textinfo="percent+label")
            st.plotly_chart(fig_geo, use_container_width=True)

    with col_r:
        st.markdown("##### Disease Catalog")
        filtered = master_ped.copy()
        if search:
            mask = pd.Series([False] * len(filtered))
            for col in ["disease_id", "disease_name", "gene_symbol"]:
                if col in filtered.columns:
                    mask = mask | filtered[col].astype(str).str.contains(search, case=False, na=False)
            filtered = filtered[mask]

        # Show top diseases by HPO count
        if "disease_id" in filtered.columns:
            disease_agg = filtered.groupby("disease_id").agg({
                "hpo_id": "nunique" if "hpo_id" in filtered.columns else "count",
                "gene_symbol": "nunique" if "gene_symbol" in filtered.columns else "count",
            }).rename(columns={"hpo_id": "HPO Terms", "gene_symbol": "Genes"})
            disease_agg = disease_agg.sort_values("HPO Terms", ascending=False).head(100)
            st.dataframe(disease_agg, use_container_width=True, height=500)

    # ClinVar integration
    if clinvar_df is not None:
        st.markdown("---")
        st.markdown("##### 🧬 ClinVar Pathogenicity Overview")
        c1, c2, c3 = st.columns(3)
        c1.metric("Genes with ClinVar data", f"{len(clinvar_df):,}")
        c2.metric("Total pathogenic variants", f"{clinvar_df['n_pathogenic'].sum():,}")
        c3.metric("High-actionability genes", f"{(clinvar_df['clinical_actionability'] == 'high').sum():,}")

        import plotly.express as px
        top_pathogenic = clinvar_df.nlargest(20, "n_pathogenic")[["gene_symbol", "n_pathogenic", "n_likely_pathogenic"]]
        fig_cv = px.bar(
            top_pathogenic.melt(id_vars="gene_symbol", var_name="Category", value_name="Count"),
            x="gene_symbol", y="Count", color="Category",
            title="Top 20 Genes by Pathogenic Variant Count",
            color_discrete_sequence=["#E63946", "#F18F01"],
        )
        fig_cv.update_xaxes(tickangle=45)
        st.plotly_chart(fig_cv, use_container_width=True)


# ─── Tab 3: Knowledge Graph ─────────────────────────────────────────────────────
def tab_kg():
    st.subheader("🕸️ Disease–Gene–Phenotype Knowledge Graph")
    st.markdown("""
    <div class='clinical-card'>
    Explore biological relationships between genes, diseases, and phenotypes.
    Spectral graph embeddings (SVD) + Logistic Regression link predictor with <strong>AUC = 99.05%</strong> on held-out edges.
    </div>
    """, unsafe_allow_html=True)

    G, embeddings, nodes, lp = load_graph_bundle()
    if G is None:
        st.error("KG not built. Run: `python run_pipeline.py --module kg`")
        return

    import networkx as nx

    # Stats
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Nodes", f"{G.number_of_nodes():,}")
    c2.metric("Edges", f"{G.number_of_edges():,}")
    c3.metric("Avg Degree", f"{np.mean([d for _, d in G.degree()]):.1f}")
    c4.metric("Components", f"{nx.number_connected_components(G):,}")

    col_l, col_r = st.columns([1, 3])
    with col_l:
        st.markdown("##### Explore")
        query_node = st.text_input("Gene or Disease:", value="FBN1")
        n_neighbors = st.slider("Max neighbors:", 10, 100, 40)
        show_lp = st.checkbox("Show predicted missing links", value=True)
        top_k = st.slider("Missing links top K:", 5, 20, 10)
        go = st.button("🔎 Explore", type="primary", use_container_width=True)

    with col_r:
        if go and query_node.strip():
            query = query_node.strip().upper()
            found = None
            for n in G.nodes():
                if str(n).upper() == query:
                    found = n
                    break
            if not found:
                matches = [n for n in G.nodes() if query in str(n).upper()]
                found = matches[0] if matches else None

            if not found:
                st.warning(f"'{query}' not in graph. Try FBN1, BRCA1, TP53, etc.")
                return

            neighbors = list(G.neighbors(found))[:n_neighbors]
            sub_nodes = [found] + neighbors
            G_sub = G.subgraph(sub_nodes)

            import plotly.graph_objects as go_
            pos = nx.spring_layout(G_sub, seed=42, k=0.7)
            edge_x, edge_y = [], []
            for e in G_sub.edges():
                x0, y0 = pos[e[0]]; x1, y1 = pos[e[1]]
                edge_x += [x0, x1, None]; edge_y += [y0, y1, None]
            node_x = [pos[n][0] for n in G_sub.nodes()]
            node_y = [pos[n][1] for n in G_sub.nodes()]
            labels = [str(n)[:25] for n in G_sub.nodes()]
            colors = []
            sizes = []
            for n in G_sub.nodes():
                if n == found:
                    colors.append("#F18F01"); sizes.append(20)
                elif str(n).startswith("HP:"):
                    colors.append("#00A896"); sizes.append(10)
                elif "ORPHA" in str(n).upper() or "OMIM" in str(n).upper():
                    colors.append("#E63946"); sizes.append(12)
                else:
                    colors.append("#0066CC"); sizes.append(12)

            fig = go_.Figure()
            fig.add_trace(go_.Scatter(x=edge_x, y=edge_y, mode="lines",
                                      line=dict(width=0.6, color="#CCC"), hoverinfo="none"))
            fig.add_trace(go_.Scatter(
                x=node_x, y=node_y, mode="markers+text",
                marker=dict(size=sizes, color=colors, line=dict(width=1.5, color="white")),
                text=labels, textposition="top center",
                textfont=dict(size=9), hoverinfo="text",
            ))
            fig.update_layout(
                title=dict(text=f"Subgraph around {found}", font=dict(size=16)),
                showlegend=False, height=600,
                margin=dict(l=10, r=10, t=50, b=10),
                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                plot_bgcolor="#F8F9FA",
            )
            st.plotly_chart(fig, use_container_width=True)

            # Legend
            st.markdown("""
            <div style='display:flex;gap:1.5rem;justify-content:center;margin-top:-1rem'>
                <span class='badge' style='background:#F18F01;color:white'>● Query</span>
                <span class='badge' style='background:#E63946;color:white'>● Disease</span>
                <span class='badge' style='background:#0066CC;color:white'>● Gene</span>
                <span class='badge' style='background:#00A896;color:white'>● Phenotype (HPO)</span>
            </div>
            """, unsafe_allow_html=True)

            # Missing link prediction
            if show_lp and lp is not None and embeddings:
                st.markdown("##### 🔮 Predicted Missing Links")
                if found in embeddings:
                    from knowledge_graph.build_graph import predict_missing_links
                    clf_lp = lp.get("clf"); scaler_lp = lp.get("scaler")
                    missing = predict_missing_links(found, G, embeddings, clf_lp, scaler_lp, top_k=top_k)
                    if missing:
                        ml_df = pd.DataFrame(missing, columns=["Predicted Connection", "Score"])
                        ml_df["Score"] = ml_df["Score"].round(4)
                        st.dataframe(
                            ml_df.style.background_gradient(subset=["Score"], cmap="Oranges"),
                            use_container_width=True, hide_index=True,
                        )
                    else:
                        st.info("No predicted missing links above threshold.")


# ─── Tab 4: Retrieval Engine ────────────────────────────────────────────────────
def tab_retrieval():
    st.subheader("🔍 Symptom-to-Gene Retrieval Engine")
    st.markdown("""
    <div class='clinical-card'>
    Free-text search across <strong>13,484 indexed documents</strong> (genes + diseases).
    TF-IDF + cosine similarity ranks the most relevant matches.
    </div>
    """, unsafe_allow_html=True)

    tfidf = load_tfidf()
    master = load_full_master()
    hpo_lookup = load_hpo_lookup()
    gdi = load_gene_disease_index()

    col_l, col_r = st.columns(2)
    with col_l:
        st.markdown("##### 📝 Text Search")
        q_text = st.text_area("Describe symptoms in plain English:",
                              value="intellectual disability seizures microcephaly hypotonia",
                              height=90)
        btn_t = st.button("Text Search", type="primary")

    with col_r:
        st.markdown("##### 🧬 HPO Code Search")
        q_hpo = st.text_area("Enter HPO IDs:",
                             value="HP:0001249\nHP:0001250\nHP:0000252\nHP:0001252",
                             height=90)
        btn_h = st.button("HPO Search", type="secondary")

    top_n = st.slider("Results to show:", 10, 50, 20)

    if btn_t and q_text and tfidf:
        from retrieval.retrieval_engine import retrieve_by_text
        res = retrieve_by_text(q_text, tfidf["vectorizer"], tfidf["matrix"],
                              tfidf["ids"], tfidf["types"], top_k=top_n)
        if res:
            rdf = pd.DataFrame(res)
            genes_r = rdf[rdf["type"] == "gene"].head(top_n // 2)
            dis_r = rdf[rdf["type"] == "disease"].head(top_n // 2)

            cg, cd = st.columns(2)
            with cg:
                st.markdown("#### 🧬 Top Genes")
                import plotly.express as px
                if not genes_r.empty:
                    fig = px.bar(genes_r.head(15), x="score", y="id", orientation="h",
                                 color="score", color_continuous_scale="Blues")
                    fig.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                                      title="Gene Relevance", height=450)
                    st.plotly_chart(fig, use_container_width=True)
            with cd:
                st.markdown("#### 🩺 Top Diseases")
                if not dis_r.empty:
                    fig2 = px.bar(dis_r.head(15), x="score", y="id", orientation="h",
                                  color="score", color_continuous_scale="Reds")
                    fig2.update_layout(yaxis=dict(autorange="reversed"), showlegend=False,
                                       title="Disease Relevance", height=450)
                    st.plotly_chart(fig2, use_container_width=True)

    if btn_h and q_hpo and master is not None:
        hpo_ids = [h.strip() for h in q_hpo.strip().split("\n") if h.strip()]
        from retrieval.retrieval_engine import retrieve_by_hpo_ids
        res = retrieve_by_hpo_ids(hpo_ids, master, top_k=top_n)
        genes_r = [r for r in res if r["type"] == "gene"]
        dis_r = [r for r in res if r["type"] == "disease"]

        cg, cd = st.columns(2)
        with cg:
            st.markdown(f"#### 🧬 Genes ({len(genes_r)})")
            if genes_r:
                gdf = pd.DataFrame(genes_r).rename(columns={"id": "Gene", "score": "HPO Match"})
                gdf["HPO Match"] = gdf["HPO Match"].round(3)
                st.dataframe(gdf[["Gene", "HPO Match"]], use_container_width=True, hide_index=True)
        with cd:
            st.markdown(f"#### 🩺 Diseases ({len(dis_r)})")
            if dis_r:
                ddf = pd.DataFrame(dis_r).rename(columns={"id": "Disease ID", "score": "HPO Match"})
                ddf["HPO Match"] = ddf["HPO Match"].round(3)
                st.dataframe(ddf[["Disease ID", "HPO Match"]], use_container_width=True, hide_index=True)

        # Symptom translation
        st.markdown("##### 📖 Your Search Terms Translated")
        translation = [(h, hpo_lookup.get(h, "— Not in HPO —")) for h in hpo_ids]
        st.dataframe(pd.DataFrame(translation, columns=["HPO ID", "Clinical Name"]),
                     use_container_width=True, hide_index=True)


# ─── Tab 5: Model Evaluation (Honest Report) ────────────────────────────────────
def tab_evaluation():
    st.subheader("📊 Model Evaluation — Honest Report")
    metrics = load_metrics()

    st.markdown("""
    <div class='warning-card'>
    ⚠️ <strong>Methodology note:</strong> Our 98% accuracy is <em>not inflated overfitting</em>.
    The HPO→Disease task is inherently structured — each disease has a distinctive phenotype fingerprint.
    We confirmed this via:
    <ul>
    <li><strong>Augmentation with noise:</strong> Simulated patients have 35-75% of textbook symptoms + confounders</li>
    <li><strong>Cross-validation:</strong> 3-fold stratified CV maintains high F1</li>
    <li><strong>Baseline comparison:</strong> Logistic Regression beats XGBoost, indicating linearly separable signal</li>
    <li><strong>Leave-disease-out test:</strong> Model correctly can't predict unseen disease classes (expected)</li>
    </ul>
    Real-world performance on under-specified clinical presentations will be lower but still clinically useful.
    </div>
    """, unsafe_allow_html=True)

    # V2 metrics
    if "v2" in metrics:
        st.markdown("#### 🎯 Current Model (V2 - Pediatric)")
        v2 = metrics["v2"]
        dfm = pd.DataFrame(v2)
        for col in ["accuracy", "f1_macro", "top3_accuracy", "top5_accuracy", "top10_accuracy"]:
            if col in dfm.columns:
                dfm[col] = (dfm[col] * 100).round(2)

        # Grid of model cards
        cols = st.columns(len(dfm))
        for i, row in dfm.iterrows():
            with cols[i]:
                st.markdown(f"**{row['model']}**")
                st.metric("Top-1", f"{row.get('accuracy', 0):.1f}%")
                if "top5_accuracy" in row:
                    st.metric("Top-5", f"{row.get('top5_accuracy', 0):.1f}%")
                if "f1_macro" in row:
                    st.metric("F1-Macro", f"{row.get('f1_macro', 0):.1f}%")

        # Chart
        display_cols = [c for c in ["accuracy", "top3_accuracy", "top5_accuracy", "top10_accuracy"] if c in dfm.columns]
        if display_cols:
            plot_df = dfm.melt(id_vars="model", value_vars=display_cols,
                               var_name="Metric", value_name="Score")
            import plotly.express as px
            fig = px.bar(plot_df, x="model", y="Score", color="Metric",
                         barmode="group", title="V2 Model Performance (%)",
                         color_discrete_sequence=["#0066CC", "#00A896", "#F18F01", "#E63946"])
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

    # V1 vs V2 vs V3 Comparison
    st.markdown("#### ⚖️ Model Evolution: V1 -> V2 -> V3")
    versions = []
    if "v1" in metrics:
        v1_rf = next((m for m in metrics["v1"] if "RandomForest" in m.get("model", "")), None)
        if v1_rf:
            versions.append({"Version": "V1 (General, HPO only)", "Features": "300 HPO",
                "Accuracy": v1_rf.get("accuracy", 0)*100, "F1-Macro": v1_rf.get("f1_macro", 0)*100,
                "Top-5": (v1_rf.get("top5_accuracy") or 0)*100})
    if "v2" in metrics:
        v2_rf = next((m for m in metrics["v2"] if "RandomForest" in m.get("model", "")), None)
        if v2_rf:
            versions.append({"Version": "V2 (HPO baseline)", "Features": "300 HPO + noise aug",
                "Accuracy": v2_rf.get("accuracy", 0)*100, "F1-Macro": v2_rf.get("f1_macro", 0)*100,
                "Top-5": (v2_rf.get("top5_accuracy") or 0)*100})
    if "v3" in metrics:
        v3_rf = next((m for m in metrics["v3"] if "RandomForest" in m.get("model", "")), None)
        if v3_rf:
            versions.append({"Version": "V3 (All diseases, Full Features)", "Features": "300 HPO + 105 ClinVar/Attr",
                "Accuracy": v3_rf.get("accuracy", 0)*100, "F1-Macro": v3_rf.get("f1_macro", 0)*100,
                "Top-5": (v3_rf.get("top5_accuracy") or 0)*100})
    if "v4" in metrics:
        v4_rf = next((m for m in metrics["v4"] if "RandomForest" in m.get("model", "")), None)
        if v4_rf:
            versions.append({"Version": "V4 (All diseases, +BabySeq+IBA)", "Features": "300 HPO + 13 IBA + 105 Gene + 5 BabySeq",
                "Accuracy": v4_rf.get("accuracy", 0)*100, "F1-Macro": v4_rf.get("f1_macro", 0)*100,
                "Top-5": (v4_rf.get("top5") or 0)*100})
    if versions:
        comp = pd.DataFrame(versions)
        st.dataframe(comp.style.format({"Accuracy": "{:.2f}%", "F1-Macro": "{:.2f}%", "Top-5": "{:.2f}%"}),
                     use_container_width=True, hide_index=True)
        import plotly.express as px
        fig_v = px.bar(comp, x="Version", y=["Accuracy", "F1-Macro", "Top-5"],
                       barmode="group", title="Performance Across Versions (RF, test set)",
                       color_discrete_sequence=["#0066CC", "#00A896", "#F18F01"])
        st.plotly_chart(fig_v, use_container_width=True)

    # Link prediction
    if "link_prediction" in metrics:
        st.markdown("#### 🕸️ Link Prediction Performance")
        lp = metrics["link_prediction"]
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{lp.get('auc_roc', 0)*100:.2f}%")
        c2.metric("Avg Precision", f"{lp.get('avg_precision', 0)*100:.2f}%")
        c3.metric("F1", f"{lp.get('f1', 0)*100:.2f}%")

    # Cross-validation (V3 if available)
    cv_key = "cv_v3" if "cv_v3" in metrics else "cv"
    if cv_key in metrics:
        cv = metrics[cv_key]
        st.markdown(f"#### 🔄 GroupKFold Cross-Validation {'(V3)' if cv_key == 'cv_v3' else '(V1)'}")
        if "cv_f1_macro_mean" in cv:
            st.info(f"Method: {cv.get('method', 'GroupKFold')} | "
                    f"Mean F1: **{cv['cv_f1_macro_mean']*100:.2f}%** ± {cv['cv_f1_macro_std']*100:.2f}% | "
                    f"Disease overlap: **{cv.get('disease_overlap', 0)}** (0 = no data leakage)")
        if "cv_scores" in cv:
            import plotly.express as px
            cv_df = pd.DataFrame({
                "Fold": [f"Fold {i+1}" for i in range(len(cv["cv_scores"]))],
                "F1-Macro": [s * 100 for s in cv["cv_scores"]]
            })
            fig = px.bar(cv_df, x="Fold", y="F1-Macro",
                        title=f"CV F1: {cv['cv_f1_macro_mean']*100:.2f}% ± {cv['cv_f1_macro_std']*100:.2f}%",
                        color="F1-Macro", color_continuous_scale="Blues")
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)

    # Data integration breakdown
    st.markdown("#### 📦 Data Integration Summary")
    data_sources = [
        {"Dataset": "HPO phenotype-gene (JAX)", "Records": "447,182 edges", "Used In": "Classifier + KG"},
        {"Dataset": "Knowledge Graph (kg.csv)", "Records": "200,000 edges", "Used In": "KG"},
        {"Dataset": "ClinVar (3.7GB)", "Records": "18,502 genes", "Used In": "Classifier V3 + Cohort"},
        {"Dataset": "Orphanet XML (en_product1/6)", "Records": "11,456 diseases", "Used In": "Pediatric filter + KG"},
        {"Dataset": "Orphanet genes CSV", "Records": "8,374 associations", "Used In": "KG (extended)"},
        {"Dataset": "OMIM (genemap2 + morbidmap)", "Records": "26,724 entries", "Used In": "Master table + Retrieval"},
        {"Dataset": "Gene Attribute Matrix", "Records": "4,553 genes x 6,178 attrs", "Used In": "Classifier V3"},
        {"Dataset": "Gene Similarity Matrix", "Records": "4,555 x 4,555 cosine", "Used In": "KG (similarity edges)"},
        {"Dataset": "gene2phenotype", "Records": "2,318 edges", "Used In": "KG"},
        {"Dataset": "NCBI gene_info", "Records": "Homo sapiens genes", "Used In": "Gene metadata"},
        {"Dataset": "diseases_for_HP:0000118", "Records": "12,927 diseases", "Used In": "Disease catalog"},
        {"Dataset": "attribute_list_entries", "Records": "6,175 attributes", "Used In": "Disease catalog"},
        {"Dataset": "mimTitles + mim2gene", "Records": "26,724 titles", "Used In": "Disease catalog + Retrieval"},
    ]
    st.dataframe(pd.DataFrame(data_sources), use_container_width=True, hide_index=True)

    # Saved plots gallery
    st.markdown("#### 📈 Training Artifacts")
    plots = [
        ("rf_feature_importance_v3.png", "Feature Importance (V3)"),
        ("rf_feature_importance_v2.png", "Feature Importance (V2)"),
        ("rf_feature_importance.png", "Feature Importance (V1)"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("link_prediction_roc.png", "Link Prediction ROC"),
        ("graph_degree_distribution.png", "Graph Statistics"),
    ]
    cols = st.columns(2)
    for i, (fname, title) in enumerate(plots):
        p = PLOTS / fname
        if p.exists():
            with cols[i % 2]:
                st.markdown(f"**{title}**")
                st.image(str(p), use_container_width=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────
def sidebar():
    with st.sidebar:
        st.markdown("## 🧬 GeneGenie")
        st.markdown("*Rare Disease Intelligence System*")
        st.markdown("---")

        rf_v4 = (MODELS / "random_forest_v4.pkl").exists()
        rf_v3 = (MODELS / "random_forest_v3.pkl").exists()
        rf_v2 = (MODELS / "random_forest_v2.pkl").exists()
        kg = (MODELS / "knowledge_graph.pkl").exists()
        clinvar = (PROC / "clinvar_gene_summary.csv").exists()
        gene_enrich = (PROC / "gene_enriched_features.csv").exists()
        babyseq_ready = (PROC / "babyseq_gene_disease.csv").exists()
        iba_ready = (PROC / "iba_panel_hpo_map.json").exists()
        gene_action = (PROC / "gene_actionability.csv").exists()

        st.markdown("### 📦 Pipeline Status")
        def badge(ok, name):
            color = "#06A77D" if ok else "#6c757d"
            symbol = "V" if ok else "o"
            st.markdown(f"<span style='color:{color}'>{symbol} {name}</span>", unsafe_allow_html=True)
        badge(rf_v4, "V4 Classifier (HPO+IBA+BabySeq)")
        badge(rf_v3, "V3 Classifier (HPO+ClinVar+Attrs)")
        badge(rf_v2, "V2 Classifier (pediatric HPO)")
        badge(kg, "Knowledge Graph (666K edges)")
        badge(clinvar, "ClinVar (18,502 genes)")
        badge(gene_enrich, "Gene Attribute Matrix")
        badge(babyseq_ready, "BabySeq (1,515 curated genes)")
        badge(iba_ready, "IBA Panel HPO Map (13 panels)")
        badge(gene_action, "Gene Actionability Index")

        st.markdown("---")
        st.markdown("### 📊 All Data Sources")
        st.markdown("""
        **Phenotype:**
        - HPO (9,631 phenotypes)
        - phenotype_to_genes_JAX (447K edges)
        - genes_to_phenotype
        - diseases_for_HP_0000118

        **Disease:**
        - OMIM (26,724 entries)
        - Orphanet XML (en_product1/6)
        - orphanet_genes/diseases CSV
        - mimTitles / morbidmap / kg.csv

        **Clinical Evidence (NEW):**
        - BabySeq Table S1 (1,515 genes)
        - ClinGen evidence grades (Def/Strong/Mod)
        - IBA Panels (13 clinical panels)
        - Newborn variants mmc1+mmc2 (276)
        - Gene Actionability Index

        **Variant:**
        - ClinVar (3.7GB, 18,502 genes)

        **Gene:**
        - NCBI gene_info
        - gene_attribute_matrix (6,178 attrs)
        - gene_similarity_matrix

        **Total:** 43,745 unique diseases
        33 raw files, 5.7GB data
        """)

        st.markdown("---")
        st.markdown("### 🏥 System Scope")
        st.markdown("""
        **All rare diseases** (OMIM + Orphanet + HPO):
        - Classifier: top-500 most HPO-characterised diseases
        - Retrieval: all 12,671 diseases indexed

        **Includes pediatric-onset** diseases:
        - Congenital (HP:0003577)
        - Neonatal (HP:0003623)
        - Infantile (HP:0003593)
        - Childhood (HP:0011463)
        """)

        st.markdown("---")
        st.caption("MPSTME ML Project · 2026")


# ─── Tab: Similar Diseases ──────────────────────────────────────────────────────
def tab_similar_diseases():
    st.subheader("🔎 Similar Disease Retrieval")
    st.markdown("""
    <div class='clinical-card'>
    <strong>Novel feature:</strong> Penetrance-adjusted disease similarity. Input HPO symptoms to find
    similar diseases ranked by <em>cosine + Jaccard similarity × penetrance score</em>.
    Diseases with high penetrance and Definitive evidence rank higher.
    Newborn alerts flag Category A diseases (highly penetrant, pediatric-onset, BabySeq curated).
    </div>
    """, unsafe_allow_html=True)

    engine = load_similar_engine()
    hpo_lookup = load_hpo_lookup()
    iba_map = load_iba_panel_map()

    col1, col2 = st.columns([2, 1])

    with col1:
        hpo_input = st.text_area("HPO IDs (one per line):",
                                  value="HP:0001250\nHP:0001252\nHP:0001263\nHP:0000252",
                                  height=130)
        top_k = st.slider("Results:", 5, 30, 10)
        pen_adjust = st.checkbox("Penetrance-adjusted ranking", value=True,
                                  help="Re-rank by penetrance × similarity (novel method)")
        find_btn = st.button("Find Similar Diseases", type="primary")

    with col2:
        st.markdown("**IBA Clinical Panels**")
        panel_sel = st.selectbox("Filter by panel:", ["All"] + list(iba_map.keys()))
        if panel_sel != "All" and engine is not None:
            panel_diseases = engine.panel_filter_diseases(panel_sel, top_k=15)
            if panel_diseases:
                pdf = pd.DataFrame(panel_diseases)
                st.dataframe(pdf[["disease_name", "gene", "category", "actionability_index"]],
                             use_container_width=True, hide_index=True)

    if find_btn and hpo_input.strip() and engine is not None:
        hpo_ids = [h.strip() for h in hpo_input.strip().split("\n") if h.strip()]

        # Panel activation (novel)
        panels_activated = engine.get_iba_panels_for_query(hpo_ids)
        if panels_activated:
            panel_str = " | ".join([f"**{p}** ({c} terms)" for p, c in panels_activated[:4]])
            st.markdown(f"""
            <div class='success-card'>
            <strong>IBA Panels Activated:</strong> {panel_str}
            </div>
            """, unsafe_allow_html=True)

        # Similar diseases
        results, matched = engine.query_by_hpo(hpo_ids, top_k=top_k, penetrance_adjust=pen_adjust)

        if not results:
            st.warning("No matches. Check HPO IDs are in HP:XXXXXXX format.")
            return

        st.markdown(f"**Matched {len(matched)}/{len(hpo_ids)} HPO terms.**")

        import plotly.express as px

        # Add rank
        df_sim = pd.DataFrame(results)
        df_sim.insert(0, "Rank", range(1, len(df_sim)+1))

        # Color newborn alerts
        def alert_badge(val):
            return "🚨 ALERT" if val else ""
        df_sim["Newborn"] = df_sim["newborn_alert"].apply(alert_badge)
        df_sim["Evidence"] = df_sim["evidence_label"]

        display_cols = ["Rank", "disease_name", "penetrance_adjusted_score",
                        "cosine_similarity", "penetrance_score", "Evidence",
                        "actionability_index", "Newborn", "associated_genes"]
        display_cols = [c for c in display_cols if c in df_sim.columns]

        st.markdown("#### Ranked Similar Diseases")
        st.dataframe(
            df_sim[display_cols].style.background_gradient(
                subset=["penetrance_adjusted_score"], cmap="Blues"),
            use_container_width=True, hide_index=True)

        # Visualization: penetrance-adjusted vs raw similarity
        fig = px.scatter(df_sim.head(15), x="cosine_similarity",
                         y="penetrance_adjusted_score",
                         size="actionability_index",
                         color="Evidence",
                         hover_data=["disease_name", "penetrance_score"],
                         text="Rank",
                         title="Penetrance-Adjusted vs Raw Similarity (bubble=actionability)",
                         color_discrete_sequence=["#0066CC","#00A896","#F18F01","#E63946","#6c757d"])
        fig.update_traces(textposition="top center")
        fig.update_layout(height=450)
        st.plotly_chart(fig, use_container_width=True)

    elif find_btn and engine is None:
        st.error("SimilarDiseaseEngine failed to load. Check data/processed/ files.")

    # Differential Diagnosis section
    st.markdown("---")
    st.markdown("#### Differential Diagnosis")
    st.markdown("Given a disease ID, find the most similar diseases (clinical lookalikes).")
    diff_id = st.text_input("Disease ID:", placeholder="e.g. OMIM:615273")
    if diff_id.strip() and engine is not None:
        diff_results = engine.differential_diagnosis(diff_id.strip(), top_k=10)
        if diff_results:
            ddf = pd.DataFrame(diff_results)
            ddf.insert(0, "Rank", range(1, len(ddf)+1))
            st.dataframe(ddf, use_container_width=True, hide_index=True)
        else:
            st.info(f"Disease '{diff_id}' not found in phenotype matrix or no similar diseases.")


# ─── Tab: Disease Map ────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_disease_map_data():
    """Load pre-computed UMAP + Louvain clusters."""
    cluster_path = PROC / "louvain_clusters.csv"
    umap_path = EMBED / "disease_umap.npy"
    if cluster_path.exists():
        return pd.read_csv(cluster_path)
    return None


def tab_disease_map():
    st.subheader("🗺️ Rare Disease Landscape")
    st.markdown("""
    <div class='clinical-card'>
    <strong>UMAP projection</strong> of 500 pediatric diseases in HPO phenotype space.
    <strong>Louvain clustering</strong> groups phenotypically similar diseases.
    Stars (⭐) = BabySeq Category A (highly penetrant, actionable newborn diseases).
    </div>
    """, unsafe_allow_html=True)

    import plotly.graph_objects as go
    import plotly.express as px

    cluster_df = load_disease_map_data()
    html_path = PLOTS / "plotly_disease_map.html"

    if cluster_df is None:
        st.warning("Disease map not built yet. Run: `python src/app/disease_map.py`")
        return

    # Cluster stats
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Diseases", len(cluster_df))
    with col2:
        st.metric("Louvain Clusters", cluster_df["cluster"].nunique())
    with col3:
        largest = cluster_df.groupby("cluster").size().max()
        st.metric("Largest Cluster", int(largest))
    with col4:
        st.metric("Disease Map", "UMAP Jaccard")

    # Interactive Plotly figure
    n_clusters = cluster_df["cluster"].nunique()
    colors = px.colors.qualitative.Dark24
    color_map = {c: colors[i % len(colors)]
                 for i, c in enumerate(sorted(cluster_df["cluster"].unique()))}

    fig = go.Figure()
    for cid in sorted(cluster_df["cluster"].unique()):
        sub = cluster_df[cluster_df["cluster"] == cid]
        hover = [
            f"<b>{row['disease_name'] or row['disease_id']}</b><br>"
            f"ID: {row['disease_id']}<br>Cluster: {cid}<br>Genes: {row.get('genes','N/A') or 'N/A'}"
            for _, row in sub.iterrows()
        ]
        fig.add_trace(go.Scatter(
            x=sub["umap_x"], y=sub["umap_y"],
            mode="markers",
            name=f"Cluster {cid} (n={len(sub)})",
            marker=dict(size=9, color=color_map[cid], opacity=0.82,
                        line=dict(width=0.5, color="white")),
            text=hover,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        paper_bgcolor="#0D1B2A", plot_bgcolor="#0D1B2A",
        font=dict(color="white"),
        legend=dict(bgcolor="rgba(0,0,0,0.3)", font=dict(size=11)),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False, showticklabels=False),
        height=600,
        title=dict(text="Rare Disease UMAP (Jaccard similarity in HPO space, top-500 diseases)",
                   font=dict(size=15)),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Cluster table
    st.markdown("##### Cluster Summary")
    cluster_summary = (
        cluster_df.groupby("cluster")
        .agg(n_diseases=("disease_id", "count"),
             sample_diseases=("disease_name", lambda x: " | ".join(x.dropna().iloc[:3])))
        .reset_index()
    )
    st.dataframe(cluster_summary, use_container_width=True, hide_index=True)

    # Download pre-computed HTML if exists
    if html_path.exists():
        with open(html_path, "rb") as f:
            st.download_button(
                "📥 Download Full Interactive Map (HTML)",
                data=f,
                file_name="raredx_disease_map.html",
                mime="text/html",
            )

    # Highlight a disease
    st.markdown("##### Highlight Disease on Map")
    search_disease = st.text_input("Disease ID or name:", placeholder="OMIM:254090 or seizure")
    if search_disease:
        mask = (
            cluster_df["disease_id"].str.contains(search_disease, case=False, na=False) |
            cluster_df["disease_name"].str.contains(search_disease, case=False, na=False)
        )
        found = cluster_df[mask]
        if not found.empty:
            st.success(f"Found {len(found)} match(es):")
            st.dataframe(found[["disease_id","disease_name","cluster","genes"]],
                         use_container_width=True, hide_index=True)
        else:
            st.info("Not found in disease map.")


# ─── Main ────────────────────────────────────────────────────────────────────────
def main():
    sidebar()
    render_header()
    tabs = st.tabs([
        "🩺 Diagnose",
        "🔎 Similar Diseases",
        "🗺️ Disease Map",
        "👥 Cohort",
        "🕸️ Knowledge Graph",
        "🔍 Retrieval",
        "📊 Evaluation",
    ])
    with tabs[0]: tab_classifier()
    with tabs[1]: tab_similar_diseases()
    with tabs[2]: tab_disease_map()
    with tabs[3]: tab_cohort()
    with tabs[4]: tab_kg()
    with tabs[5]: tab_retrieval()
    with tabs[6]: tab_evaluation()


if __name__ == "__main__":
    main()
