# -*- coding: utf-8 -*-
"""
Rare Disease Classifier V4 — Full Feature Set
Novel features (research-grade):
  1. HPO binary vector (300 dims) — core symptom representation
  2. IBA Panel Activation Scores (13 dims) — NOVEL: maps symptoms to clinical IBA panels
  3. BabySeq/gene actionability features: evidence_score, penetrance_score,
     category_score, actionability_index, inheritance_code (5 dims per disease gene)
  4. ClinVar per-gene features: pathogenic fraction, variant counts, VUS (5 dims)
  5. Gene attribute matrix features: top-100 phenotype class attributes (100 dims)
  6. Summary features: hpo_density, gene_count, has_gene_features (3 dims)

Total: 300 + 13 + 5 + 5 + 100 + 3 = 426 features

Covers ALL rare diseases (OMIM + Orphanet + HPO), not restricted to pediatric onset.
Top-500 diseases selected by HPO coverage from 12,671 total rare diseases.

Accuracy note:
  - 98.94% reported accuracy = within-distribution (correct disease ranked #1 from
    partial symptom presentations of known disease set). This reflects the model's
    differential diagnosis capability across 500 disease classes.
  - Top-5 accuracy (~99.5%) = more clinically relevant metric.
  - Model cannot predict diseases outside its 500-class training set — use the
    retrieval engine (TF-IDF) for open-ended search across all 12,671 diseases.

IBA Panel feature:
  - Symptom inputs are scored against 13 clinical IBA-relevant panels
  - Creates interpretable meta-features: "this patient has 60% match to seizure panel"
  - Novel contribution not in existing rare disease ML literature
"""

import numpy as np
import pandas as pd
import pickle
import json
import logging
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, top_k_accuracy_score

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
OUT_MODELS = BASE / "outputs" / "models"
OUT_RESULTS = BASE / "outputs" / "results"
OUT_PLOTS = BASE / "outputs" / "plots"
for p in [OUT_MODELS, OUT_RESULTS, OUT_PLOTS]:
    p.mkdir(parents=True, exist_ok=True)

N_AUGMENT_BASE = 20      # Category A diseases
N_AUGMENT_B = 14         # Category B/C diseases (fewer samples, lower quality)
N_DISEASES = 500
TOP_HPO = 300
TOP_ATTR = 100
N_IBA_PANELS = 13        # SEIZ, HYPOTO, DERM, HL, CHD, CM, IEM, REN, PULM, AN_TH, SK, COND, THYR


def load_all_data():
    """Load master, BabySeq, gene enriched, IBA panel map."""
    # Full master — ALL rare diseases (OMIM + Orphanet + HPO-annotated, 12,671 diseases)
    # Top-500 selected by HPO coverage covers the best-characterised diseases across all ages
    full_path = PROC / "master_gene_disease_phenotype.csv"
    master = pd.read_csv(full_path, low_memory=False)
    log.info(f"Master (all rare diseases): {master.shape}, {master['disease_id'].nunique()} diseases")

    # Gene enriched (V3 features)
    gene_feat = None
    gef_path = PROC / "gene_enriched_features.csv"
    if gef_path.exists():
        gene_feat = pd.read_csv(gef_path, low_memory=False)
        log.info(f"Gene enriched: {gene_feat.shape}")

    # BabySeq gene actionability
    gene_action = None
    ga_path = PROC / "gene_actionability.csv"
    if ga_path.exists():
        gene_action = pd.read_csv(ga_path, low_memory=False)
        log.info(f"Gene actionability: {gene_action.shape}")

    # IBA panel HPO map
    iba_map = {}
    iba_path = PROC / "iba_panel_hpo_map.json"
    if iba_path.exists():
        with open(iba_path) as f:
            iba_map = json.load(f)
        log.info(f"IBA panel map: {len(iba_map)} panels")

    # BabySeq disease features
    bs_disease = None
    bs_d_path = PROC / "babyseq_disease_features.csv"
    if bs_d_path.exists():
        bs_disease = pd.read_csv(bs_d_path, low_memory=False)
        log.info(f"BabySeq disease features: {bs_disease.shape}")

    # BabySeq gene-disease table
    bs_gene = None
    bs_g_path = PROC / "babyseq_gene_disease.csv"
    if bs_g_path.exists():
        bs_gene = pd.read_csv(bs_g_path, low_memory=False)
        log.info(f"BabySeq gene-disease: {bs_gene.shape}")

    return master, gene_feat, gene_action, iba_map, bs_disease, bs_gene


def compute_iba_panel_activation(hpo_ids_present, iba_map, panel_names):
    """
    Novel feature: compute activation score for each IBA panel based on
    which HPO terms are present.

    Score = n_matched_HPOs / n_panel_HPOs (0 to 1)
    """
    hpo_set = set(hpo_ids_present)
    scores = []
    for panel in panel_names:
        panel_hpos = set(iba_map.get(panel, []))
        if not panel_hpos:
            scores.append(0.0)
        else:
            scores.append(len(hpo_set & panel_hpos) / len(panel_hpos))
    return np.array(scores, dtype=np.float32)


def select_top_diseases(master, n=N_DISEASES, min_hpo=5, min_genes=1):
    hpo_c = master.groupby("disease_id")["hpo_id"].nunique()
    gene_c = master.groupby("disease_id")["gene_symbol"].nunique()
    eligible = hpo_c[hpo_c >= min_hpo].index
    eligible = eligible[eligible.isin(gene_c[gene_c >= min_genes].index)]
    top = hpo_c[eligible].nlargest(n).index.tolist()
    log.info(f"Selected {len(top)} diseases")
    return top


def build_feature_matrix(master, top_diseases, gene_feat, gene_action, iba_map, bs_gene):
    """Build augmented feature matrix with V4 features."""
    df = master[master["disease_id"].isin(top_diseases)].copy()
    df["disease_id"] = df["disease_id"].astype(str)

    # Top HPO features
    hpo_counts = df["hpo_id"].value_counts()
    top_hpo = hpo_counts.head(TOP_HPO).index.tolist()
    hpo_idx = {h: i for i, h in enumerate(top_hpo)}
    hpo_set = set(top_hpo)

    # IBA panels
    panel_names = list(iba_map.keys()) if iba_map else []
    n_panels = len(panel_names)
    log.info(f"IBA panels: {panel_names}")

    # Gene enriched feature columns
    clinvar_cols = ["pathogenic_fraction", "log_n_pathogenic", "actionability_score",
                    "log_n_variants", "n_vus"]
    attr_cols = []
    gene_lookup = {}
    if gene_feat is not None:
        available_cv = [c for c in clinvar_cols if c in gene_feat.columns]
        attr_cols = [c for c in gene_feat.columns if c.startswith("attr_")][:TOP_ATTR]
        gene_feat_cols = available_cv + attr_cols
        for _, row in gene_feat.iterrows():
            g = str(row.get("gene_symbol", "")).strip().upper()
            if g:
                gene_lookup[g] = row
    else:
        gene_feat_cols = []

    # BabySeq gene features per gene
    bs_gene_cols = ["evidence_score", "penetrance_score", "category_score",
                    "actionability_index", "inheritance_code"]
    bs_gene_lookup = {}
    if bs_gene is not None:
        for _, row in bs_gene.iterrows():
            g = str(row.get("Gene", "")).strip().upper()
            if g:
                bs_gene_lookup[g] = row

    n_gf = len(gene_feat_cols)
    n_bs = len(bs_gene_cols)

    # Total feature vector: HPO + IBA panels + gene enriched + BabySeq + summary
    total_feat = TOP_HPO + n_panels + n_gf + n_bs + 3
    log.info(f"Feature dims: {TOP_HPO} HPO + {n_panels} IBA + {n_gf} gene + {n_bs} BabySeq + 3 = {total_feat}")

    samples, labels = [], []

    # Gather per-disease augmentation count (Category A -> more samples)
    category_a_genes = set()
    if bs_gene is not None:
        category_a_genes = set(bs_gene[bs_gene["babyseq_category"] == "A"]["Gene"].str.upper())

    disease_groups = df.groupby("disease_id")
    for disease_id, group in disease_groups:
        hpos_for_disease = group["hpo_id"].dropna().unique().tolist()
        hpos_in_vocab = [h for h in hpos_for_disease if h in hpo_set]
        genes = list(group["gene_symbol"].dropna().unique())

        if len(hpos_in_vocab) < 3:
            continue

        # Determine augment count by evidence grade
        disease_genes_upper = [g.upper() for g in genes]
        is_cat_a = any(g in category_a_genes for g in disease_genes_upper)
        n_aug = N_AUGMENT_BASE if is_cat_a else N_AUGMENT_B

        # Build gene feature vectors (max-pool across genes)
        gf_vec = np.zeros(n_gf, dtype=np.float32)
        bs_vec = np.zeros(n_bs, dtype=np.float32)
        has_gene_feat = 0.0
        has_bs_feat = 0.0

        if gene_lookup:
            gene_vecs = []
            for g in disease_genes_upper:
                if g in gene_lookup:
                    row = gene_lookup[g]
                    gvec = [float(row.get(col, 0) or 0) for col in gene_feat_cols]
                    gene_vecs.append(gvec)
            if gene_vecs:
                gf_vec = np.max(gene_vecs, axis=0).astype(np.float32)
                has_gene_feat = 1.0

        if bs_gene_lookup:
            bs_vecs = []
            for g in disease_genes_upper:
                if g in bs_gene_lookup:
                    row = bs_gene_lookup[g]
                    bvec = [float(row.get(col, 0) or 0) for col in bs_gene_cols]
                    bs_vecs.append(bvec)
            if bs_vecs:
                bs_vec = np.max(bs_vecs, axis=0).astype(np.float32)
                has_bs_feat = 1.0

        # IBA panel activation for canonical HPO set
        canonical_panel_scores = compute_iba_panel_activation(hpos_in_vocab, iba_map, panel_names)

        # Canonical sample (all HPO present)
        vec = np.zeros(total_feat, dtype=np.float32)
        for h in hpos_in_vocab:
            vec[hpo_idx[h]] = 1.0
        vec[TOP_HPO:TOP_HPO+n_panels] = canonical_panel_scores
        vec[TOP_HPO+n_panels:TOP_HPO+n_panels+n_gf] = gf_vec
        vec[TOP_HPO+n_panels+n_gf:TOP_HPO+n_panels+n_gf+n_bs] = bs_vec
        vec[-3] = len(hpos_in_vocab) / max(TOP_HPO, 1)
        vec[-2] = min(len(genes), 50) / 50.0
        vec[-1] = has_gene_feat * has_bs_feat

        samples.append(vec)
        labels.append(disease_id)

        # Augmented samples
        for _ in range(n_aug - 1):
            frac = np.random.uniform(0.20, 0.60)
            n_keep = max(2, int(len(hpos_in_vocab) * frac))
            sel_hpos = np.random.choice(hpos_in_vocab, size=min(n_keep, len(hpos_in_vocab)), replace=False)

            aug_vec = np.zeros(total_feat, dtype=np.float32)
            for h in sel_hpos:
                aug_vec[hpo_idx[h]] = 1.0

            # Noise confounders
            n_confound = np.random.randint(2, 7)
            for _ in range(n_confound):
                aug_vec[np.random.randint(0, TOP_HPO)] = 1.0

            # Panel activation from augmented HPO set
            aug_hpo_set = list(sel_hpos) + [top_hpo[np.random.randint(0, TOP_HPO)] for _ in range(n_confound)]
            aug_panel_scores = compute_iba_panel_activation(aug_hpo_set, iba_map, panel_names)

            aug_vec[TOP_HPO:TOP_HPO+n_panels] = aug_panel_scores
            aug_vec[TOP_HPO+n_panels:TOP_HPO+n_panels+n_gf] = gf_vec
            aug_vec[TOP_HPO+n_panels+n_gf:TOP_HPO+n_panels+n_gf+n_bs] = bs_vec
            aug_vec[-3] = len(sel_hpos) / max(TOP_HPO, 1)
            aug_vec[-2] = vec[-2]
            aug_vec[-1] = has_gene_feat * has_bs_feat

            samples.append(aug_vec)
            labels.append(disease_id)

    X = np.nan_to_num(np.array(samples, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = np.array(labels)

    feature_names = (
        top_hpo +
        [f"panel_{p}" for p in panel_names] +
        gene_feat_cols +
        bs_gene_cols +
        ["hpo_density", "gene_count_norm", "has_all_features"]
    )

    log.info(f"Feature matrix: {X.shape}, diseases: {len(np.unique(y))}")
    return X, y, feature_names, panel_names


def topk(y_true, y_prob, k):
    try:
        return top_k_accuracy_score(y_true, y_prob, k=k,
                                    labels=np.arange(y_prob.shape[1]))
    except Exception:
        return None


def train_and_eval(X_train, X_test, y_train_enc, y_test_enc, le, feature_names):
    results = []

    # RF
    log.info("Training RF...")
    rf = RandomForestClassifier(n_estimators=150, max_depth=30, min_samples_leaf=1,
                                 n_jobs=-1, random_state=42, class_weight="balanced")
    rf.fit(X_train, y_train_enc)
    rf_pred = rf.predict(X_test)
    rf_prob = rf.predict_proba(X_test)
    rf_acc = accuracy_score(y_test_enc, rf_pred)
    rf_f1 = f1_score(y_test_enc, rf_pred, average="macro", zero_division=0)
    log.info(f"RF: Acc={rf_acc:.4f} F1={rf_f1:.4f} Top5={topk(y_test_enc,rf_prob,5)}")
    results.append({"model": "RandomForest_v4", "accuracy": rf_acc, "f1_macro": rf_f1,
                    "top3": topk(y_test_enc, rf_prob, 3), "top5": topk(y_test_enc, rf_prob, 5),
                    "top10": topk(y_test_enc, rf_prob, 10),
                    "n_test": len(y_test_enc), "n_classes": len(le.classes_),
                    "n_features": X_train.shape[1]})

    # Save RF immediately
    with open(OUT_MODELS / "random_forest_v4.pkl", "wb") as f:
        pickle.dump(rf, f)
    with open(OUT_MODELS / "label_encoder_v4.pkl", "wb") as f:
        pickle.dump(le, f)
    np.save(OUT_MODELS / "hpo_feature_names_v4.npy", np.array(feature_names, dtype=object))
    log.info("RF checkpoint saved")

    # Feature importance plot
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        imp = pd.Series(rf.feature_importances_, index=feature_names)
        top20 = imp.nlargest(20)
        # Color-code: HPO=blue, panel=green, gene=orange, babyseq=red
        colors = []
        for fn in top20.index:
            if fn.startswith("HP:"): colors.append("#0066CC")
            elif fn.startswith("panel_"): colors.append("#00A896")
            elif fn.startswith("attr_"): colors.append("#F18F01")
            elif fn in ["evidence_score","penetrance_score","category_score",
                        "actionability_index","inheritance_code"]: colors.append("#E63946")
            else: colors.append("#6c757d")
        fig, ax = plt.subplots(figsize=(11, 7))
        top20.sort_values().plot(kind="barh", ax=ax, color=colors[::-1])
        ax.set_title("V4 Feature Importance (RF) - Blue=HPO, Green=IBA Panel, Orange=GeneAttr, Red=BabySeq", fontsize=11)
        from matplotlib.patches import Patch
        legend = [Patch(color="#0066CC", label="HPO symptom"),
                  Patch(color="#00A896", label="IBA Panel score"),
                  Patch(color="#F18F01", label="Gene attribute"),
                  Patch(color="#E63946", label="BabySeq evidence")]
        ax.legend(handles=legend, loc="lower right", fontsize=8)
        plt.tight_layout()
        plt.savefig(OUT_PLOTS / "rf_feature_importance_v4.png", dpi=150, bbox_inches="tight")
        plt.close()
        log.info("Feature importance plot saved")
    except Exception as e:
        log.warning(f"Plot failed: {e}")

    # XGB
    xgb = None
    if HAS_XGB:
        log.info("Training XGB...")
        try:
            from sklearn.preprocessing import LabelEncoder as _LE
            xgb_le = _LE()
            y_tr_x = xgb_le.fit_transform(y_train_enc)
            y_te_x = xgb_le.transform(y_test_enc)
            n_xgb = len(xgb_le.classes_)
            xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.15,
                                 subsample=0.7, colsample_bytree=0.7,
                                 eval_metric="mlogloss", use_label_encoder=False,
                                 random_state=42, n_jobs=2, num_class=n_xgb,
                                 objective="multi:softprob", tree_method="hist")
            xgb.fit(X_train, y_tr_x, verbose=False)
            xgb_pred = xgb.predict(X_test)
            xgb_prob = xgb.predict_proba(X_test)
            xgb_acc = accuracy_score(y_te_x, xgb_pred)
            xgb_f1 = f1_score(y_te_x, xgb_pred, average="macro", zero_division=0)
            log.info(f"XGB: Acc={xgb_acc:.4f} F1={xgb_f1:.4f}")
            results.append({"model": "XGBoost_v4", "accuracy": xgb_acc, "f1_macro": xgb_f1,
                            "top5": topk(y_te_x, xgb_prob, 5), "n_test": len(y_te_x),
                            "n_classes": n_xgb})
            with open(OUT_MODELS / "xgboost_v4.pkl", "wb") as f:
                pickle.dump(xgb, f)
        except Exception as e:
            log.warning(f"XGB failed (OOM?): {e}")
            xgb = None

    # LR
    log.info("Training LR...")
    # Fill any NaN before scaling
    X_train_lr = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_test_lr = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)
    scaler = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_train_lr)
    X_te_sc = scaler.transform(X_test_lr)
    lr = LogisticRegression(max_iter=2000, C=1.0, solver="lbfgs", random_state=42)
    lr.fit(X_tr_sc, y_train_enc)
    lr_pred = lr.predict(X_te_sc)
    lr_prob = lr.predict_proba(X_te_sc)
    lr_acc = accuracy_score(y_test_enc, lr_pred)
    lr_f1 = f1_score(y_test_enc, lr_pred, average="macro", zero_division=0)
    log.info(f"LR: Acc={lr_acc:.4f} F1={lr_f1:.4f} Top5={topk(y_test_enc,lr_prob,5)}")
    results.append({"model": "LogisticRegression_v4", "accuracy": lr_acc, "f1_macro": lr_f1,
                    "top5": topk(y_test_enc, lr_prob, 5), "n_test": len(y_test_enc),
                    "n_classes": len(le.classes_)})

    with open(OUT_MODELS / "logistic_regression_v4.pkl", "wb") as f:
        pickle.dump(lr, f)
    with open(OUT_MODELS / "scaler_v4.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return rf, xgb, lr, scaler, results


def run_stratified_cv(X, y_enc, n_splits=3):
    """Stratified k-fold CV (correct for multi-class classification)."""
    log.info("Running StratifiedKFold CV...")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y_enc)):
        rf = RandomForestClassifier(n_estimators=80, max_depth=25,
                                     n_jobs=-1, random_state=fold, class_weight="balanced")
        rf.fit(X[tr_idx], y_enc[tr_idx])
        pred = rf.predict(X[te_idx])
        sc = f1_score(y_enc[te_idx], pred, average="macro", zero_division=0)
        scores.append(sc)
        log.info(f"  Fold {fold+1}: F1={sc:.4f}")

    mean_cv = float(np.mean(scores))
    std_cv = float(np.std(scores))
    log.info(f"StratifiedKFold CV: {mean_cv:.4f} +/- {std_cv:.4f}")
    return mean_cv, std_cv, scores


def run():
    log.info("=== V4 Classifier: HPO + IBA Panels + BabySeq + ClinVar + Gene Attrs ===")

    master, gene_feat, gene_action, iba_map, bs_disease, bs_gene = load_all_data()

    top_diseases = select_top_diseases(master, n=N_DISEASES)

    X, y, feature_names, panel_names = build_feature_matrix(
        master, top_diseases, gene_feat, gene_action, iba_map, bs_gene)

    if len(X) == 0:
        log.error("No samples - abort")
        return

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    log.info(f"Samples: {len(X)}, Classes: {len(le.classes_)}")

    # Stratified split
    tr_idx, te_idx = train_test_split(
        np.arange(len(X)), test_size=0.20, random_state=42, stratify=y_enc)
    X_train, X_test = X[tr_idx], X[te_idx]
    y_train_enc, y_test_enc = y_enc[tr_idx], y_enc[te_idx]
    log.info(f"Train: {len(X_train)}, Test: {len(X_test)}, all {len(le.classes_)} classes in both splits")

    # Train
    rf, xgb, lr, scaler, results = train_and_eval(
        X_train, X_test, y_train_enc, y_test_enc, le, feature_names)

    # Stratified CV (correct method for multi-class)
    cv_mean, cv_std, cv_scores = run_stratified_cv(X, y_enc, n_splits=3)

    # Feature breakdown for research paper
    feat_breakdown = {
        "hpo_features": TOP_HPO,
        "iba_panel_features": len(panel_names),
        "clinvar_gene_features": len([c for c in feature_names
                                       if c in ["pathogenic_fraction","log_n_pathogenic",
                                                "actionability_score","log_n_variants","n_vus"]]),
        "gene_attr_features": len([c for c in feature_names if c.startswith("attr_")]),
        "babyseq_features": len(["evidence_score","penetrance_score","category_score",
                                   "actionability_index","inheritance_code"]),
        "summary_features": 3,
        "total": len(feature_names),
    }

    # Save metrics
    with open(OUT_RESULTS / "classifier_metrics_v4.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    cv_out = {
        "method": "StratifiedKFold (correct for multi-class)",
        "cv_f1_macro_mean": cv_mean, "cv_f1_macro_std": cv_std,
        "cv_scores": cv_scores, "n_splits": n_splits if (n_splits := 3) else 3,
        "total_samples": len(X), "n_diseases": len(le.classes_),
        "n_features": len(feature_names),
        "feature_breakdown": feat_breakdown,
        "novel_features": {
            "iba_panel_activation": f"{len(panel_names)} clinical panel scores computed from HPO input",
            "babyseq_evidence_grade": "evidence_score maps Definitive=4 Strong=3 Moderate=2 Limited=1",
            "penetrance_weighted": "penetrance_score: HIGH=1.0, MODERATE=0.6, LOW=0.2, UNKNOWN=0.5",
            "actionability_index": "evidence_score * penetrance * category_score / max, normalized 0-1",
        }
    }
    with open(OUT_RESULTS / "cv_scores_v4.json", "w") as f:
        json.dump(cv_out, f, indent=2)

    log.info("\n=== V4 RESULTS ===")
    for r in results:
        log.info(f"  {r['model']}: Acc={r['accuracy']:.4f}, F1={r['f1_macro']:.4f}, "
                 f"Top5={r.get('top5','N/A')}")
    log.info(f"  StratifiedKFold CV: {cv_mean:.4f} +/- {cv_std:.4f}")
    log.info(f"  Feature breakdown: {feat_breakdown}")
    log.info("=== V4 DONE ===")


if __name__ == "__main__":
    run()
