"""
Disease Classifier — Module 1
Input: multi-hot HPO phenotype vector
Output: top-5 disease predictions + SHAP explanations

Models: Random Forest + XGBoost ensemble (VotingClassifier)
Handles: class imbalance, evaluation with macro-F1, saves model + SHAP values
"""

import pandas as pd
import numpy as np
import joblib
import shap
import logging
import json
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                             f1_score, accuracy_score, top_k_accuracy_score)
from sklearn.preprocessing import LabelEncoder
from sklearn.calibration import CalibratedClassifierCV
from sklearn.utils.class_weight import compute_sample_weight
import xgboost as xgb
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "outputs" / "models"
PLOTS = BASE / "outputs" / "plots"
RESULTS = BASE / "outputs" / "results"

for d in [MODELS, PLOTS, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)


def load_data():
    matrix_path = PROC / "disease_phenotype_matrix.csv"
    label_path = PROC / "disease_label_map.csv"

    if not matrix_path.exists():
        raise FileNotFoundError(f"Matrix not found: {matrix_path}. Run merge_datasets.py first.")

    log.info("Loading disease-phenotype matrix...")
    matrix = pd.read_csv(matrix_path, index_col=0)
    label_map = pd.read_csv(label_path)

    log.info(f"Matrix: {matrix.shape} ({matrix.index.nunique()} diseases x {len(matrix.columns)} HPO terms)")

    X = matrix.values.astype(np.float32)
    disease_ids = matrix.index.tolist()

    le = LabelEncoder()
    y = le.fit_transform(disease_ids)

    # Save label encoder
    joblib.dump(le, MODELS / "label_encoder.pkl")
    np.save(MODELS / "hpo_feature_names.npy", np.array(matrix.columns.tolist()))

    return X, y, le, matrix.columns.tolist(), disease_ids


def get_class_weights(y):
    """Compute sample weights for imbalanced classes."""
    return compute_sample_weight("balanced", y)


def train_models(X_train, y_train, sample_weights):
    """Train RF + XGBoost. Return individual + ensemble."""

    n_classes = len(np.unique(y_train))
    log.info(f"Training on {X_train.shape[0]} samples, {n_classes} classes...")

    # Random Forest
    log.info("Training Random Forest...")
    rf = RandomForestClassifier(
        n_estimators=300,
        max_depth=20,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
        verbose=0,
    )
    rf.fit(X_train, y_train, sample_weight=sample_weights)
    log.info("RF done")

    # XGBoost
    log.info("Training XGBoost...")
    scale_pos = (y_train == 0).sum() / max((y_train != 0).sum(), 1)
    xgb_clf = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0,
        tree_method="hist",
    )
    xgb_clf.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)
    log.info("XGBoost done")

    return rf, xgb_clf


def evaluate(clf, X_test, y_test, le, name="Model"):
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test) if hasattr(clf, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    top5_acc = None
    if y_proba is not None:
        try:
            top5_acc = top_k_accuracy_score(y_test, y_proba, k=5)
        except Exception:
            pass

    log.info(f"\n{'='*50}")
    log.info(f"{name} Results:")
    log.info(f"  Accuracy:      {acc:.4f}")
    log.info(f"  F1 (macro):    {f1_macro:.4f}")
    log.info(f"  F1 (weighted): {f1_weighted:.4f}")
    if top5_acc:
        log.info(f"  Top-5 Acc:     {top5_acc:.4f}")
    log.info(f"{'='*50}")

    results = {
        "model": name,
        "accuracy": float(acc),
        "f1_macro": float(f1_macro),
        "f1_weighted": float(f1_weighted),
        "top5_accuracy": float(top5_acc) if top5_acc else None,
        "n_test": int(len(y_test)),
        "n_classes": int(len(np.unique(y_test))),
    }
    return results, y_pred, y_proba


def compute_shap_values(rf, X_train, X_test, feature_names, n_samples=200):
    """Compute SHAP values for RF model on a sample."""
    log.info("Computing SHAP values (TreeExplainer)...")
    try:
        X_sample = X_test[:n_samples]
        explainer = shap.TreeExplainer(rf)
        shap_values = explainer.shap_values(X_sample)
        # shap_values is list of arrays (one per class) or 3D array
        if isinstance(shap_values, list):
            # Average absolute SHAP across all classes
            mean_abs_shap = np.mean([np.abs(sv) for sv in shap_values], axis=0)
        else:
            mean_abs_shap = np.abs(shap_values).mean(axis=(0, 2)) if shap_values.ndim == 3 else np.abs(shap_values).mean(axis=0)

        # Feature importance from SHAP
        shap_importance = pd.DataFrame({
            "feature": feature_names,
            "mean_abs_shap": mean_abs_shap.mean(axis=0) if mean_abs_shap.ndim > 1 else mean_abs_shap
        }).sort_values("mean_abs_shap", ascending=False)

        shap_importance.to_csv(RESULTS / "shap_feature_importance.csv", index=False)
        log.info("SHAP values computed and saved")
        return shap_importance, explainer, shap_values
    except Exception as e:
        log.warning(f"SHAP computation failed: {e}")
        return None, None, None


def plot_feature_importance(rf, feature_names, top_n=30):
    """Plot top N feature importances."""
    importances = rf.feature_importances_
    indices = np.argsort(importances)[::-1][:top_n]

    fig, ax = plt.subplots(figsize=(12, 8))
    ax.bar(range(top_n), importances[indices])
    ax.set_xticks(range(top_n))
    ax.set_xticklabels([feature_names[i] for i in indices], rotation=90, fontsize=7)
    ax.set_title(f"Top {top_n} Most Predictive HPO Features (Random Forest)", fontsize=14)
    ax.set_xlabel("HPO Term")
    ax.set_ylabel("Feature Importance")
    plt.tight_layout()
    plt.savefig(PLOTS / "rf_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Feature importance plot saved")


def plot_confusion_matrix_top(y_test, y_pred, le, top_n=20):
    """Plot confusion matrix for top N most frequent classes."""
    from collections import Counter
    top_classes = [c for c, _ in Counter(y_test).most_common(top_n)]
    mask = np.isin(y_test, top_classes)
    yt = y_test[mask]
    yp = y_pred[mask]

    labels = sorted(set(yt))
    cm = confusion_matrix(yt, yp, labels=labels)

    fig, ax = plt.subplots(figsize=(14, 12))
    disease_labels = [le.classes_[i][:20] for i in labels]
    sns.heatmap(cm, xticklabels=disease_labels, yticklabels=disease_labels,
                fmt="d", cmap="Blues", ax=ax, annot=len(labels) <= 15)
    ax.set_title(f"Confusion Matrix (Top {top_n} Diseases)", fontsize=13)
    ax.set_xlabel("Predicted Disease")
    ax.set_ylabel("True Disease")
    plt.xticks(rotation=45, ha="right", fontsize=7)
    plt.yticks(rotation=0, fontsize=7)
    plt.tight_layout()
    plt.savefig(PLOTS / "confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Confusion matrix plot saved")


def predict_disease(symptom_hpo_ids, rf, xgb_clf, le, feature_names, top_k=5):
    """
    Predict disease from list of HPO IDs.
    Returns top-k predictions with probabilities.
    """
    # Build feature vector
    x = np.zeros(len(feature_names), dtype=np.float32)
    for hpo in symptom_hpo_ids:
        if hpo in feature_names:
            x[feature_names.index(hpo)] = 1.0

    if x.sum() == 0:
        return [{"disease": "No matching HPO terms found", "probability": 0.0, "model": "N/A"}]

    x = x.reshape(1, -1)

    results = []
    for model, name in [(rf, "RandomForest"), (xgb_clf, "XGBoost")]:
        proba = model.predict_proba(x)[0]
        top_idx = np.argsort(proba)[::-1][:top_k]
        for rank, idx in enumerate(top_idx):
            results.append({
                "rank": rank + 1,
                "disease_id": le.classes_[idx],
                "probability": float(proba[idx]),
                "model": name,
            })

    return results


def run_training():
    log.info("=== Disease Classifier Training ===")
    X, y, le, feature_names, disease_ids = load_data()

    log.info(f"Dataset: {X.shape[0]} samples, {X.shape[1]} features, {len(np.unique(y))} classes")

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    sample_weights = get_class_weights(y_train)

    # Train both models
    rf, xgb_clf = train_models(X_train, y_train, sample_weights)

    # Evaluate
    all_results = []
    rf_results, rf_pred, rf_proba = evaluate(rf, X_test, y_test, le, "RandomForest")
    all_results.append(rf_results)
    xgb_results, xgb_pred, xgb_proba = evaluate(xgb_clf, X_test, y_test, le, "XGBoost")
    all_results.append(xgb_results)

    # Ensemble (average probabilities)
    if rf_proba is not None and xgb_proba is not None:
        ensemble_proba = (rf_proba + xgb_proba) / 2
        ensemble_pred = np.argmax(ensemble_proba, axis=1)
        acc = accuracy_score(y_test, ensemble_pred)
        f1_macro = f1_score(y_test, ensemble_pred, average="macro", zero_division=0)
        try:
            top5 = top_k_accuracy_score(y_test, ensemble_proba, k=5)
        except Exception:
            top5 = None
        top5_str = f"{top5:.4f}" if top5 else "N/A"
        log.info(f"\nEnsemble: Acc={acc:.4f}, F1={f1_macro:.4f}, Top5={top5_str}")
        all_results.append({
            "model": "Ensemble_RF_XGB",
            "accuracy": float(acc),
            "f1_macro": float(f1_macro),
            "top5_accuracy": float(top5) if top5 else None,
        })
        joblib.dump({"rf": rf, "xgb": xgb_clf, "weights": [0.5, 0.5]},
                    MODELS / "ensemble_model.pkl")

    # Save individual models
    joblib.dump(rf, MODELS / "random_forest.pkl")
    joblib.dump(xgb_clf, MODELS / "xgboost.pkl")
    log.info("Models saved")

    # Save results
    with open(RESULTS / "classifier_metrics.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Plots
    plot_feature_importance(rf, feature_names)
    plot_confusion_matrix_top(y_test, rf_pred, le)

    # SHAP
    shap_imp, explainer, shap_vals = compute_shap_values(rf, X_train, X_test, feature_names)
    if explainer is not None:
        joblib.dump(explainer, MODELS / "shap_explainer.pkl")
        if shap_vals is not None:
            try:
                sv = shap_vals[0] if isinstance(shap_vals, list) else shap_vals
                np.save(MODELS / "shap_values_sample.npy", np.array(sv))
            except Exception as e:
                log.warning(f"Could not save SHAP values: {e}")

    # Cross-validation on RF (3-fold for speed)
    log.info("Running 3-fold cross-validation...")
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    cv_scores = cross_val_score(rf, X, y, cv=cv, scoring="f1_macro", n_jobs=-1)
    log.info(f"CV F1-macro: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    with open(RESULTS / "cv_scores.json", "w") as f:
        json.dump({"cv_f1_macro_mean": float(cv_scores.mean()),
                   "cv_f1_macro_std": float(cv_scores.std()),
                   "cv_scores": cv_scores.tolist()}, f, indent=2)

    log.info("=== Classifier training complete ===")
    return rf, xgb_clf, le, feature_names


if __name__ == "__main__":
    rf, xgb_clf, le, feature_names = run_training()
    log.info("All models saved to outputs/models/")
