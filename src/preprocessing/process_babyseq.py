# -*- coding: utf-8 -*-
"""
BabySeq + Newborn Data Processor - Day 6
Sources:
  NIHMS856987-supplement-Table_S1.xlsx  -- 1,515 curated gene-disease pairs
    Fields: Gene, Disease, Evidence (Definitive/Strong/Moderate/Limited),
            Inheritance (AR/AD/XLR), Penetrance, BabySeq Category (A/B/C),
            IBA Panel (SEIZ/HYPOTO/IEM/DERM/...), Age<18

  mmc1.xlsx  -- 20 newborn pathogenic/likely-path variants
  mmc2.xlsx  -- 256 newborn variants with penetrance data

Outputs (all in data/processed/):
  babyseq_gene_disease.csv     -- clean gene-disease table with numeric features
  babyseq_category_a.csv       -- Category A only (highly penetrant pediatric)
  newborn_variants.csv         -- combined mmc1+mmc2 newborn variants
  iba_panel_hpo_map.json        -- IBA panel -> HPO term cluster mapping
  gene_actionability.csv        -- per-gene actionability index
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
RAW = BASE
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)

# ── Evidence grade mapping ──────────────────────────────────────────────────
EVIDENCE_SCORE = {
    "Definitive": 4, "Strong": 3, "Moderate": 2,
    "Limited": 1, "Conflicting": 0,
}

# ── Penetrance mapping ──────────────────────────────────────────────────────
PENETRANCE_SCORE = {
    "HIGH (A)": 1.0, "HIGH (B)": 0.85, "MODERATE (A)": 0.60,
    "MODERATE (B)": 0.40, "LOW (A)": 0.20, "LOW (B)": 0.15,
    "UNKNOWN": 0.50,
}

# ── Inheritance encoding ────────────────────────────────────────────────────
INHERITANCE_ENC = {
    "AR": 0, "AD": 1, "XLR": 2, "XLD": 3,
    "MITOCHONDRIAL": 4, "COMPLEX": 5, "IMP": 6,
    "AR-DIGENIC": 0, "UNKNOWN": -1,
}

# ── BabySeq Category score ──────────────────────────────────────────────────
CATEGORY_SCORE = {"A": 3, "B": 2, "C": 1}

# ── IBA Panel -> HPO term clusters (novel feature) ──────────────────────────
# Each panel maps to characteristic HPO terms. Used to compute panel activation.
IBA_PANEL_HPO = {
    "SEIZ": [
        "HP:0001250",  # Seizures
        "HP:0001251",  # Ataxia
        "HP:0002353",  # Febrile seizures
        "HP:0007359",  # Focal seizures
        "HP:0010851",  # EEG with burst suppression
        "HP:0002197",  # Generalized seizures
        "HP:0012469",  # Infantile spasms
    ],
    "HYPOTO": [
        "HP:0001252",  # Muscular hypotonia
        "HP:0001290",  # Generalized hypotonia
        "HP:0003326",  # Myalgia
        "HP:0001276",  # Hypertonia
        "HP:0002380",  # Fasciculations
        "HP:0003236",  # Elevated CK
        "HP:0003198",  # Myopathy
    ],
    "DERM": [
        "HP:0000951",  # Abnormal skin
        "HP:0001574",  # Abnormal integument
        "HP:0001596",  # Alopecia
        "HP:0008404",  # Nail dystrophy
        "HP:0100588",  # Abnormal blistering
        "HP:0000958",  # Dry skin
    ],
    "HL": [
        "HP:0000407",  # Sensorineural hearing loss
        "HP:0000365",  # Hearing impairment
        "HP:0001319",  # Neonatal hearing loss
        "HP:0001751",  # Vestibular dysfunction
    ],
    "CHD": [
        "HP:0001631",  # Atrial septal defect
        "HP:0001629",  # Ventricular septal defect
        "HP:0001636",  # Tetralogy of Fallot
        "HP:0001627",  # Abnormal heart morphology
        "HP:0004383",  # Hypoplastic left heart
    ],
    "CM": [
        "HP:0001644",  # Dilated cardiomyopathy
        "HP:0001639",  # Hypertrophic cardiomyopathy
        "HP:0001671",  # Abnormal cardiac septum
    ],
    "IEM": [
        "HP:0001985",  # Hypoglycemia
        "HP:0002150",  # Hyperuricemia
        "HP:0000739",  # Anxiety
        "HP:0001987",  # Hyperammonemia
        "HP:0003355",  # Amino aciduria
        "HP:0010985",  # Elevated plasma amino acids
        "HP:0001992",  # Organic aciduria
    ],
    "REN": [
        "HP:0000077",  # Kidney abnormality
        "HP:0000093",  # Proteinuria
        "HP:0000124",  # Renal tubular dysfunction
        "HP:0001917",  # Renal amyloidosis
        "HP:0000822",  # Hypertension
    ],
    "PULM": [
        "HP:0002093",  # Respiratory insufficiency
        "HP:0002090",  # Pneumonia
        "HP:0000765",  # Abnormal thorax
        "HP:0002878",  # Respiratory failure
        "HP:0006530",  # Interstitial lung disease
    ],
    "AN_TH": [
        "HP:0001903",  # Anemia
        "HP:0001897",  # Normocytic anemia
        "HP:0001896",  # Neutropenia
        "HP:0011991",  # Abnormal platelet morphology
        "HP:0005502",  # Increased HbF
    ],
    "SK": [
        "HP:0002652",  # Skeletal dysplasia
        "HP:0000924",  # Abnormal skeletal morphology
        "HP:0002650",  # Scoliosis
        "HP:0003510",  # Short stature
        "HP:0002857",  # Genu valgum
    ],
    "COND": [
        "HP:0001382",  # Joint hypermobility
        "HP:0001065",  # Striae distensae
        "HP:0000023",  # Inguinal hernia
        "HP:0002616",  # Aortic regurgitation
        "HP:0004933",  # Ascending aorta aneurysm
    ],
    "THYR": [
        "HP:0000820",  # Thyroid abnormality
        "HP:0000821",  # Hypothyroidism
        "HP:0000822",  # Hypertension (secondary)
        "HP:0001508",  # Failure to thrive
    ],
}


def parse_iba_panels(panel_str):
    """Parse 'SEIZ, HYPOTO' -> ['SEIZ', 'HYPOTO']"""
    if pd.isna(panel_str) or str(panel_str).strip() == "":
        return []
    return [p.strip() for p in str(panel_str).split(",") if p.strip()]


def process_table_s1():
    path = RAW / "NIHMS856987-supplement-Table_S1.xlsx"
    if not path.exists():
        log.warning("Table S1 not found at project root")
        return None

    df = pd.read_excel(path, header=1)
    log.info(f"Table S1 loaded: {df.shape}")

    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns={
        "Typical inheritance ": "Typical inheritance",
        "Age of onset <18 Yrs": "pediatric_onset",
        "BabySeq Category": "babyseq_category",
        "BabySeq IBA panel": "iba_panel",
        "Evidence for gene-disease association": "evidence_level",
        "Curated disease": "disease_name",
        "Meets NGSR citeria?": "meets_ngsr",
        "Key references used in curation (PubMed ID)": "pubmed_ids",
    })

    # Numeric features
    df["evidence_score"] = df["evidence_level"].map(EVIDENCE_SCORE).fillna(0)
    df["penetrance_score"] = df["Penetrance"].map(PENETRANCE_SCORE).fillna(0.5)
    df["category_score"] = df["babyseq_category"].map(CATEGORY_SCORE).fillna(0)
    df["inheritance_code"] = df["Typical inheritance"].map(INHERITANCE_ENC).fillna(-1)
    df["is_pediatric"] = (df["pediatric_onset"].astype(str).str.strip().str.lower() == "yes").astype(int)
    df["is_category_a"] = (df["babyseq_category"].astype(str).str.strip() == "A").astype(int)
    df["meets_ngsr"] = (df["meets_ngsr"].astype(str).str.strip().str.lower() == "yes").astype(int)

    # Parse IBA panels -> one-hot
    all_panels = list(IBA_PANEL_HPO.keys())
    for panel in all_panels:
        df[f"panel_{panel}"] = df["iba_panel"].apply(
            lambda x: 1 if panel in parse_iba_panels(x) else 0)

    # Actionability = evidence_score * penetrance_score * category_score (normalized)
    df["actionability_index"] = (
        df["evidence_score"] * df["penetrance_score"] * df["category_score"]
    ) / (4 * 1.0 * 3)  # normalize to 0-1

    out = PROC / "babyseq_gene_disease.csv"
    df.to_csv(out, index=False)
    log.info(f"Saved babyseq_gene_disease.csv: {df.shape}")

    # Category A subset
    cat_a = df[df["babyseq_category"] == "A"].copy()
    cat_a.to_csv(PROC / "babyseq_category_a.csv", index=False)
    log.info(f"Category A: {len(cat_a)} genes")

    return df


def process_newborn_variants():
    """Combine mmc1 + mmc2 into newborn_variants.csv"""
    frames = []

    mmc1_path = RAW / "mmc1.xlsx"
    if mmc1_path.exists():
        df1 = pd.read_excel(mmc1_path)
        df1.columns = [c.strip() for c in df1.columns]
        # Direct rename using known columns
        df1 = df1.rename(columns={
            "Gene (Transcript)": "gene_symbol", "Gene": "gene_symbol",
            "Variant": "variant", "Classification": "classification",
            "Disease": "disease_name",
        })
        df1["source"] = "mmc1_babyseq_ill"
        df1["penetrance"] = "High"
        keep1 = [c for c in ["gene_symbol", "variant", "classification", "disease_name",
                              "source", "penetrance"] if c in df1.columns]
        out1 = pd.DataFrame({k: df1[k].values for k in keep1})
        frames.append(out1)
        log.info(f"mmc1 loaded: {df1.shape}")

    mmc2_path = RAW / "mmc2.xlsx"
    if mmc2_path.exists():
        df2 = pd.read_excel(mmc2_path, sheet_name="Sheet1")
        df2.columns = [c.strip() for c in df2.columns]
        rename2 = {}
        for c in df2.columns:
            if c.lower() == "gene": rename2[c] = "gene_symbol"
            if "variant" in c.lower(): rename2[c] = "variant"
            if "classif" in c.lower(): rename2[c] = "classification"
            if c.lower() == "disease": rename2[c] = "disease_name"
            if "penetrance" in c.lower(): rename2[c] = "penetrance"
        df2 = df2.rename(columns={
            "Gene": "gene_symbol", "Variant": "variant",
            "Disease": "disease_name", "Classification": "classification",
            "Penetrance": "penetrance",
        })
        df2["source"] = "mmc2_babyseq_variants"
        keep = [c for c in ["gene_symbol", "variant", "classification", "disease_name",
                             "source", "penetrance"] if c in df2.columns]
        out2 = pd.DataFrame({k: df2[k].values for k in keep})
        frames.append(out2)
        log.info(f"mmc2 loaded: {df2.shape}")

    if not frames:
        log.warning("No newborn variant files found")
        return None

    combined = pd.concat(frames, ignore_index=True)
    combined.to_csv(PROC / "newborn_variants.csv", index=False)
    log.info(f"newborn_variants.csv: {combined.shape}")
    return combined


def build_gene_actionability(babyseq_df):
    """Per-gene actionability index combining BabySeq + ClinVar."""
    if babyseq_df is None:
        return None

    # Aggregate per gene (some genes appear for multiple diseases)
    gene_agg = babyseq_df.groupby("Gene").agg(
        n_diseases=("disease_name", "nunique"),
        max_evidence_score=("evidence_score", "max"),
        max_penetrance_score=("penetrance_score", "max"),
        max_category_score=("category_score", "max"),
        max_actionability=("actionability_index", "max"),
        mean_actionability=("actionability_index", "mean"),
        is_pediatric=("is_pediatric", "max"),
        is_category_a=("is_category_a", "max"),
        meets_ngsr=("meets_ngsr", "max"),
        inheritance_code=("inheritance_code", "first"),
        panels=("iba_panel", lambda x: "|".join([str(v) for v in x.dropna().unique()])),
    ).reset_index()
    gene_agg = gene_agg.rename(columns={"Gene": "gene_symbol"})

    # Merge ClinVar if available
    clinvar_path = PROC / "clinvar_gene_summary.csv"
    if clinvar_path.exists():
        clinvar = pd.read_csv(clinvar_path)
        clinvar_cols = ["gene_symbol"]
        for col in ["pathogenic_fraction", "log_n_pathogenic", "actionability_score",
                    "clinical_actionability", "n_pathogenic"]:
            if col in clinvar.columns:
                clinvar_cols.append(col)
        gene_agg = gene_agg.merge(clinvar[clinvar_cols], on="gene_symbol", how="left")
        pf_col = "pathogenic_fraction" if "pathogenic_fraction" in gene_agg.columns else None
        gene_agg["combined_actionability"] = (
            gene_agg["max_actionability"] * 0.6 +
            (gene_agg[pf_col].fillna(0) if pf_col else 0) * 0.4
        )
    else:
        gene_agg["combined_actionability"] = gene_agg["max_actionability"]

    gene_agg.to_csv(PROC / "gene_actionability.csv", index=False)
    log.info(f"gene_actionability.csv: {gene_agg.shape}")
    return gene_agg


def save_iba_panel_hpo_map():
    """Save IBA panel -> HPO mapping as JSON."""
    out = PROC / "iba_panel_hpo_map.json"
    with open(out, "w") as f:
        json.dump(IBA_PANEL_HPO, f, indent=2)
    log.info(f"Saved iba_panel_hpo_map.json ({len(IBA_PANEL_HPO)} panels)")

    # Also save HPO -> panels reverse map
    hpo_to_panels = {}
    for panel, hpos in IBA_PANEL_HPO.items():
        for hpo in hpos:
            hpo_to_panels.setdefault(hpo, []).append(panel)
    with open(PROC / "hpo_to_iba_panel.json", "w") as f:
        json.dump(hpo_to_panels, f, indent=2)
    log.info(f"Saved hpo_to_iba_panel.json ({len(hpo_to_panels)} HPO terms mapped)")


def build_newborn_disease_features(babyseq_df):
    """
    Per-disease features from BabySeq (some diseases appear for multiple genes).
    Maps disease_name to aggregated features.
    """
    if babyseq_df is None:
        return None

    # Match diseases by name to our processed disease catalog
    disease_agg = babyseq_df.groupby("disease_name").agg(
        n_genes=("Gene", "nunique"),
        max_evidence_score=("evidence_score", "max"),
        max_penetrance_score=("penetrance_score", "max"),
        max_category_score=("category_score", "max"),
        max_actionability=("actionability_index", "max"),
        is_pediatric=("is_pediatric", "max"),
        is_category_a=("is_category_a", "max"),
        meets_ngsr=("meets_ngsr", "max"),
    ).reset_index()

    # Add panel one-hots
    panel_cols = [c for c in babyseq_df.columns if c.startswith("panel_")]
    for col in panel_cols:
        disease_agg = disease_agg.join(
            babyseq_df.groupby("disease_name")[col].max(),
            on="disease_name"
        )

    disease_agg.to_csv(PROC / "babyseq_disease_features.csv", index=False)
    log.info(f"babyseq_disease_features.csv: {disease_agg.shape}")
    return disease_agg


def run():
    log.info("=== BabySeq Data Processor - Day 6 ===")

    # 1. Process Table S1
    babyseq_df = process_table_s1()

    # 2. Process newborn variants
    newborn_df = process_newborn_variants()

    # 3. Gene actionability index
    gene_action = build_gene_actionability(babyseq_df)

    # 4. Disease-level features from BabySeq
    disease_feat = build_newborn_disease_features(babyseq_df)

    # 5. IBA panel HPO mapping
    save_iba_panel_hpo_map()

    log.info("=== BabySeq Processing Complete ===")
    if babyseq_df is not None:
        log.info(f"  Category A genes: {(babyseq_df['babyseq_category']=='A').sum()}")
        log.info(f"  Pediatric onset: {babyseq_df['is_pediatric'].sum()}")
        log.info(f"  Meets NGSR: {babyseq_df['meets_ngsr'].sum()}")
    if gene_action is not None:
        log.info(f"  Gene actionability table: {gene_action.shape}")


if __name__ == "__main__":
    run()
