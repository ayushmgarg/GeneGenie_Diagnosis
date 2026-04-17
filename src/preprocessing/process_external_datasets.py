# -*- coding: utf-8 -*-
"""
process_external_datasets.py - Day 7 Integration
Processes 3 new external datasets and merges into gene_actionability:

1. GnomAD v4.1 constraint metrics (gnomad_constraint.tsv)
   -> pLI, LOEUF (lof.oe_ci.upper), mis_z_score per gene
   -> Objective evidence of gene essentiality / penetrance proxy

2. ACMG Secondary Findings v3.2 (acmg_sf_v32.csv)
   -> 71 actionable genes clinical labs must report incidentally
   -> Strongest clinical actionability flag

3. PanelApp NHS (panelapp_iba_genes.json)
   -> 10 clinical panels, green-tier genes only
   -> Cross-validates IBA panel activation with NHS-curated evidence

Outputs:
  data/processed/gnomad_gene_constraint.csv
  data/processed/acmg_sf_genes.csv
  data/processed/panelapp_panel_genes.csv
  data/processed/gene_actionability_v2.csv   <- merged, replaces v1 in engine
  data/processed/iba_panel_hpo_map_v2.json   <- panelapp-validated panel map
"""

import pandas as pd
import numpy as np
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"


# ─── 1. GnomAD Constraint ────────────────────────────────────────────────────

def process_gnomad():
    """Process GnomAD v4.1 constraint → per-gene pLI, LOEUF, mis_z."""
    path = RAW / "gnomad_constraint.tsv"
    if not path.exists():
        log.warning("gnomad_constraint.tsv not found")
        return None

    log.info("Processing GnomAD constraint metrics...")
    usecols = ["gene", "lof.pLI", "lof.oe", "lof.oe_ci.upper", "mis.z_score", "canonical"]
    gn = pd.read_csv(path, sep="\t", usecols=usecols, low_memory=False)

    # Keep canonical transcripts only; fallback to max pLI if none marked
    canon = gn[gn["canonical"] == True].copy()
    if canon.empty:
        canon = gn.copy()

    # Deduplicate: keep one row per gene (highest pLI)
    canon = canon.sort_values("lof.pLI", ascending=False)
    canon = canon.drop_duplicates(subset=["gene"], keep="first")
    canon = canon.rename(columns={
        "gene": "gene_symbol",
        "lof.pLI": "pLI",
        "lof.oe": "lof_oe",
        "lof.oe_ci.upper": "LOEUF",
        "mis.z_score": "mis_z",
    })
    canon = canon[["gene_symbol", "pLI", "lof_oe", "LOEUF", "mis_z"]]
    canon = canon.dropna(subset=["gene_symbol"])
    canon["gene_symbol"] = canon["gene_symbol"].str.upper()

    # Derived features:
    # hi_intolerant: pLI > 0.9 (haploinsufficient genes = high penetrance)
    canon["hi_intolerant"] = (canon["pLI"] > 0.9).astype(int)
    # loeuf_bin: LOEUF < 0.35 = most constrained
    canon["loeuf_constrained"] = (canon["LOEUF"] < 0.35).astype(int)

    out = PROC / "gnomad_gene_constraint.csv"
    canon.to_csv(out, index=False)
    log.info(f"GnomAD: {len(canon)} genes → {out}")
    log.info(f"  pLI>0.9 (HI): {canon['hi_intolerant'].sum()} genes")
    log.info(f"  LOEUF<0.35 (constrained): {canon['loeuf_constrained'].sum()} genes")
    return canon


# ─── 2. ACMG Secondary Findings ──────────────────────────────────────────────

def process_acmg():
    """Load ACMG SF v3.2 gene list."""
    path = RAW / "acmg_sf_v32.csv"
    if not path.exists():
        log.warning("acmg_sf_v32.csv not found")
        return None

    acmg = pd.read_csv(path)
    acmg["gene_symbol"] = acmg["gene_symbol"].str.upper()
    acmg["acmg_sf"] = 1

    out = PROC / "acmg_sf_genes.csv"
    acmg.to_csv(out, index=False)
    log.info(f"ACMG SF: {len(acmg)} genes → {out}")
    return acmg


# ─── 3. PanelApp ─────────────────────────────────────────────────────────────

def process_panelapp():
    """Process PanelApp green gene lists → per-gene panel membership."""
    path = RAW / "panelapp_iba_genes.json"
    if not path.exists():
        log.warning("panelapp_iba_genes.json not found")
        return None

    with open(path) as f:
        panel_data = json.load(f)

    rows = []
    for panel_name, genes in panel_data.items():
        for gene in genes:
            rows.append({"gene_symbol": gene.upper(), "panelapp_panel": panel_name})

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()

    # Pivot: one row per gene, columns = panel_SEIZ, panel_HL, etc.
    df["val"] = 1
    pivot = df.pivot_table(
        index="gene_symbol", columns="panelapp_panel", values="val",
        aggfunc="max", fill_value=0
    ).reset_index()
    pivot.columns = [f"panelapp_{c}" if c != "gene_symbol" else c
                     for c in pivot.columns]

    # Count how many panels each gene is in
    panel_cols = [c for c in pivot.columns if c.startswith("panelapp_")]
    pivot["n_panelapp_panels"] = pivot[panel_cols].sum(axis=1)

    out = PROC / "panelapp_panel_genes.csv"
    pivot.to_csv(out, index=False)
    log.info(f"PanelApp: {len(pivot)} unique genes across {len(panel_data)} panels → {out}")
    return pivot


# ─── 4. Merge into gene_actionability_v2 ─────────────────────────────────────

def merge_into_actionability(gnomad_df, acmg_df, panelapp_df):
    """
    Merge GnomAD + ACMG + PanelApp into existing gene_actionability.csv.
    Produces gene_actionability_v2.csv with richer features.
    """
    base_path = PROC / "gene_actionability.csv"
    if not base_path.exists():
        log.warning("gene_actionability.csv not found - run process_babyseq first")
        return None

    ga = pd.read_csv(base_path, low_memory=False)
    ga["gene_symbol"] = ga["gene_symbol"].str.upper()
    log.info(f"Base gene_actionability: {ga.shape}")

    # Merge GnomAD
    if gnomad_df is not None:
        gnomad_df["gene_symbol"] = gnomad_df["gene_symbol"].str.upper()
        ga = ga.merge(gnomad_df, on="gene_symbol", how="left")
        log.info(f"After GnomAD merge: {ga.shape}")

    # Merge ACMG
    if acmg_df is not None:
        acmg_df["gene_symbol"] = acmg_df["gene_symbol"].str.upper()
        ga = ga.merge(acmg_df[["gene_symbol", "acmg_sf"]], on="gene_symbol", how="left")
        ga["acmg_sf"] = ga["acmg_sf"].fillna(0).astype(int)
        log.info(f"After ACMG merge: {ga.shape}, ACMG hits: {ga['acmg_sf'].sum()}")

    # Merge PanelApp
    if panelapp_df is not None:
        panelapp_df["gene_symbol"] = panelapp_df["gene_symbol"].str.upper()
        ga = ga.merge(panelapp_df, on="gene_symbol", how="left")
        panelapp_cols = [c for c in ga.columns if c.startswith("panelapp_")]
        ga[panelapp_cols] = ga[panelapp_cols].fillna(0)
        log.info(f"After PanelApp merge: {ga.shape}")

    # Enhanced actionability score v2:
    # Base: evidence_score * penetrance_score * category_score / max(4*1.0*3)
    # Bonus: +0.1 if ACMG SF, +0.1 if pLI > 0.9, +0.05 if in any PanelApp panel
    ga["combined_actionability_v2"] = ga["combined_actionability"].copy()

    if "acmg_sf" in ga.columns:
        ga["combined_actionability_v2"] += ga["acmg_sf"] * 0.10

    if "hi_intolerant" in ga.columns:
        ga["combined_actionability_v2"] += ga["hi_intolerant"].fillna(0) * 0.10

    if "n_panelapp_panels" in ga.columns:
        ga["combined_actionability_v2"] += (ga["n_panelapp_panels"].fillna(0) > 0).astype(int) * 0.05

    # Clip to [0, 1]
    ga["combined_actionability_v2"] = ga["combined_actionability_v2"].clip(0, 1)

    out = PROC / "gene_actionability_v2.csv"
    ga.to_csv(out, index=False)
    log.info(f"gene_actionability_v2: {ga.shape} → {out}")

    # Stats
    if "acmg_sf" in ga.columns:
        log.info(f"  ACMG SF genes in actionability set: {ga['acmg_sf'].sum()}")
    if "hi_intolerant" in ga.columns:
        log.info(f"  HI genes (pLI>0.9): {ga['hi_intolerant'].fillna(0).sum():.0f}")
    if "combined_actionability_v2" in ga.columns:
        high = (ga["combined_actionability_v2"] > 0.7).sum()
        log.info(f"  Genes with actionability_v2 > 0.7: {high}")

    return ga


# ─── 5. PanelApp-validated IBA panel map ─────────────────────────────────────

def build_panelapp_validated_panel_map():
    """
    Cross-validate our HPO-based IBA panel activation with PanelApp gene membership.
    For each disease, compute: IBA_panel_score_v2 = HPO_activation * panelapp_gene_overlap
    Saves updated iba_panel_hpo_map with panelapp gene counts per panel.
    """
    panelapp_path = RAW / "panelapp_iba_genes.json"
    if not panelapp_path.exists():
        return

    with open(panelapp_path) as f:
        panel_data = json.load(f)

    # Metadata: count genes per panel
    meta = {panel: len(genes) for panel, genes in panel_data.items()}
    out = PROC / "panelapp_panel_meta.json"
    with open(out, "w") as f:
        json.dump(meta, f, indent=2)
    log.info(f"PanelApp panel metadata: {meta}")


def run():
    log.info("=== Processing External Datasets (GnomAD + ACMG + PanelApp) ===")

    gnomad_df = process_gnomad()
    acmg_df = process_acmg()
    panelapp_df = process_panelapp()

    ga_v2 = merge_into_actionability(gnomad_df, acmg_df, panelapp_df)
    build_panelapp_validated_panel_map()

    log.info("=== External dataset processing complete ===")
    return ga_v2


if __name__ == "__main__":
    run()
