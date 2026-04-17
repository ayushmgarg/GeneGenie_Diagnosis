# -*- coding: utf-8 -*-
"""
Feature Enrichment Pipeline - Day 5/6
Integrates ALL remaining raw datasets into enriched feature tables:
  1. gene_attribute_matrix.csv   -> per-gene phenotype class features
  2. gene_similarity_matrix_cosine.csv -> gene similarity edges for KG
  3. orphanet_genes.csv          -> additional disease-gene links
  4. diseases_for_HP_0000118     -> all OMIM/ORPHA disease catalog
  5. genes_for_HP_0000118        -> root HPO gene catalog
  6. attribute_list_entries.txt  -> disease-attribute lookup
  7. gene_list_terms.txt         -> gene term annotations
  8. clinvar_gene_summary.csv    -> per-gene pathogenicity (already processed)

Outputs:
  data/processed/gene_enriched_features.csv  <- per-gene feature vector
  data/processed/all_diseases_catalog.csv    <- full disease list
  data/processed/graph_edges_enriched.csv    <- extended KG edges
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def load_gene_attribute_matrix(top_n=150):
    """
    gene_attribute_matrix.csv: rows=genes, cols=OMIM phenotype classes (6178).
    Returns per-gene summary: top_n most frequent attribute features + summary stats.
    """
    path = RAW / "gene_attribute_matrix.csv"
    if not path.exists():
        log.warning("gene_attribute_matrix.csv not found")
        return None

    log.info("Loading gene_attribute_matrix (107MB)...")
    # First row is header, first 2 cols are GeneSym, OMIMID
    df = pd.read_csv(path, index_col=0, low_memory=False)
    log.info(f"gene_attribute_matrix shape: {df.shape}")

    # Drop OMIMID column if present
    if "OMIMID" in df.columns:
        df = df.drop(columns=["OMIMID"])
    if "GeneID/PhenotypeClass" in df.columns:
        df = df.drop(columns=["GeneID/PhenotypeClass"])

    # Convert to numeric
    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)

    # Top N most frequent attribute columns (non-zero across genes)
    col_sums = df.sum(axis=0).sort_values(ascending=False)
    top_cols = col_sums.head(top_n).index.tolist()
    df_top = df[top_cols].copy()

    # Summary statistics per gene
    gene_attr = pd.DataFrame(index=df.index)
    gene_attr["n_attributes"] = (df > 0).sum(axis=1)
    gene_attr["attr_sum"] = df.sum(axis=1)
    gene_attr["attr_density"] = gene_attr["n_attributes"] / max(df.shape[1], 1)
    gene_attr["top_attr_sum"] = df_top.sum(axis=1)

    # Merge top features
    df_top.columns = [f"attr_{i}" for i in range(top_n)]
    result = pd.concat([gene_attr, df_top], axis=1)
    result.index.name = "gene_symbol"
    result = result.reset_index()

    log.info(f"Gene attribute features: {result.shape}")
    return result


def load_gene_similarity_edges(threshold=0.8, max_edges=50000):
    """
    gene_similarity_matrix_cosine.csv: gene x gene cosine similarity.
    Extract high-similarity pairs as new KG edges.
    """
    path = RAW / "gene_similarity_matrix_cosine.csv"
    if not path.exists():
        log.warning("gene_similarity_matrix_cosine.csv not found")
        return None

    log.info("Loading gene_similarity_matrix (80MB)...")
    df = pd.read_csv(path, index_col=0, low_memory=False)

    # Drop metadata cols
    for c in ["OMIMID", "GeneID/GeneID"]:
        if c in df.columns:
            df = df.drop(columns=[c])

    df = df.apply(pd.to_numeric, errors="coerce").fillna(0)
    gene_names = df.index.tolist()

    edges = []
    arr = df.values
    n = len(gene_names)

    for i in range(n):
        for j in range(i + 1, n):
            sim = arr[i, j]
            if sim >= threshold:
                edges.append({
                    "source_node": gene_names[i],
                    "target_node": gene_names[j],
                    "source": "gene_similarity",
                    "edge_type": "similar_gene",
                    "confidence": float(sim),
                })
                if len(edges) >= max_edges:
                    break
        if len(edges) >= max_edges:
            break

    df_edges = pd.DataFrame(edges)
    log.info(f"Gene similarity edges (>= {threshold}): {len(df_edges)}")
    return df_edges


def load_orphanet_genes_csv():
    """
    orphanet_genes.csv: richer gene-disease with ensembl/HGNC/OMIM links.
    Returns edges to add to KG.
    """
    path = RAW / "orphanet_genes.csv"
    if not path.exists():
        log.warning("orphanet_genes.csv not found")
        return None

    df = pd.read_csv(path, low_memory=False)
    log.info(f"orphanet_genes.csv: {df.shape}")

    edges = []
    for _, row in df.iterrows():
        gene = str(row.get("gene_symbol", "")).strip().upper()
        disease = str(row.get("disease_name", "")).strip()
        orpha = str(row.get("orpha_code", "")).strip()
        assoc = str(row.get("association_type", "")).strip()

        if not gene or gene == "NAN" or not disease:
            continue

        disease_id = orpha if orpha and orpha != "NAN" else disease

        edges.append({
            "source_node": gene,
            "target_node": disease_id,
            "source": "orphanet_genes_csv",
            "edge_type": "gene_disease_association",
            "confidence": 0.9 if "causative" in assoc.lower() else 0.7,
        })

    df_edges = pd.DataFrame(edges)
    log.info(f"Orphanet gene edges: {len(df_edges)}")
    return df_edges


def load_diseases_catalog():
    """
    diseases_for_HP_0000118: all diseases associated with root HPO phenotype.
    Parses TSV: id [tab] name
    """
    path = RAW / "diseases_for_HP_0000118"
    if not path.exists():
        log.warning("diseases_for_HP_0000118 not found")
        return None

    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("id"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                records.append({"disease_id": parts[0].strip(), "disease_name": parts[1].strip()})

    df = pd.DataFrame(records)
    log.info(f"Diseases catalog (HP:0000118): {len(df)} diseases")
    return df


def load_genes_catalog():
    """
    genes_for_HP_0000118: all genes for root HPO term.
    """
    path = RAW / "genes_for_HP_0000118"
    if not path.exists():
        log.warning("genes_for_HP_0000118 not found")
        return None

    records = []
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("id"):
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                records.append({"gene_id": parts[0].strip(), "gene_name": parts[1].strip()})

    df = pd.DataFrame(records)
    log.info(f"Genes catalog (HP:0000118): {len(df)} genes")
    return df


def load_attribute_list():
    """
    attribute_list_entries.txt: Disease, OMIMID, PhenotypeClass
    """
    path = RAW / "attribute_list_entries.txt"
    if not path.exists():
        log.warning("attribute_list_entries.txt not found")
        return None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False, on_bad_lines="skip")
        log.info(f"Attribute list: {df.shape}, cols={df.columns.tolist()}")
        return df
    except Exception as e:
        log.error(f"Failed to load attribute_list_entries.txt: {e}")
        return None


def load_gene_list_terms():
    """
    gene_list_terms.txt: GeneSym, OMIMID, GeneID
    Additional gene metadata (locus, entrez IDs).
    """
    path = RAW / "gene_list_terms.txt"
    if not path.exists():
        log.warning("gene_list_terms.txt not found")
        return None

    try:
        df = pd.read_csv(path, sep="\t", low_memory=False, on_bad_lines="skip")
        df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
        log.info(f"Gene list terms: {df.shape}")
        return df
    except Exception as e:
        log.error(f"Failed to load gene_list_terms.txt: {e}")
        return None


def build_gene_enriched_features():
    """
    Merge all per-gene features into one enriched feature CSV:
      - gene_attribute_matrix features (top 150 attributes + summary)
      - ClinVar pathogenicity features
      - Gene list terms (OMIM ID, GeneID)
    """
    log.info("=== Building enriched gene feature table ===")

    # 1. Gene attribute matrix
    attr_df = load_gene_attribute_matrix(top_n=150)
    if attr_df is None:
        log.warning("No gene attribute features - creating empty")
        attr_df = pd.DataFrame(columns=["gene_symbol"])

    # 2. ClinVar features
    clinvar_path = PROC / "clinvar_gene_summary.csv"
    if clinvar_path.exists():
        clinvar = pd.read_csv(clinvar_path)
        clinvar = clinvar[["gene_symbol", "n_total_variants", "n_pathogenic",
                            "n_likely_pathogenic", "n_vus", "pathogenic_fraction",
                            "benign_fraction", "clinical_actionability"]].copy()
        clinvar["log_n_pathogenic"] = np.log1p(clinvar["n_pathogenic"])
        clinvar["log_n_variants"] = np.log1p(clinvar["n_total_variants"])
        clinvar["actionability_score"] = clinvar["clinical_actionability"].map(
            {"high": 3, "medium": 2, "low": 1}).fillna(1)
        log.info(f"ClinVar features: {clinvar.shape}")
    else:
        clinvar = pd.DataFrame(columns=["gene_symbol"])
        log.warning("ClinVar summary not found")

    # 3. Gene list terms
    gene_terms = load_gene_list_terms()

    # Merge all
    merged = attr_df.copy() if not attr_df.empty else pd.DataFrame()

    if not clinvar.empty:
        if merged.empty:
            merged = clinvar
        else:
            merged = merged.merge(clinvar, on="gene_symbol", how="outer")
        log.info(f"After ClinVar merge: {merged.shape}")

    if gene_terms is not None and "genesym" in gene_terms.columns:
        gene_terms = gene_terms.rename(columns={"genesym": "gene_symbol"})
        merged = merged.merge(gene_terms[["gene_symbol", "omimid", "geneid"]].dropna(subset=["gene_symbol"]),
                              on="gene_symbol", how="left")
        log.info(f"After gene_terms merge: {merged.shape}")

    if merged.empty:
        log.warning("No enriched features built")
        return None

    # 4. gnomAD v4.1 constraint scores (pLI, LOEUF, mis_z)
    gnomad_path = PROC / "gnomad_gene_constraint.csv"
    if gnomad_path.exists():
        gnomad = pd.read_csv(gnomad_path, low_memory=False)
        gnomad_cols = ["gene_symbol"]
        for c in ["pli", "loeuf", "mis_z", "hi_intolerant", "loeuf_constrained"]:
            if c in gnomad.columns:
                gnomad_cols.append(c)
        if len(gnomad_cols) > 1:
            merged = merged.merge(gnomad[gnomad_cols].drop_duplicates("gene_symbol"),
                                  on="gene_symbol", how="left")
            # Fill missing gnomAD with neutral defaults
            for c in ["pli", "loeuf", "mis_z"]:
                if c in merged.columns:
                    merged[c] = merged[c].fillna(merged[c].median() if merged[c].notna().any() else 0)
            for c in ["hi_intolerant", "loeuf_constrained"]:
                if c in merged.columns:
                    merged[c] = merged[c].fillna(0).astype(int)
            log.info(f"After gnomAD merge: {merged.shape}")
    else:
        log.info("gnomad_gene_constraint.csv not found — run process_external_datasets.py first")

    merged.to_csv(PROC / "gene_enriched_features.csv", index=False)
    log.info(f"Saved gene_enriched_features.csv: {merged.shape}")
    return merged


def build_extended_graph_edges():
    """
    Extend graph_edges.csv with:
    - orphanet_genes.csv edges
    - gene_similarity edges (high-sim only)
    """
    log.info("=== Building extended graph edges ===")

    # Load existing edges
    existing_path = PROC / "graph_edges.csv"
    if existing_path.exists():
        existing = pd.read_csv(existing_path, low_memory=False)
        log.info(f"Existing edges: {len(existing)}")
    else:
        existing = pd.DataFrame(columns=["source_node", "target_node", "source", "edge_type", "confidence"])

    new_edges = []

    # Orphanet genes
    orphanet_edges = load_orphanet_genes_csv()
    if orphanet_edges is not None:
        new_edges.append(orphanet_edges)

    # Gene similarity (threshold 0.85 for quality)
    sim_edges = load_gene_similarity_edges(threshold=0.85, max_edges=30000)
    if sim_edges is not None:
        new_edges.append(sim_edges)

    if not new_edges:
        log.warning("No new edges to add")
        return existing

    combined = pd.concat([existing] + new_edges, ignore_index=True)
    # Deduplicate
    combined = combined.drop_duplicates(subset=["source_node", "target_node", "edge_type"])
    combined.to_csv(PROC / "graph_edges_enriched.csv", index=False)
    log.info(f"Extended graph edges: {len(combined)} (was {len(existing)})")
    return combined


def build_all_diseases_catalog():
    """
    Build comprehensive disease catalog from all sources.
    """
    log.info("=== Building all-diseases catalog ===")

    dfs = []

    # diseases_for_HP_0000118
    d1 = load_diseases_catalog()
    if d1 is not None:
        d1["source"] = "HPO_root"
        dfs.append(d1)

    # orphanet_diseases.csv
    orpha_path = RAW / "orphanet_diseases.csv"
    if orpha_path.exists():
        d2 = pd.read_csv(orpha_path)
        d2.columns = d2.columns.str.lower()
        if "orphacode" in d2.columns:
            d2 = d2.rename(columns={"orphacode": "disease_id", "name": "disease_name"})
        d2["source"] = "orphanet"
        dfs.append(d2[["disease_id", "disease_name", "source"]])
        log.info(f"Orphanet diseases: {len(d2)}")

    # From mimTitles
    mim_path = RAW / "mimTitles.csv"
    if mim_path.exists():
        mim = pd.read_csv(mim_path, low_memory=False)
        mim.columns = mim.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
        if "mim_number" in mim.columns and "preferred_title" in mim.columns:
            d3 = pd.DataFrame({
                "disease_id": "OMIM:" + mim["mim_number"].astype(str),
                "disease_name": mim["preferred_title"],
                "source": "OMIM",
            })
            dfs.append(d3)
            log.info(f"OMIM titles: {len(d3)}")

    if not dfs:
        log.warning("No disease catalog data found")
        return None

    catalog = pd.concat(dfs, ignore_index=True).drop_duplicates(subset=["disease_id"])
    catalog.to_csv(PROC / "all_diseases_catalog.csv", index=False)
    log.info(f"All diseases catalog: {len(catalog)} unique diseases")
    return catalog


def run():
    log.info("=== Feature Enrichment Pipeline ===")

    # 1. Enrich gene features (uses gene_attribute_matrix + ClinVar + gene_list_terms)
    gene_features = build_gene_enriched_features()
    if gene_features is not None:
        log.info(f"Gene enriched features: {gene_features.shape}")

    # 2. Extend graph edges
    ext_edges = build_extended_graph_edges()
    log.info(f"Extended graph: {len(ext_edges)} edges")

    # 3. Build disease catalog
    catalog = build_all_diseases_catalog()

    # 4. Save attribute list
    attr_list = load_attribute_list()
    if attr_list is not None:
        attr_list.to_csv(PROC / "attribute_list.csv", index=False)
        log.info(f"Saved attribute list: {attr_list.shape}")

    # 5. Save genes catalog
    genes_cat = load_genes_catalog()
    if genes_cat is not None:
        genes_cat.to_csv(PROC / "genes_catalog.csv", index=False)
        log.info(f"Saved genes catalog: {len(genes_cat)}")

    log.info("=== Enrichment complete ===")


if __name__ == "__main__":
    run()
