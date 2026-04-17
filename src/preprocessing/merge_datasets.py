# -*- coding: utf-8 -*-
"""
Merge all datasets into unified master tables:
1. master_gene_disease_phenotype  - core triplet table
2. disease_phenotype_matrix       - sample-level matrix (gene-disease pair x HPO) for classifier
3. graph_edges                    - unified edge list for KG
4. gene_metadata                  - enriched gene info
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from collections import Counter
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.data_loader import load_all, load_gene_attribute_matrix

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

PROC = Path(__file__).parent.parent.parent / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def build_master_table(datasets):
    """Merge genes_to_phenotype + gene2phenotype + orphanet + morbidmap into one big table."""
    frames = []

    # Source 1: genes_to_phenotype_ontology (293K rows)
    g2p = datasets.get("genes_to_phenotype")
    if g2p is not None:
        cols = list(g2p.columns)
        gene_col = next((c for c in cols if "gene_symbol" in c or c == "genesymbol"), None)
        gene_id_col = next((c for c in cols if "gene_id" in c or "entrez" in c or "ncbi" in c), None)
        hpo_col = next((c for c in cols if "hpo_id" in c), None)
        hpo_name_col = next((c for c in cols if "hpo_name" in c or "hpo_label" in c), None)
        disease_col = next((c for c in cols if "disease_id" in c), None)
        freq_col = next((c for c in cols if "freq" in c), None)

        frame = pd.DataFrame()
        if gene_col:
            frame["gene_symbol"] = g2p[gene_col]
        if gene_id_col:
            frame["gene_id"] = g2p[gene_id_col].astype(str)
        if hpo_col:
            frame["hpo_id"] = g2p[hpo_col]
        if hpo_name_col:
            frame["hpo_name"] = g2p[hpo_name_col]
        if disease_col:
            frame["disease_id"] = g2p[disease_col].astype(str)
        if freq_col:
            frame["frequency"] = g2p[freq_col]
        frame["source"] = "HPO_JAX"
        frame["confidence"] = "medium"
        frames.append(frame)
        log.info(f"Source 1 (genes_to_phenotype): {len(frame):,} rows")

    # Source 2: gene2phenotype (gold standard, confidence labels)
    g2pheno = datasets.get("gene2phenotype")
    if g2pheno is not None:
        cols = list(g2pheno.columns)
        gene_col = next((c for c in cols if "gene_symbol" in c), None) or cols[0]
        disease_col = next((c for c in cols if "disease_name" in c), None)
        confidence_col = next((c for c in cols if "confidence" in c), None)
        hpo_col = next((c for c in cols if "phenotype" in c or "hpo" in c), None)

        frame = pd.DataFrame()
        frame["gene_symbol"] = g2pheno[gene_col] if gene_col else np.nan
        if disease_col:
            frame["disease_name"] = g2pheno[disease_col]
        if hpo_col:
            frame["hpo_id"] = g2pheno[hpo_col]
        if confidence_col:
            frame["confidence"] = g2pheno[confidence_col]
        else:
            frame["confidence"] = "definitive"
        frame["source"] = "gene2phenotype"
        frames.append(frame)
        log.info(f"Source 2 (gene2phenotype): {len(frame):,} rows")

    # Source 3: orphanet_genes
    orpha_genes = datasets.get("orphanet_genes")
    if orpha_genes is not None:
        cols = list(orpha_genes.columns)
        gene_col = next((c for c in cols if "gene_symbol" in c), None)
        disease_col = next((c for c in cols if "disease_name" in c), None)
        orpha_col = next((c for c in cols if "orpha" in c or "orphacode" in c), None)
        assoc_col = next((c for c in cols if "association" in c or "type" in c), None)

        frame = pd.DataFrame()
        if gene_col:
            frame["gene_symbol"] = orpha_genes[gene_col]
        if disease_col:
            frame["disease_name"] = orpha_genes[disease_col]
        if orpha_col:
            frame["disease_id"] = "ORPHA:" + orpha_genes[orpha_col].astype(str)
        if assoc_col:
            frame["confidence"] = orpha_genes[assoc_col]
        else:
            frame["confidence"] = "orphanet"
        frame["source"] = "orphanet"
        frames.append(frame)
        log.info(f"Source 3 (orphanet_genes): {len(frame):,} rows")

    # Source 4: morbidmap
    morbid = datasets.get("morbidmap")
    if morbid is not None:
        frame = morbid.copy()
        frame["source"] = "OMIM_morbidmap"
        frame["confidence"] = "OMIM"
        frames.append(frame)
        log.info(f"Source 4 (morbidmap): {len(frame):,} rows")

    master = pd.concat(frames, ignore_index=True, sort=False)
    master = master.dropna(how="all")

    if "gene_symbol" in master.columns:
        master["gene_symbol"] = master["gene_symbol"].astype(str).str.strip().str.upper()
        master = master[master["gene_symbol"].notna() & (master["gene_symbol"] != "NAN")]

    log.info(f"Master table: {master.shape}")
    return master


def build_disease_phenotype_matrix(datasets):
    """
    Build SAMPLE-LEVEL matrix for classifier.
    Each sample = one (gene, disease) pair, features = multi-hot HPO vector.
    Multiple samples per disease since same disease links to multiple genes.
    Returns: (matrix_df, label_encoder_dict)
    """
    g2p = datasets.get("genes_to_phenotype")
    if g2p is None:
        log.error("genes_to_phenotype not loaded - cannot build matrix")
        return None, None

    cols = list(g2p.columns)
    disease_col = next((c for c in cols if "disease_id" in c or "disease" in c), None)
    hpo_col = next((c for c in cols if "hpo_id" in c), None)
    gene_col = next((c for c in cols if "gene_symbol" in c or "symbol" in c), None)

    if not disease_col or not hpo_col:
        log.error(f"Missing disease/hpo cols. Available: {cols}")
        return None, None

    log.info(f"Building sample-level matrix: disease={disease_col}, hpo={hpo_col}, gene={gene_col}")

    use_cols = [disease_col, hpo_col] + ([gene_col] if gene_col else [])
    df = g2p[use_cols].dropna(subset=[disease_col, hpo_col])
    df.columns = ["disease_id", "hpo_id"] + (["gene_symbol"] if gene_col else [])

    # Keep diseases with >= 3 HPO terms
    disease_hpo_counts = df.groupby("disease_id")["hpo_id"].nunique()
    valid_diseases = disease_hpo_counts[disease_hpo_counts >= 3].index
    df = df[df["disease_id"].isin(valid_diseases)]

    # Limit to top 500 diseases by HPO count
    top_diseases = disease_hpo_counts[disease_hpo_counts >= 3].nlargest(500).index
    df = df[df["disease_id"].isin(top_diseases)]

    # Top 300 HPO terms by frequency
    top_hpo = df["hpo_id"].value_counts().head(300).index
    df = df[df["hpo_id"].isin(top_hpo)]

    log.info(f"Filtered: {df['disease_id'].nunique()} diseases, {df['hpo_id'].nunique()} HPO terms")

    # Build per-disease HPO sets
    hpo_list = sorted(df["hpo_id"].unique())
    hpo_idx = {h: i for i, h in enumerate(hpo_list)}

    disease_hpo_sets = {}
    for disease_id, group in df.groupby("disease_id"):
        disease_hpo_sets[disease_id] = list(group["hpo_id"].unique())

    log.info(f"Diseases with HPO sets: {len(disease_hpo_sets)}")

    # Augmented sampling: for each disease, simulate 15 patients by random HPO subsets
    # Each simulated patient has 60-95% of the disease's HPO terms
    # This is a standard approach for rare disease ML with limited samples
    np.random.seed(42)
    N_AUGMENT = 15
    MIN_HPO_SUBSET = 3
    samples = []
    for disease_id, hpo_terms in disease_hpo_sets.items():
        if len(hpo_terms) < MIN_HPO_SUBSET:
            continue
        # Always include a "full" sample
        vec_full = np.zeros(len(hpo_list), dtype=np.int8)
        for hpo in hpo_terms:
            if hpo in hpo_idx:
                vec_full[hpo_idx[hpo]] = 1
        samples.append((disease_id, vec_full))
        # Augmented samples
        for _ in range(N_AUGMENT - 1):
            fraction = np.random.uniform(0.6, 0.95)
            n_select = max(MIN_HPO_SUBSET, int(len(hpo_terms) * fraction))
            selected = np.random.choice(hpo_terms, size=min(n_select, len(hpo_terms)), replace=False)
            vec = np.zeros(len(hpo_list), dtype=np.int8)
            for hpo in selected:
                if hpo in hpo_idx:
                    vec[hpo_idx[hpo]] = 1
            samples.append((disease_id, vec))

    log.info(f"Built {len(samples)} augmented samples from {len(disease_hpo_sets)} diseases")

    # Keep diseases with >= 10 samples after augmentation
    disease_sample_counts = Counter(s[0] for s in samples)
    valid = {d for d, c in disease_sample_counts.items() if c >= 10}
    samples = [(d, v) for d, v in samples if d in valid]

    log.info(f"After min-10-sample filter: {len(samples)} samples, {len(valid)} diseases")

    if not samples:
        log.error("No samples after filtering!")
        return None, None

    diseases_out = [s[0] for s in samples]
    X_out = np.array([s[1] for s in samples])

    matrix = pd.DataFrame(X_out, columns=hpo_list, index=diseases_out)
    matrix.index.name = "disease_id"

    disease_list = sorted(valid)
    label_enc = {d: i for i, d in enumerate(disease_list)}
    label_dec = {i: d for d, i in label_enc.items()}

    log.info(f"Final sample matrix: {matrix.shape}")
    return matrix, {"enc": label_enc, "dec": label_dec}


def build_graph_edges(datasets, master):
    """Build unified edge list for knowledge graph construction."""
    edges = []

    if "gene_symbol" in master.columns and "hpo_id" in master.columns:
        ge = master[["gene_symbol", "hpo_id", "source"]].dropna()
        ge = ge.rename(columns={"gene_symbol": "source_node", "hpo_id": "target_node"})
        ge["edge_type"] = "gene_phenotype"
        edges.append(ge)
        log.info(f"Gene-phenotype edges: {len(ge):,}")

    if "gene_symbol" in master.columns and "disease_id" in master.columns:
        gd = master[["gene_symbol", "disease_id", "confidence", "source"]].dropna()
        gd = gd.rename(columns={"gene_symbol": "source_node", "disease_id": "target_node"})
        gd["edge_type"] = "gene_disease"
        edges.append(gd)
        log.info(f"Gene-disease edges: {len(gd):,}")

    if "disease_id" in master.columns and "hpo_id" in master.columns:
        dp = master[["disease_id", "hpo_id", "source"]].dropna()
        dp = dp.rename(columns={"disease_id": "source_node", "hpo_id": "target_node"})
        dp["edge_type"] = "disease_phenotype"
        edges.append(dp)
        log.info(f"Disease-phenotype edges: {len(dp):,}")

    kg = datasets.get("kg")
    if kg is not None:
        cols = list(kg.columns)
        src = next((c for c in cols if "x_name" in c or "subject" in c), cols[0])
        tgt = next((c for c in cols if "y_name" in c or "object" in c), cols[1] if len(cols) > 1 else cols[0])
        rel = next((c for c in cols if "relation" in c or "display_relation" in c), None)
        kg_edge = kg[[src, tgt]].dropna().head(200000)
        kg_edge.columns = ["source_node", "target_node"]
        if rel:
            kg_edge["edge_type"] = kg[rel].head(200000).values
        else:
            kg_edge["edge_type"] = "kg_relation"
        kg_edge["source"] = "kg"
        edges.append(kg_edge)
        log.info(f"KG edges: {len(kg_edge):,}")

    if not edges:
        log.error("No edges built")
        return None

    all_edges = pd.concat(edges, ignore_index=True, sort=False)
    all_edges["source_node"] = all_edges["source_node"].astype(str).str.strip()
    all_edges["target_node"] = all_edges["target_node"].astype(str).str.strip()
    all_edges = all_edges.drop_duplicates(subset=["source_node", "target_node", "edge_type"])
    log.info(f"Total graph edges: {len(all_edges):,}")
    return all_edges


def build_gene_metadata(datasets):
    """Enrich gene info from multiple sources."""
    frames = []

    mim2gene = datasets.get("mim2gene")
    if mim2gene is not None:
        cols = list(mim2gene.columns)
        sym_col = next((c for c in cols if "symbol" in c), None)
        mim_col = next((c for c in cols if "mim" in c), cols[0])
        type_col = next((c for c in cols if "type" in c or "entry" in c), None)
        if sym_col:
            frame = mim2gene[[sym_col, mim_col] + ([type_col] if type_col else [])].dropna(subset=[sym_col])
            frame = frame.rename(columns={sym_col: "gene_symbol", mim_col: "mim_number"})
            frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
            frames.append(frame)

    gene_attr = datasets.get("gene_attribute_edges")
    if gene_attr is not None:
        cols = list(gene_attr.columns)
        src_col = next((c for c in cols if c == "source" or "gene" in c), cols[0])
        frame = gene_attr[[src_col]].dropna().rename(columns={src_col: "gene_symbol"})
        frame["gene_symbol"] = frame["gene_symbol"].astype(str).str.upper()
        frames.append(frame)

    if not frames:
        return None

    meta = pd.concat(frames, ignore_index=True, sort=False)
    meta = meta.drop_duplicates(subset=["gene_symbol"])
    log.info(f"Gene metadata: {meta.shape}")
    return meta


def run_merge(skip_large=False):
    log.info("=== Loading all datasets ===")
    datasets = load_all(skip_large=skip_large)

    log.info("=== Building master table ===")
    master = build_master_table(datasets)
    master.to_csv(PROC / "master_gene_disease_phenotype.csv", index=False)
    log.info(f"Saved master table: {master.shape}")

    log.info("=== Building disease-phenotype matrix ===")
    matrix, label_maps = build_disease_phenotype_matrix(datasets)
    if matrix is not None:
        matrix.to_csv(PROC / "disease_phenotype_matrix.csv")
        pd.DataFrame({"disease_id": list(label_maps["enc"].keys()),
                      "label": list(label_maps["enc"].values())}).to_csv(
            PROC / "disease_label_map.csv", index=False)
        log.info(f"Saved disease-phenotype matrix: {matrix.shape}")

    log.info("=== Building graph edges ===")
    edges = build_graph_edges(datasets, master)
    if edges is not None:
        edges.to_csv(PROC / "graph_edges.csv", index=False)
        log.info(f"Saved graph edges: {edges.shape}")

    log.info("=== Building gene metadata ===")
    gene_meta = build_gene_metadata(datasets)
    if gene_meta is not None:
        gene_meta.to_csv(PROC / "gene_metadata.csv", index=False)
        log.info(f"Saved gene metadata: {gene_meta.shape}")

    log.info("=== Preprocessing complete ===")
    return {
        "master": master,
        "matrix": matrix,
        "label_maps": label_maps,
        "edges": edges,
        "gene_meta": gene_meta,
        "raw_datasets": datasets,
    }


if __name__ == "__main__":
    result = run_merge(skip_large=False)
    print("\n=== Summary ===")
    for k, v in result.items():
        if hasattr(v, "shape"):
            print(f"  {k}: {v.shape}")
        elif isinstance(v, dict) and "enc" in (v or {}):
            print(f"  {k}: {len(v['enc'])} diseases")
