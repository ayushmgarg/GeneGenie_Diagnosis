"""
Data Loader — loads ALL datasets with edge case handling.
Handles encoding errors, bad lines, large files (chunked), missing columns.
"""

import pandas as pd
import numpy as np
import os
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

RAW = Path(__file__).parent.parent.parent / "data" / "raw"
PROC = Path(__file__).parent.parent.parent / "data" / "processed"


def safe_read_csv(path, sep=",", comment=None, encoding="utf-8", chunksize=None,
                  nrows=None, dtype=None, on_bad_lines="warn", **kwargs):
    """Read CSV/TSV with full fallback chain."""
    path = Path(path)
    if not path.exists():
        log.warning(f"File not found: {path}")
        return None
    try:
        df = pd.read_csv(
            path, sep=sep, comment=comment, encoding=encoding,
            chunksize=chunksize, nrows=nrows, dtype=dtype,
            on_bad_lines=on_bad_lines, low_memory=False, **kwargs
        )
        if chunksize:
            return df  # iterator
        log.info(f"Loaded {path.name}: {len(df):,} rows")
        return df
    except UnicodeDecodeError:
        log.warning(f"UTF-8 failed for {path.name}, trying latin-1")
        df = pd.read_csv(
            path, sep=sep, comment=comment, encoding="latin-1",
            nrows=nrows, on_bad_lines=on_bad_lines, low_memory=False, **kwargs
        )
        log.info(f"Loaded {path.name} (latin-1): {len(df):,} rows")
        return df
    except Exception as e:
        log.error(f"Failed to load {path.name}: {e}")
        return None


def load_genes_to_phenotype():
    """293K rows. Gene-HPO-Disease triplets. Primary dataset."""
    df = safe_read_csv(RAW / "genes_to_phenotype_ontology.csv")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    # Standardise key column names
    rename = {}
    for col in df.columns:
        if "gene" in col and "id" in col and "ncbi" not in col and "hpo" not in col:
            rename[col] = "gene_id"
        if "symbol" in col or col == "gene_symbol":
            rename[col] = "gene_symbol"
        if col.startswith("hpo_id") or col == "hpo_id":
            rename[col] = "hpo_id"
        if col.startswith("hpo_name") or col == "hpo_name":
            rename[col] = "hpo_name"
        if "disease" in col and ("id" in col or "omim" in col):
            rename[col] = "disease_id"
        if "freq" in col:
            rename[col] = "frequency"
    df = df.rename(columns=rename)
    df = df.dropna(subset=[c for c in ["gene_symbol", "hpo_id"] if c in df.columns])
    log.info(f"genes_to_phenotype: {df.shape}, cols={list(df.columns)}")
    return df


def load_gene2phenotype():
    """4,573 rows with confidence levels — gold standard links."""
    df = safe_read_csv(RAW / "gene2phenotype.csv")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"gene2phenotype: {df.shape}, cols={list(df.columns)}")
    return df


def load_orphanet_genes():
    """8,375 rows. Orphanet disease-gene associations."""
    df = safe_read_csv(RAW / "orphanet_genes.csv")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"orphanet_genes: {df.shape}, cols={list(df.columns)}")
    return df


def load_orphanet_diseases():
    """11,457 rows. Orphanet disease metadata."""
    df = safe_read_csv(RAW / "orphanet_diseases.csv")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"orphanet_diseases: {df.shape}, cols={list(df.columns)}")
    return df


def load_mim2gene():
    """26,725 rows. MIM number ↔ gene ID mapping."""
    df = safe_read_csv(RAW / "mim2gene.csv", comment="#")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"mim2gene: {df.shape}, cols={list(df.columns)}")
    return df


def load_mimtitles():
    """26,725 rows. OMIM disease/gene titles."""
    df = safe_read_csv(RAW / "mimTitles.csv", comment="#")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"mimTitles: {df.shape}, cols={list(df.columns)}")
    return df


def load_morbidmap():
    """OMIM morbid map — phenotype-gene-disease links."""
    df = safe_read_csv(RAW / "morbidmap.csv", comment="#")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"morbidmap: {df.shape}, cols={list(df.columns)}")
    return df


def load_genemap2():
    """OMIM genemap — gene-disease map with phenotype classifications."""
    df = safe_read_csv(RAW / "genemap2.txt", sep="\t", comment="#")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"genemap2: {df.shape}, cols={list(df.columns)}")
    return df


def load_gene_attribute_edges():
    """Gene-phenotype edge list."""
    df = safe_read_csv(RAW / "gene_attribute_edges.csv")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"gene_attribute_edges: {df.shape}, cols={list(df.columns)}")
    return df


def load_homo_sapiens_gene_info(nrows=None):
    """34MB. NCBI gene info — symbols, descriptions, chromosomes."""
    df = safe_read_csv(RAW / "Homo_sapiens.gene_info", sep="\t", comment="#", nrows=nrows)
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"gene_info: {df.shape}, cols={list(df.columns[:10])}...")
    return df


def load_phenotype_to_genes_jax(nrows=200000):
    """1M rows — sample for speed. HPO→gene→disease."""
    df = safe_read_csv(RAW / "phenotype_to_genes_JAX.csv", nrows=nrows)
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"phenotype_to_genes_JAX (sampled): {df.shape}, cols={list(df.columns)}")
    return df


def load_variant_summary(nrows=100000):
    """3.7GB ClinVar — sample pathogenic variants only."""
    path = RAW / "variant_summary.txt"
    if not path.exists():
        log.warning("variant_summary.txt not found")
        return None
    log.info("Loading ClinVar variant_summary (chunked, pathogenic only)...")
    chunks = []
    chunk_iter = safe_read_csv(path, sep="\t", chunksize=50000, on_bad_lines="skip")
    if chunk_iter is None:
        return None
    total = 0
    for chunk in chunk_iter:
        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
        # Filter to pathogenic/likely pathogenic only
        if "clinicalsignificance" in chunk.columns:
            sig_col = "clinicalsignificance"
        elif "clinical_significance" in chunk.columns:
            sig_col = "clinical_significance"
        else:
            sig_col = None
        if sig_col:
            mask = chunk[sig_col].str.contains("athogenic", case=False, na=False)
            chunk = chunk[mask]
        chunks.append(chunk)
        total += len(chunk)
        if total >= nrows:
            break
    if not chunks:
        log.warning("No pathogenic variants found in ClinVar sample")
        return None
    df = pd.concat(chunks, ignore_index=True)
    log.info(f"variant_summary (pathogenic sample): {df.shape}")
    return df


def load_gene_attribute_matrix(nrows=None):
    """108MB sparse matrix — gene × phenotype features."""
    df = safe_read_csv(RAW / "gene_attribute_matrix.csv", index_col=0, nrows=nrows)
    if df is not None:
        log.info(f"gene_attribute_matrix: {df.shape}")
    return df


def load_attribute_list():
    """HPO disease phenotype class mapping."""
    df = safe_read_csv(RAW / "attribute_list_entries.txt", sep="\t", on_bad_lines="skip")
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"attribute_list_entries: {df.shape}")
    return df


def load_kg(nrows=500000):
    """8.1M row knowledge graph — sample."""
    df = safe_read_csv(RAW / "kg.csv", nrows=nrows)
    if df is None:
        return None
    df.columns = df.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)
    log.info(f"kg (sampled): {df.shape}, cols={list(df.columns)}")
    return df


def load_all(skip_large=False):
    """
    Load all datasets needed for merge_datasets.py pipeline.

    Note on variant_summary.txt (3.7 GB ClinVar):
      NOT loaded here — it is processed separately by clinvar_processor.py which
      streams it in chunks and outputs clinvar_gene_summary.csv. That CSV is then
      loaded by enrich_features.py into the gene feature pipeline. Loading the raw
      variant_summary here would waste ~30s I/O with no consumer in this pipeline.

    skip_large=True: skips gene_attribute_matrix (108 MB) for faster dev runs.
    """
    datasets = {}
    datasets["genes_to_phenotype"] = load_genes_to_phenotype()
    datasets["gene2phenotype"] = load_gene2phenotype()
    datasets["orphanet_genes"] = load_orphanet_genes()
    datasets["orphanet_diseases"] = load_orphanet_diseases()
    datasets["mim2gene"] = load_mim2gene()
    datasets["mimtitles"] = load_mimtitles()
    datasets["morbidmap"] = load_morbidmap()
    datasets["genemap2"] = load_genemap2()
    datasets["gene_attribute_edges"] = load_gene_attribute_edges()
    datasets["attribute_list"] = load_attribute_list()
    datasets["gene_info"] = load_homo_sapiens_gene_info()
    datasets["phenotype_to_genes_jax"] = load_phenotype_to_genes_jax()
    datasets["kg"] = load_kg()
    if not skip_large:
        # gene_attribute_matrix is used by enrich_features.py → gene_enriched_features.csv
        datasets["gene_attribute_matrix"] = load_gene_attribute_matrix()
    loaded = {k: v for k, v in datasets.items() if v is not None}
    failed = [k for k, v in datasets.items() if v is None]
    log.info(f"Loaded {len(loaded)}/{len(datasets)} datasets. Failed: {failed}")
    return loaded


if __name__ == "__main__":
    data = load_all(skip_large=False)
    for name, df in data.items():
        print(f"  {name:35s}: {df.shape}")
