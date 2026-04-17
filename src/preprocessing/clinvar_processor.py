# -*- coding: utf-8 -*-
"""
ClinVar Variant Processor
Processes the 3.7GB variant_summary.txt into a per-gene pathogenicity summary.

Output: gene -> {n_pathogenic, n_likely_pathogenic, n_benign, n_vus, pathogenic_fraction}
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from collections import defaultdict

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
RAW = BASE / "data" / "raw"
PROC = BASE / "data" / "processed"
PROC.mkdir(parents=True, exist_ok=True)


def classify_significance(s):
    """Bucket ClinVar significance strings."""
    if not isinstance(s, str):
        return "unknown"
    sl = s.lower()
    if "likely pathogenic" in sl:
        return "likely_pathogenic"
    if "pathogenic" in sl:
        return "pathogenic"
    if "likely benign" in sl:
        return "likely_benign"
    if "benign" in sl:
        return "benign"
    if "uncertain" in sl or "vus" in sl:
        return "vus"
    return "other"


def process_clinvar(chunksize=100000, max_rows=None):
    """
    Stream ClinVar variant_summary.txt and aggregate per gene.
    """
    path = RAW / "variant_summary.txt"
    if not path.exists():
        log.warning("variant_summary.txt not found - skipping")
        return None

    log.info(f"Processing ClinVar {path.name} (chunked)...")
    log.info("This may take 2-5 minutes for 3.7GB file...")

    gene_stats = defaultdict(lambda: {
        "pathogenic": 0, "likely_pathogenic": 0,
        "benign": 0, "likely_benign": 0,
        "vus": 0, "other": 0, "total": 0,
    })

    total_rows = 0
    for chunk in pd.read_csv(
        path, sep="\t", chunksize=chunksize, on_bad_lines="skip",
        low_memory=False, encoding="utf-8", dtype=str,
    ):
        # Normalize column names
        chunk.columns = chunk.columns.str.strip().str.lower().str.replace(r"[^a-z0-9_]", "_", regex=True)

        gene_col = None
        for c in ["genesymbol", "gene_symbol", "gene"]:
            if c in chunk.columns:
                gene_col = c
                break
        sig_col = None
        for c in ["clinicalsignificance", "clinical_significance", "clinsig"]:
            if c in chunk.columns:
                sig_col = c
                break

        if not gene_col or not sig_col:
            log.warning(f"Missing columns. Available: {chunk.columns.tolist()[:15]}")
            continue

        # Process
        for gene, sig in zip(chunk[gene_col].fillna(""), chunk[sig_col].fillna("")):
            if not gene or gene == "-":
                continue
            # Handle multi-gene entries (semicolon-separated)
            for g in str(gene).split(";"):
                g = g.strip().upper()
                if not g or g == "-":
                    continue
                category = classify_significance(sig)
                gene_stats[g][category] += 1
                gene_stats[g]["total"] += 1

        total_rows += len(chunk)
        if total_rows % 500000 == 0:
            log.info(f"  Processed {total_rows:,} rows, {len(gene_stats):,} genes so far")
        if max_rows and total_rows >= max_rows:
            log.info(f"Reached max_rows={max_rows}")
            break

    log.info(f"Total rows processed: {total_rows:,}")
    log.info(f"Unique genes: {len(gene_stats):,}")

    # Build DataFrame
    rows = []
    for gene, stats in gene_stats.items():
        path_count = stats["pathogenic"] + stats["likely_pathogenic"]
        benign_count = stats["benign"] + stats["likely_benign"]
        rows.append({
            "gene_symbol": gene,
            "n_total_variants": stats["total"],
            "n_pathogenic": stats["pathogenic"],
            "n_likely_pathogenic": stats["likely_pathogenic"],
            "n_benign": stats["benign"],
            "n_likely_benign": stats["likely_benign"],
            "n_vus": stats["vus"],
            "pathogenic_fraction": path_count / max(stats["total"], 1),
            "benign_fraction": benign_count / max(stats["total"], 1),
            "clinical_actionability": "high" if path_count >= 10 else ("medium" if path_count >= 3 else "low"),
        })

    df = pd.DataFrame(rows)
    df = df.sort_values("n_pathogenic", ascending=False).reset_index(drop=True)
    log.info(f"ClinVar gene summary: {df.shape}")
    log.info(f"Top 10 pathogenic genes: {df.head(10)[['gene_symbol', 'n_pathogenic']].to_dict('records')}")

    df.to_csv(PROC / "clinvar_gene_summary.csv", index=False)
    log.info(f"Saved to {PROC / 'clinvar_gene_summary.csv'}")
    return df


if __name__ == "__main__":
    # Sample 500K rows for speed (full file is too large for quick iteration)
    df = process_clinvar(chunksize=50000, max_rows=500000)
    if df is not None:
        print(f"\n=== ClinVar Summary: {len(df)} genes ===")
        print(df.head(15))
