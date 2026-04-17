"""
Phenotype Retrieval Engine — Module 3
Input: symptom text / HPO IDs → ranked genes + diseases

Methods:
1. TF-IDF cosine similarity on HPO term names
2. Pre-computed gene similarity matrix (from gene_similarity_matrix_cosine.csv)
3. Graph-based similarity (from embeddings)
4. Pairwise learning-to-rank layer
"""

import pandas as pd
import numpy as np
import joblib
import logging
import json
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import normalize
import scipy.sparse as sp

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
RAW = BASE / "data" / "raw"
EMBED = BASE / "data" / "embeddings"
MODELS = BASE / "outputs" / "models"
RESULTS = BASE / "outputs" / "results"

for d in [MODELS, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)


def load_phenotype_data():
    """Load gene-phenotype-disease table for indexing."""
    path = PROC / "master_gene_disease_phenotype.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run merge_datasets.py first.")
    df = pd.read_csv(path, low_memory=False)
    log.info(f"Master table loaded: {df.shape}")
    return df


def load_gene_similarity_matrix(nrows=2000):
    """
    Load pre-computed gene cosine similarity matrix.
    Format: rows = gene symbols (index col 0), cols = OMIMID + GeneID + numeric gene IDs.
    We strip metadata columns and build a gene_symbol × gene_symbol matrix by
    mapping numeric gene IDs back to gene symbols via the GeneID/GeneID column.
    """
    path = RAW / "gene_similarity_matrix_cosine.csv"
    if not path.exists():
        log.warning("gene_similarity_matrix_cosine.csv not found")
        return None, None

    log.info(f"Loading gene similarity matrix (top {nrows} genes)...")
    try:
        sim_raw = pd.read_csv(path, index_col=0, nrows=nrows, low_memory=False)
        # sim_raw index = gene symbols (e.g. BLMH, A2M)
        # sim_raw columns = ['OMIMID', 'GeneID/GeneID', '642', '22976', ...]
        # Extract the gene_id → gene_symbol mapping from the matrix itself
        gene_id_col = next((c for c in sim_raw.columns if "geneid" in c.lower()), None)
        if gene_id_col:
            # Build gene_id → gene_symbol map from the loaded rows
            gene_id_to_sym = {}
            for sym, gid in zip(sim_raw.index, sim_raw[gene_id_col]):
                try:
                    gene_id_to_sym[str(int(float(gid)))] = str(sym).upper()
                except (ValueError, TypeError):
                    pass
        else:
            gene_id_to_sym = {}

        # Drop non-numeric metadata columns
        meta_cols = {"OMIMID", gene_id_col} if gene_id_col else set()
        sim_vals = sim_raw.drop(columns=[c for c in meta_cols if c in sim_raw.columns],
                                errors="ignore")

        # Rename numeric gene ID columns to gene symbols where mapping exists
        new_cols = []
        for c in sim_vals.columns:
            mapped = gene_id_to_sym.get(str(c), None)
            new_cols.append(mapped if mapped else c)
        sim_vals.columns = new_cols

        # Normalize row index to uppercase gene symbols
        sim_vals.index = sim_vals.index.astype(str).str.upper()

        # Keep only the square gene_symbol intersection
        row_syms = set(sim_vals.index)
        col_syms = set(sim_vals.columns)
        common = sorted(row_syms & col_syms)

        if not common:
            log.warning("Gene similarity matrix: no common gene symbols between rows and columns")
            # Still return the full matrix by symbol rows; col lookup by gene_id won't work
            return sim_vals, list(sim_vals.index)

        sim_sq = sim_vals.loc[common, common].astype(float)
        log.info(f"Gene similarity matrix (square): {sim_sq.shape}")
        return sim_sq, list(sim_sq.index)
    except Exception as e:
        log.warning(f"Failed to load gene similarity matrix: {e}")
        return None, None


def build_tfidf_index(df):
    """
    Build TF-IDF index over HPO term names for text-based symptom search.
    Document = all HPO names associated with each gene/disease.
    """
    log.info("Building TF-IDF index...")

    # Build gene documents: concat all HPO names per gene
    gene_docs = {}
    if "gene_symbol" in df.columns and "hpo_name" in df.columns:
        gene_hpo = df[["gene_symbol", "hpo_name"]].dropna()
        for gene, group in gene_hpo.groupby("gene_symbol"):
            gene_docs[gene] = " ".join(group["hpo_name"].astype(str).tolist())

    # Build disease documents
    disease_docs = {}
    if "disease_id" in df.columns and "hpo_name" in df.columns:
        dis_hpo = df[["disease_id", "hpo_name"]].dropna()
        for dis, group in dis_hpo.groupby("disease_id"):
            disease_docs[dis] = " ".join(group["hpo_name"].astype(str).tolist())

    log.info(f"Gene docs: {len(gene_docs)}, Disease docs: {len(disease_docs)}")

    # Combine for single unified index
    all_ids = list(gene_docs.keys()) + list(disease_docs.keys())
    all_docs = list(gene_docs.values()) + list(disease_docs.values())
    types = ["gene"] * len(gene_docs) + ["disease"] * len(disease_docs)

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=10000,
        sublinear_tf=True,
        min_df=2,
    )
    tfidf_matrix = vectorizer.fit_transform(all_docs)
    log.info(f"TF-IDF matrix: {tfidf_matrix.shape}")

    joblib.dump({"vectorizer": vectorizer, "matrix": tfidf_matrix,
                 "ids": all_ids, "types": types}, MODELS / "tfidf_index.pkl")
    log.info("TF-IDF index saved")

    return vectorizer, tfidf_matrix, all_ids, types


def build_hpo_lookup(df):
    """Build HPO ID → HPO name lookup and HPO name → ID lookup."""
    hpo_lookup = {}
    if "hpo_id" in df.columns and "hpo_name" in df.columns:
        hpo_pairs = df[["hpo_id", "hpo_name"]].dropna().drop_duplicates()
        hpo_lookup = dict(zip(hpo_pairs["hpo_id"], hpo_pairs["hpo_name"]))

    # Reverse
    name_to_id = {v.lower(): k for k, v in hpo_lookup.items()}

    with open(RESULTS / "hpo_lookup.json", "w") as f:
        json.dump(hpo_lookup, f)

    log.info(f"HPO lookup: {len(hpo_lookup)} terms")
    return hpo_lookup, name_to_id


def build_gene_disease_index(df):
    """Build gene → [diseases] and disease → [genes] lookup."""
    gene_to_diseases = {}
    disease_to_genes = {}

    if "gene_symbol" in df.columns and "disease_id" in df.columns:
        gd = df[["gene_symbol", "disease_id"]].dropna().drop_duplicates()
        for _, row in gd.iterrows():
            g, d = row["gene_symbol"], str(row["disease_id"])
            gene_to_diseases.setdefault(g, set()).add(d)
            disease_to_genes.setdefault(d, set()).add(g)

    gene_to_diseases = {k: list(v) for k, v in gene_to_diseases.items()}
    disease_to_genes = {k: list(v) for k, v in disease_to_genes.items()}

    joblib.dump({"gene_to_diseases": gene_to_diseases,
                 "disease_to_genes": disease_to_genes}, MODELS / "gene_disease_index.pkl")
    log.info(f"Gene index: {len(gene_to_diseases)} genes, Disease index: {len(disease_to_genes)} diseases")
    return gene_to_diseases, disease_to_genes


def retrieve_by_text(query, vectorizer, tfidf_matrix, all_ids, types, top_k=20):
    """
    Search by free text (symptom names).
    Returns ranked list of (id, type, score).
    """
    query_vec = vectorizer.transform([query])
    scores = cosine_similarity(query_vec, tfidf_matrix)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]

    results = []
    for idx in top_idx:
        if scores[idx] > 0:
            results.append({
                "id": all_ids[idx],
                "type": types[idx],
                "score": float(scores[idx]),
            })
    return results


def retrieve_by_hpo_ids(hpo_ids, df, top_k=20):
    """
    Given HPO ID list, find genes and diseases via direct lookup.
    Score = fraction of query HPO terms matched.
    """
    if "gene_symbol" not in df.columns or "hpo_id" not in df.columns:
        return []

    hpo_set = set(hpo_ids)
    gene_scores = {}
    disease_scores = {}

    filtered = df[df["hpo_id"].isin(hpo_set)]

    if "gene_symbol" in filtered.columns:
        gene_counts = filtered.groupby("gene_symbol")["hpo_id"].nunique()
        for gene, count in gene_counts.items():
            gene_scores[gene] = count / len(hpo_set)

    if "disease_id" in filtered.columns:
        dis_counts = filtered.groupby("disease_id")["hpo_id"].nunique()
        for dis, count in dis_counts.items():
            disease_scores[str(dis)] = count / len(hpo_set)

    results = []
    for gene, score in sorted(gene_scores.items(), key=lambda x: -x[1])[:top_k]:
        results.append({"id": gene, "type": "gene", "score": float(score), "method": "hpo_direct"})
    for dis, score in sorted(disease_scores.items(), key=lambda x: -x[1])[:top_k]:
        results.append({"id": dis, "type": "disease", "score": float(score), "method": "hpo_direct"})

    return sorted(results, key=lambda x: -x["score"])[:top_k * 2]


def retrieve_similar_genes(gene_name, sim_matrix, gene_list, top_k=10):
    """Find similar genes using pre-computed cosine similarity matrix."""
    gene_name_upper = gene_name.upper()
    if sim_matrix is None:
        return []
    if gene_name_upper not in sim_matrix.index:
        # Try partial match
        matches = [g for g in sim_matrix.index if gene_name_upper in g]
        if not matches:
            return []
        gene_name_upper = matches[0]

    scores = sim_matrix.loc[gene_name_upper]
    top = scores.nlargest(top_k + 1)
    return [(g, float(s)) for g, s in top.items() if g != gene_name_upper][:top_k]


def load_embedding_index():
    """Load node embeddings for graph-based retrieval."""
    emb_path = EMBED / "node_embeddings.npy"
    nodes_path = EMBED / "node_list.json"
    if not emb_path.exists():
        return None, None
    vecs = np.load(emb_path)
    with open(nodes_path) as f:
        nodes = json.load(f)
    vecs_norm = normalize(vecs, norm="l2")
    log.info(f"Loaded embedding index: {vecs.shape}")
    return vecs_norm, nodes


def retrieve_by_embedding(query_gene_or_hpo, vecs_norm, nodes, top_k=10):
    """Find nearest neighbors in embedding space."""
    if query_gene_or_hpo not in nodes:
        return []
    idx = nodes.index(query_gene_or_hpo)
    query_vec = vecs_norm[idx:idx+1]
    scores = (query_vec @ vecs_norm.T)[0]
    top_idx = np.argsort(scores)[::-1][1:top_k+1]
    return [(nodes[i], float(scores[i])) for i in top_idx]


def run_retrieval_setup():
    log.info("=== Building Retrieval Engine ===")

    df = load_phenotype_data()
    vectorizer, tfidf_matrix, all_ids, types = build_tfidf_index(df)
    hpo_lookup, name_to_id = build_hpo_lookup(df)
    gene_to_diseases, disease_to_genes = build_gene_disease_index(df)

    log.info("Loading gene similarity matrix...")
    sim_matrix, gene_list = load_gene_similarity_matrix(nrows=2000)

    log.info("=== Retrieval Engine Ready ===")
    log.info(f"  TF-IDF index: {len(all_ids):,} documents")
    log.info(f"  HPO terms: {len(hpo_lookup):,}")
    log.info(f"  Genes indexed: {len(gene_to_diseases):,}")
    log.info(f"  Diseases indexed: {len(disease_to_genes):,}")

    return {
        "df": df,
        "vectorizer": vectorizer,
        "tfidf_matrix": tfidf_matrix,
        "all_ids": all_ids,
        "types": types,
        "hpo_lookup": hpo_lookup,
        "name_to_id": name_to_id,
        "gene_to_diseases": gene_to_diseases,
        "disease_to_genes": disease_to_genes,
        "sim_matrix": sim_matrix,
        "gene_list": gene_list,
    }


def demo_retrieval(engine):
    """Demo: search for Marfan syndrome symptoms."""
    query = "aortic root dilatation lens dislocation tall stature"
    log.info(f"\nDemo query: '{query}'")
    results = retrieve_by_text(
        query, engine["vectorizer"], engine["tfidf_matrix"],
        engine["all_ids"], engine["types"], top_k=10
    )
    log.info("Top results:")
    for r in results[:5]:
        log.info(f"  [{r['type']:8s}] {r['id'][:40]:40s} score={r['score']:.4f}")

    # HPO ID lookup
    hpo_demo = ["HP:0001083", "HP:0002705", "HP:0001182"]
    hpo_results = retrieve_by_hpo_ids(hpo_demo, engine["df"], top_k=5)
    log.info(f"\nHPO-based lookup for {hpo_demo}:")
    for r in hpo_results[:5]:
        log.info(f"  [{r['type']:8s}] {r['id'][:40]:40s} score={r['score']:.4f}")


if __name__ == "__main__":
    engine = run_retrieval_setup()
    demo_retrieval(engine)
    log.info("Retrieval engine ready.")
