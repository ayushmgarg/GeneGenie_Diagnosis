# -*- coding: utf-8 -*-
"""
Disease Map Builder - Day 7
Generates UMAP embedding of diseases (from HPO phenotype matrix),
Louvain community detection, and interactive Plotly HTML map.

Outputs:
  outputs/embeddings/disease_umap.npy     (N x 2)
  outputs/embeddings/disease_umap_ids.json
  data/processed/louvain_clusters.csv
  outputs/plots/plotly_disease_map.html
"""

import numpy as np
import pandas as pd
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "outputs" / "models"
EMBED = BASE / "outputs" / "embeddings"
PLOTS = BASE / "outputs" / "plots"
RESULTS = BASE / "outputs" / "results"

EMBED.mkdir(parents=True, exist_ok=True)
PLOTS.mkdir(parents=True, exist_ok=True)


def load_data():
    """Load disease phenotype matrix + metadata."""
    dpm = pd.read_csv(PROC / "disease_phenotype_matrix.csv", index_col=0, low_memory=False)
    log.info(f"Disease phenotype matrix: {dpm.shape}")

    master = None
    master_path = PROC / "master_gene_disease_phenotype.csv"
    if master_path.exists():
        master = pd.read_csv(master_path, low_memory=False,
                             usecols=lambda c: c in ["disease_id", "disease_name", "gene_symbol"])
        log.info(f"Master: {master.shape}")

    babyseq = None
    bs_path = PROC / "babyseq_gene_disease.csv"
    if bs_path.exists():
        babyseq = pd.read_csv(bs_path, low_memory=False)

    return dpm, master, babyseq


def build_umap(X, n_neighbors=15, min_dist=0.1, metric="jaccard", random_state=42):
    """Compute UMAP embedding of binary HPO matrix."""
    import umap
    log.info(f"Running UMAP on {X.shape} matrix ...")
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
        verbose=False,
    )
    embedding = reducer.fit_transform(X)
    log.info(f"UMAP done: {embedding.shape}")
    return embedding


def build_louvain_clusters(X, disease_ids, resolution=1.0):
    """
    Build disease similarity graph (cosine) -> Louvain community detection.
    Returns cluster assignment array.
    """
    from sklearn.metrics.pairwise import cosine_similarity
    import networkx as nx
    import community as community_louvain

    log.info("Building similarity graph for Louvain ...")
    # Use subset of top-200 diseases for graph (full 500 is manageable)
    sims = cosine_similarity(X)

    G = nx.Graph()
    n = len(disease_ids)
    for i in range(n):
        G.add_node(disease_ids[i])

    # Add edges above threshold
    threshold = 0.15
    for i in range(n):
        for j in range(i + 1, n):
            w = float(sims[i, j])
            if w >= threshold:
                G.add_edge(disease_ids[i], disease_ids[j], weight=w)

    log.info(f"Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    partition = community_louvain.best_partition(G, weight="weight", resolution=resolution)
    log.info(f"Louvain: {len(set(partition.values()))} communities")
    return partition


def build_disease_map(save_html=True):
    """Full pipeline: load -> UMAP -> Louvain -> Plotly HTML."""
    import plotly.graph_objects as go
    import plotly.express as px

    dpm, master, babyseq = load_data()

    # Deduplicate: one HPO profile per disease (mean aggregation)
    dpm_unique = dpm.groupby(dpm.index).mean()
    log.info(f"Deduplicated: {dpm_unique.shape} unique diseases")
    disease_ids = dpm_unique.index.tolist()
    X = dpm_unique.values.astype(np.float32)

    # Disease name lookup
    name_map = {}
    gene_map = {}
    if master is not None:
        name_map = master.groupby("disease_id")["disease_name"].first().to_dict()
        gene_map = (master.groupby("disease_id")["gene_symbol"]
                    .apply(lambda x: ", ".join(x.dropna().unique()[:3]))
                    .to_dict())

    # BabySeq category A lookup
    cat_a_genes = set()
    if babyseq is not None and "is_category_a" in babyseq.columns:
        cat_a_genes = set(babyseq[babyseq["is_category_a"] == 1]["Gene"].str.upper().dropna())

    # UMAP
    embedding = build_umap(X, metric="jaccard")
    np.save(EMBED / "disease_umap.npy", embedding)
    with open(EMBED / "disease_umap_ids.json", "w") as f:
        json.dump(disease_ids, f)
    log.info("UMAP saved")

    # Louvain
    partition = build_louvain_clusters(X, disease_ids)
    cluster_arr = [partition.get(did, -1) for did in disease_ids]

    # Save louvain clusters
    cluster_df = pd.DataFrame({
        "disease_id": disease_ids,
        "disease_name": [name_map.get(d, d) for d in disease_ids],
        "cluster": cluster_arr,
        "genes": [gene_map.get(d, "") for d in disease_ids],
        "umap_x": embedding[:, 0],
        "umap_y": embedding[:, 1],
    })
    cluster_df.to_csv(PROC / "louvain_clusters.csv", index=False)
    log.info(f"Louvain clusters saved: {cluster_df['cluster'].nunique()} clusters")

    # Count HPOs per disease (for size scaling)
    hpo_count = X.sum(axis=1)

    # BabySeq newborn alert
    newborn = []
    for did in disease_ids:
        genes = gene_map.get(did, "").split(", ")
        newborn.append(any(g.upper() in cat_a_genes for g in genes if g))

    # Plotly figure
    n_clusters = cluster_df["cluster"].nunique()
    colors = px.colors.qualitative.Dark24 + px.colors.qualitative.Light24
    color_map = {c: colors[i % len(colors)] for i, c in enumerate(sorted(cluster_df["cluster"].unique()))}

    fig = go.Figure()

    # Plot by cluster
    for cluster_id in sorted(cluster_df["cluster"].unique()):
        mask = cluster_df["cluster"] == cluster_id
        sub = cluster_df[mask]
        idxs = sub.index.tolist()
        sizes = np.clip(hpo_count[idxs], 3, 20)
        nb = [newborn[i] for i in idxs]

        hover_text = [
            f"<b>{sub.iloc[k]['disease_name'] or sub.iloc[k]['disease_id']}</b><br>"
            f"ID: {sub.iloc[k]['disease_id']}<br>"
            f"Cluster: {cluster_id}<br>"
            f"Genes: {sub.iloc[k]['genes'] or 'N/A'}<br>"
            f"HPO count: {int(hpo_count[idxs[k]])}"
            f"{'<br>⚠️ NEWBORN ALERT (BabySeq Cat A)' if nb[k] else ''}"
            for k in range(len(sub))
        ]

        fig.add_trace(go.Scatter(
            x=sub["umap_x"].values,
            y=sub["umap_y"].values,
            mode="markers",
            name=f"Cluster {cluster_id}",
            marker=dict(
                size=sizes,
                color=color_map[cluster_id],
                opacity=0.8,
                line=dict(width=0.5, color="white"),
                symbol=["star" if n else "circle" for n in nb],
            ),
            text=hover_text,
            hovertemplate="%{text}<extra></extra>",
        ))

    fig.update_layout(
        title=dict(
            text="🧬 RareDx — Pediatric Disease Landscape (UMAP + Louvain)",
            font=dict(size=18, family="Inter, Arial"),
        ),
        paper_bgcolor="#0D1B2A",
        plot_bgcolor="#0D1B2A",
        font=dict(color="white"),
        legend=dict(
            title="Cluster",
            font=dict(size=10),
            bgcolor="rgba(0,0,0,0.3)",
        ),
        xaxis=dict(title="UMAP 1", showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(title="UMAP 2", showgrid=False, zeroline=False, showticklabels=False),
        width=1200,
        height=800,
        annotations=[
            dict(
                text=f"500 pediatric diseases | {n_clusters} clusters | ⭐ = BabySeq Category A (newborn alert)",
                showarrow=False, x=0.5, y=-0.04, xref="paper", yref="paper",
                font=dict(size=11, color="#aaaaaa"),
            )
        ],
    )

    if save_html:
        out_path = PLOTS / "plotly_disease_map.html"
        fig.write_html(str(out_path), include_plotlyjs="cdn")
        log.info(f"Disease map saved: {out_path}")

    return fig, cluster_df


if __name__ == "__main__":
    fig, clusters = build_disease_map()
    print(f"Done. Clusters: {clusters['cluster'].nunique()}")
    print(clusters.groupby("cluster").size().sort_values(ascending=False).head(10))
