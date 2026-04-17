"""
Knowledge Graph Module — Module 2
Builds gene-disease-phenotype KG, runs Node2Vec, trains link predictor.
"""

import pandas as pd
import numpy as np
import networkx as nx
import joblib
import logging
import json
import warnings
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
EMBED = BASE / "data" / "embeddings"
MODELS = BASE / "outputs" / "models"
PLOTS = BASE / "outputs" / "plots"
RESULTS = BASE / "outputs" / "results"

for d in [EMBED, MODELS, PLOTS, RESULTS]:
    d.mkdir(parents=True, exist_ok=True)


def load_edges(max_edges=150000):
    """Load graph edges from processed data."""
    path = PROC / "graph_edges.csv"
    if not path.exists():
        raise FileNotFoundError(f"graph_edges.csv not found. Run merge_datasets.py first.")
    edges = pd.read_csv(path)
    log.info(f"Raw edges: {len(edges):,}")

    # Filter to meaningful edges — gene-disease and gene-phenotype
    if "edge_type" in edges.columns:
        priority = edges[edges["edge_type"].isin(["gene_disease", "gene_phenotype", "disease_phenotype"])]
        other = edges[~edges["edge_type"].isin(["gene_disease", "gene_phenotype", "disease_phenotype"])]
        edges = pd.concat([priority, other.sample(min(len(other), 20000), random_state=42)], ignore_index=True)

    edges = edges.dropna(subset=["source_node", "target_node"])
    edges = edges.drop_duplicates(subset=["source_node", "target_node"])
    edges = edges.head(max_edges)
    log.info(f"Filtered edges: {len(edges):,}")
    return edges


def build_networkx_graph(edges):
    """Build NetworkX graph from edge list."""
    G = nx.Graph()
    for _, row in edges.iterrows():
        src = str(row["source_node"])
        tgt = str(row["target_node"])
        etype = row.get("edge_type", "relation") if "edge_type" in row.index else "relation"
        G.add_edge(src, tgt, edge_type=etype)

    log.info(f"Graph: {G.number_of_nodes():,} nodes, {G.number_of_edges():,} edges")
    log.info(f"  Connected components: {nx.number_connected_components(G)}")

    # Node type assignment
    node_types = {}
    for _, row in edges.iterrows():
        src, tgt = str(row["source_node"]), str(row["target_node"])
        etype = row.get("edge_type", "") if "edge_type" in row.index else ""
        if "gene_disease" in etype or "gene_phenotype" in etype:
            node_types[src] = "gene"
        if "disease" in tgt.lower() or "omim" in tgt.lower() or "orpha" in tgt.lower():
            node_types[tgt] = "disease"
        if tgt.startswith("HP:"):
            node_types[tgt] = "phenotype"
        if src.startswith("HP:"):
            node_types[src] = "phenotype"

    nx.set_node_attributes(G, node_types, "node_type")
    return G


def compute_graph_stats(G):
    """Compute and save graph statistics."""
    stats = {
        "n_nodes": G.number_of_nodes(),
        "n_edges": G.number_of_edges(),
        "n_components": nx.number_connected_components(G),
        "density": nx.density(G),
        "avg_degree": np.mean([d for _, d in G.degree()]),
        "max_degree": max(d for _, d in G.degree()),
    }

    # Degree distribution
    degrees = [d for _, d in G.degree()]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].hist(degrees, bins=50, color="steelblue", edgecolor="white", log=True)
    axes[0].set_title("Degree Distribution (log scale)")
    axes[0].set_xlabel("Degree")
    axes[0].set_ylabel("Count (log)")

    # Top hubs
    top_nodes = sorted(G.degree(), key=lambda x: x[1], reverse=True)[:20]
    names = [n[:25] for n, _ in top_nodes]
    degs = [d for _, d in top_nodes]
    axes[1].barh(names, degs, color="coral")
    axes[1].set_title("Top 20 Hub Nodes")
    axes[1].set_xlabel("Degree")
    plt.tight_layout()
    plt.savefig(PLOTS / "graph_degree_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    log.info(f"Graph stats: {stats}")
    with open(RESULTS / "graph_stats.json", "w") as f:
        json.dump(stats, f, indent=2)

    return stats


def node2vec_embeddings(G, embed_dim=64, walk_length=30, num_walks=10, p=1.0, q=1.0):
    """
    Node2Vec using random walks + Word2Vec.
    Falls back to spectral embeddings if node2vec package unavailable.
    """
    try:
        from node2vec import Node2Vec
        log.info(f"Running Node2Vec (dim={embed_dim}, walks={num_walks}, length={walk_length})...")
        n2v = Node2Vec(
            G, dimensions=embed_dim, walk_length=walk_length,
            num_walks=num_walks, p=p, q=q, workers=4, quiet=True
        )
        model = n2v.fit(window=10, min_count=1, batch_words=4)
        embeddings = {node: model.wv[node] for node in G.nodes() if node in model.wv}
        log.info(f"Node2Vec embeddings: {len(embeddings)} nodes")
        return embeddings, model
    except ImportError:
        log.warning("node2vec not installed — using spectral embeddings fallback")
        return spectral_embedding_fallback(G, embed_dim)
    except Exception as e:
        log.warning(f"Node2Vec failed ({e}) — using spectral fallback")
        return spectral_embedding_fallback(G, embed_dim)


def spectral_embedding_fallback(G, embed_dim=64):
    """Spectral/SVD-based node embeddings as fallback."""
    log.info("Computing spectral embeddings via adjacency SVD...")
    # Use largest connected component
    largest_cc = max(nx.connected_components(G), key=len)
    G_sub = G.subgraph(largest_cc).copy()
    nodes = list(G_sub.nodes())
    n = len(nodes)

    if n > 5000:
        # Sample subgraph for memory
        sample_nodes = np.random.choice(nodes, 5000, replace=False)
        G_sub = G_sub.subgraph(sample_nodes).copy()
        nodes = list(G_sub.nodes())
        n = len(nodes)

    log.info(f"Spectral embedding on {n} nodes...")
    A = nx.adjacency_matrix(G_sub, nodelist=nodes).astype(np.float32).toarray()
    # Degree normalization
    d = A.sum(axis=1, keepdims=True)
    d[d == 0] = 1
    A_norm = A / np.sqrt(d) / np.sqrt(d.T)

    from scipy.sparse.linalg import eigsh
    from scipy.sparse import csr_matrix
    k = min(embed_dim, n - 2)
    vals, vecs = eigsh(csr_matrix(A_norm), k=k, which="LM")
    embeddings = {node: vecs[i] for i, node in enumerate(nodes)}
    log.info(f"Spectral embeddings: {len(embeddings)} nodes, dim={k}")
    return embeddings, None


def save_embeddings(embeddings):
    """Save node embeddings to disk."""
    nodes = list(embeddings.keys())
    vecs = np.array([embeddings[n] for n in nodes])
    np.save(EMBED / "node_embeddings.npy", vecs)
    with open(EMBED / "node_list.json", "w") as f:
        json.dump(nodes, f)
    df = pd.DataFrame(vecs, index=nodes)
    df.to_csv(EMBED / "node_embeddings.csv")
    log.info(f"Saved embeddings: {vecs.shape}")
    return nodes, vecs


def create_link_prediction_dataset(G, embeddings, neg_ratio=1):
    """
    Create positive (existing) + negative (non-existing) edge samples.
    Feature: Hadamard product of node embeddings.
    """
    log.info("Creating link prediction dataset...")
    embed_nodes = set(embeddings.keys())

    # Positive samples — existing edges
    pos_edges = [(u, v) for u, v in G.edges() if u in embed_nodes and v in embed_nodes]
    np.random.shuffle(pos_edges)
    pos_edges = pos_edges[:min(len(pos_edges), 30000)]

    # Negative samples — non-existing edges
    node_list = list(embed_nodes)
    neg_edges = set()
    attempts = 0
    while len(neg_edges) < len(pos_edges) * neg_ratio and attempts < len(pos_edges) * 10:
        u = np.random.choice(node_list)
        v = np.random.choice(node_list)
        if u != v and not G.has_edge(u, v) and (u, v) not in neg_edges:
            neg_edges.add((u, v))
        attempts += 1

    neg_edges = list(neg_edges)
    log.info(f"Pos edges: {len(pos_edges)}, Neg edges: {len(neg_edges)}")

    # Feature extraction: Hadamard product
    def hadamard(u, v):
        return embeddings[u] * embeddings[v]

    X_pos = np.array([hadamard(u, v) for u, v in pos_edges])
    X_neg = np.array([hadamard(u, v) for u, v in neg_edges])

    X = np.vstack([X_pos, X_neg])
    y = np.hstack([np.ones(len(X_pos)), np.zeros(len(X_neg))])

    log.info(f"Link prediction dataset: {X.shape}, pos_ratio={y.mean():.3f}")
    return X, y, pos_edges, neg_edges


def train_link_predictor(X, y):
    """Train logistic regression for link prediction."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    log.info("Training link prediction model...")
    clf = LogisticRegression(max_iter=1000, C=1.0, random_state=42, n_jobs=-1)
    clf.fit(X_train, y_train)

    y_proba = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)

    auc = roc_auc_score(y_test, y_proba)
    ap = average_precision_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    log.info(f"Link Prediction — AUC: {auc:.4f}, AP: {ap:.4f}, F1: {f1:.4f}")

    results = {"auc_roc": float(auc), "avg_precision": float(ap), "f1": float(f1)}
    with open(RESULTS / "link_prediction_metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # Also try RF for comparison
    rf_clf = RFC(n_estimators=100, random_state=42, n_jobs=-1)
    rf_clf.fit(X_train, y_train)
    rf_auc = roc_auc_score(y_test, rf_clf.predict_proba(X_test)[:, 1])
    log.info(f"RF Link Predictor AUC: {rf_auc:.4f}")

    # Plot ROC
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"LR (AUC={auc:.3f})", color="steelblue")
    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Link Prediction ROC Curve")
    ax.legend()
    plt.tight_layout()
    plt.savefig(PLOTS / "link_prediction_roc.png", dpi=150)
    plt.close()

    joblib.dump({"clf": clf, "scaler": scaler}, MODELS / "link_predictor.pkl")
    return clf, scaler, results


def predict_missing_links(gene, G, embeddings, clf, scaler, top_k=10):
    """
    Given a gene, predict which diseases/phenotypes it might link to (missing links).
    """
    if gene not in embeddings:
        return []
    gene_emb = embeddings[gene]
    current_neighbors = set(G.neighbors(gene)) if gene in G else set()

    candidates = []
    for node, emb in embeddings.items():
        if node == gene or node in current_neighbors:
            continue
        feat = (gene_emb * emb).reshape(1, -1)
        feat_scaled = scaler.transform(feat)
        prob = clf.predict_proba(feat_scaled)[0][1]
        candidates.append((node, prob))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def run_knowledge_graph(embed_dim=64, max_edges=100000):
    log.info("=== Knowledge Graph + Link Prediction ===")

    edges = load_edges(max_edges=max_edges)
    G = build_networkx_graph(edges)
    stats = compute_graph_stats(G)

    log.info("=== Running Node2Vec ===")
    embeddings, n2v_model = node2vec_embeddings(G, embed_dim=embed_dim)

    nodes, vecs = save_embeddings(embeddings)

    log.info("=== Training Link Predictor ===")
    X, y, pos_edges, neg_edges = create_link_prediction_dataset(G, embeddings)
    clf, scaler, lp_results = train_link_predictor(X, y)

    # Save graph
    nx.write_gpickle(G, MODELS / "knowledge_graph.gpickle") if hasattr(nx, "write_gpickle") else joblib.dump(G, MODELS / "knowledge_graph.pkl")

    log.info("=== KG module complete ===")
    return G, embeddings, clf, scaler


if __name__ == "__main__":
    G, embeddings, clf, scaler = run_knowledge_graph(embed_dim=64, max_edges=100000)
    log.info("Graph and link predictor saved")
