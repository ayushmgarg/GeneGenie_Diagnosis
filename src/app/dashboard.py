# -*- coding: utf-8 -*-
"""
Rare Disease ML Dashboard - Streamlit App
4 tabs:
1. Disease Classifier (symptom -> disease prediction + SHAP)
2. Knowledge Graph Explorer (gene/disease subgraph + link prediction)
3. Symptom Retrieval Engine (text/HPO -> ranked genes+diseases)
4. Model Metrics (training stats, plots)
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import sys
import os
from pathlib import Path

# Setup paths
BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

st.set_page_config(
    page_title="Rare Disease ML System",
    page_icon="DNA",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODELS = BASE / "outputs" / "models"
PLOTS = BASE / "outputs" / "plots"
RESULTS = BASE / "outputs" / "results"
PROC = BASE / "data" / "processed"
EMBED = BASE / "data" / "embeddings"


# ─── Cached Loaders ────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner=False)
def load_classifier():
    try:
        rf = joblib.load(MODELS / "random_forest.pkl")
        xgb = joblib.load(MODELS / "xgboost.pkl")
        le = joblib.load(MODELS / "label_encoder.pkl")
        feature_names = np.load(MODELS / "hpo_feature_names.npy", allow_pickle=True).tolist()
        return rf, xgb, le, feature_names
    except Exception as e:
        return None, None, None, None


@st.cache_resource(show_spinner=False)
def load_shap_explainer():
    try:
        return joblib.load(MODELS / "shap_explainer.pkl")
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def load_master_table():
    p = PROC / "master_gene_disease_phenotype.csv"
    if p.exists():
        return pd.read_csv(p, low_memory=False)
    return None


@st.cache_resource(show_spinner=False)
def load_tfidf_index():
    p = MODELS / "tfidf_index.pkl"
    if p.exists():
        return joblib.load(p)
    return None


@st.cache_data(show_spinner=False)
def load_hpo_lookup():
    p = RESULTS / "hpo_lookup.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return {}


@st.cache_resource(show_spinner=False)
def load_graph_and_embeddings():
    import networkx as nx
    G = None
    # Try loading graph
    gpickle = MODELS / "knowledge_graph.gpickle"
    gpkl = MODELS / "knowledge_graph.pkl"
    if gpickle.exists():
        try:
            G = nx.read_gpickle(str(gpickle))
        except Exception:
            pass
    if G is None and gpkl.exists():
        try:
            G = joblib.load(gpkl)
        except Exception:
            pass

    # Load embeddings
    emb_path = EMBED / "node_embeddings.npy"
    nodes_path = EMBED / "node_list.json"
    embeddings = None
    nodes = None
    if emb_path.exists() and nodes_path.exists():
        vecs = np.load(emb_path)
        with open(nodes_path) as f:
            nodes = json.load(f)
        embeddings = dict(zip(nodes, vecs))

    lp_path = MODELS / "link_predictor.pkl"
    lp = joblib.load(lp_path) if lp_path.exists() else None
    return G, embeddings, nodes, lp


@st.cache_data(show_spinner=False)
def load_metrics():
    metrics = {}
    clf_path = RESULTS / "classifier_metrics.json"
    if clf_path.exists():
        with open(clf_path) as f:
            metrics["classifier"] = json.load(f)
    lp_path = RESULTS / "link_prediction_metrics.json"
    if lp_path.exists():
        with open(lp_path) as f:
            metrics["link_prediction"] = json.load(f)
    cv_path = RESULTS / "cv_scores.json"
    if cv_path.exists():
        with open(cv_path) as f:
            metrics["cross_validation"] = json.load(f)
    gs_path = RESULTS / "graph_stats.json"
    if gs_path.exists():
        with open(gs_path) as f:
            metrics["graph"] = json.load(f)
    return metrics


@st.cache_data(show_spinner=False)
def load_gene_disease_index():
    p = MODELS / "gene_disease_index.pkl"
    if p.exists():
        return joblib.load(p)
    return None


# ─── Prediction Functions ───────────────────────────────────────────────────────

def predict_disease_from_hpos(hpo_ids, rf, xgb, le, feature_names, top_k=10):
    """Predict disease from HPO ID list using ensemble."""
    if not hpo_ids or rf is None:
        return []
    x = np.zeros(len(feature_names), dtype=np.float32)
    matched = []
    for hpo in hpo_ids:
        if hpo in feature_names:
            x[feature_names.index(hpo)] = 1.0
            matched.append(hpo)
    if x.sum() == 0:
        return [], matched

    x = x.reshape(1, -1)
    rf_proba = rf.predict_proba(x)[0]
    xgb_proba = xgb.predict_proba(x)[0]
    ensemble_proba = (rf_proba + xgb_proba) / 2

    top_idx = np.argsort(ensemble_proba)[::-1][:top_k]
    results = []
    for rank, idx in enumerate(top_idx):
        results.append({
            "Rank": rank + 1,
            "Disease ID": le.classes_[idx],
            "Confidence (%)": round(float(ensemble_proba[idx]) * 100, 2),
            "RF Conf (%)": round(float(rf_proba[idx]) * 100, 2),
            "XGB Conf (%)": round(float(xgb_proba[idx]) * 100, 2),
        })
    return results, matched


def get_shap_explanation(explainer, hpo_ids, feature_names, le, top_k=15):
    """Get SHAP explanation for a prediction."""
    if explainer is None:
        return None
    import shap
    x = np.zeros(len(feature_names), dtype=np.float32)
    for hpo in hpo_ids:
        if hpo in feature_names:
            x[feature_names.index(hpo)] = 1.0
    x = x.reshape(1, -1)
    try:
        shap_values = explainer.shap_values(x)
        if isinstance(shap_values, list):
            # Sum absolute SHAP across all classes
            mean_abs = np.mean([np.abs(sv) for sv in shap_values], axis=0)[0]
        else:
            mean_abs = np.abs(shap_values).mean(axis=-1)[0] if shap_values.ndim == 3 else np.abs(shap_values)[0]
        top_idx = np.argsort(mean_abs)[::-1][:top_k]
        return [(feature_names[i], float(mean_abs[i])) for i in top_idx if mean_abs[i] > 0]
    except Exception as e:
        return None


# ─── Tab 1: Disease Classifier ─────────────────────────────────────────────────

def tab_classifier():
    st.header("Rare Disease Classifier")
    st.markdown(
        "Enter HPO (Human Phenotype Ontology) IDs to predict the most likely rare disease. "
        "The system uses an ensemble of Random Forest + XGBoost trained on 7,500 gene-disease profiles."
    )

    rf, xgb, le, feature_names = load_classifier()
    hpo_lookup = load_hpo_lookup()

    if rf is None:
        st.error("Classifier models not found. Run: `python run_pipeline.py --module classifier`")
        return

    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("Input Symptoms (HPO IDs)")
        example_hpos = "HP:0001083\nHP:0001182\nHP:0002705\nHP:0001166\nHP:0001763"
        hpo_input = st.text_area(
            "Enter HPO IDs (one per line):",
            value=example_hpos,
            height=150,
            help="Format: HP:XXXXXXX — one per line",
        )

        # Search by name
        st.markdown("**Or search HPO terms by name:**")
        search_term = st.text_input("Type symptom name:", placeholder="e.g. aortic, lens, seizure")
        if search_term and len(search_term) >= 3:
            matches = [(hid, name) for hid, name in hpo_lookup.items()
                       if search_term.lower() in name.lower()][:20]
            if matches:
                st.dataframe(pd.DataFrame(matches, columns=["HPO ID", "HPO Name"]), use_container_width=True)

        top_k = st.slider("Show top N predictions:", 5, 20, 10)

        predict_btn = st.button("Predict Disease", type="primary", use_container_width=True)

    with col2:
        st.subheader("Quick Examples")
        examples = {
            "Marfan syndrome": "HP:0001083\nHP:0001182\nHP:0002705\nHP:0001166\nHP:0001763",
            "Down syndrome": "HP:0000486\nHP:0001250\nHP:0001252\nHP:0000028\nHP:0000023",
            "Huntington disease": "HP:0002072\nHP:0000739\nHP:0000716\nHP:0002354\nHP:0001300",
            "Phenylketonuria": "HP:0001249\nHP:0001252\nHP:0000365\nHP:0002353\nHP:0001263",
        }
        for name, hpos in examples.items():
            if st.button(f"{name}", key=f"ex_{name}", use_container_width=True):
                st.session_state["hpo_input"] = hpos

    if predict_btn and hpo_input.strip():
        hpo_ids = [h.strip() for h in hpo_input.strip().split("\n") if h.strip()]
        result = predict_disease_from_hpos(hpo_ids, rf, xgb, le, feature_names, top_k)

        if isinstance(result, tuple) and len(result) == 2:
            predictions, matched = result
        else:
            predictions, matched = result, []

        if not predictions:
            st.warning(f"No HPO IDs matched the training vocabulary. Check your HPO IDs.")
            st.info(f"Available features include: {feature_names[:5]}...")
            return

        st.success(f"Matched {len(matched)}/{len(hpo_ids)} HPO terms")

        # Show predictions
        st.subheader("Top Disease Predictions")
        df_pred = pd.DataFrame(predictions)
        st.dataframe(df_pred.style.background_gradient(subset=["Confidence (%)"], cmap="Blues"),
                     use_container_width=True)

        # Bar chart
        import plotly.express as px
        fig = px.bar(
            df_pred.head(10), x="Disease ID", y="Confidence (%)",
            title="Prediction Confidence by Disease",
            color="Confidence (%)", color_continuous_scale="Blues",
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

        # SHAP explanation
        st.subheader("SHAP Feature Importance (Why this prediction?)")
        explainer = load_shap_explainer()
        shap_result = get_shap_explanation(explainer, hpo_ids, feature_names, le)
        if shap_result:
            shap_df = pd.DataFrame(shap_result, columns=["HPO Feature", "Mean |SHAP|"])
            shap_df["HPO Name"] = shap_df["HPO Feature"].map(hpo_lookup).fillna("Unknown")
            fig2 = px.bar(
                shap_df, x="Mean |SHAP|", y="HPO Feature",
                orientation="h", title="Most Influential HPO Features (SHAP)",
                color="Mean |SHAP|", color_continuous_scale="Reds",
                hover_data=["HPO Name"],
            )
            fig2.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("SHAP explainer not available. Run full pipeline to enable.")


# ─── Tab 2: Knowledge Graph ─────────────────────────────────────────────────────

def tab_knowledge_graph():
    st.header("Knowledge Graph Explorer")
    st.markdown(
        "Explore gene-disease-phenotype connections and discover missing links "
        "using Node2Vec embeddings + logistic regression."
    )

    G, embeddings, nodes, lp = load_graph_and_embeddings()
    master = load_master_table()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.subheader("Search")
        query_node = st.text_input("Gene or Disease ID:", value="FBN1",
                                   placeholder="e.g. FBN1, BRCA1, OMIM:154700")
        n_hops = st.selectbox("Subgraph depth:", [1, 2], index=0)
        show_lp = st.checkbox("Show predicted missing links", value=True)
        top_k_lp = st.slider("Top K missing links:", 5, 20, 10)
        explore_btn = st.button("Explore Graph", type="primary", use_container_width=True)

    with col2:
        st.subheader("Graph Statistics")
        if G is not None:
            metrics = {
                "Nodes": G.number_of_nodes(),
                "Edges": G.number_of_edges(),
                "Avg Degree": f"{np.mean([d for _, d in G.degree()]):.1f}",
                "Components": len(list(__import__("networkx").connected_components(G))),
            }
            cols = st.columns(4)
            for i, (k, v) in enumerate(metrics.items()):
                cols[i].metric(k, v)
        else:
            st.info("Knowledge graph not built yet. Run: `python run_pipeline.py --module kg`")

    if explore_btn and query_node.strip():
        import networkx as nx
        query = query_node.strip().upper()

        if G is None:
            st.error("Graph not loaded.")
            # Fallback: show from master table
            if master is not None:
                st.subheader("Connections from Master Table")
                gene_rows = master[master.get("gene_symbol", pd.Series()).str.upper() == query] \
                    if "gene_symbol" in master.columns else pd.DataFrame()
                if not gene_rows.empty:
                    st.dataframe(gene_rows.head(50), use_container_width=True)
            return

        # Find node in graph (case-insensitive)
        found_node = None
        for n in G.nodes():
            if str(n).upper() == query:
                found_node = n
                break
        if found_node is None:
            # partial match
            matches = [n for n in G.nodes() if query in str(n).upper()]
            if matches:
                found_node = matches[0]
                st.info(f"Exact match not found. Showing: {found_node}")
            else:
                st.warning(f"Node '{query}' not found in graph. Try a gene symbol like FBN1, BRCA1, etc.")
                return

        # Extract subgraph
        if n_hops == 1:
            neighbors = list(G.neighbors(found_node))
            sub_nodes = [found_node] + neighbors[:100]
        else:
            sub_nodes = set([found_node])
            for n1 in list(G.neighbors(found_node))[:30]:
                sub_nodes.add(n1)
                for n2 in list(G.neighbors(n1))[:10]:
                    sub_nodes.add(n2)
            sub_nodes = list(sub_nodes)

        G_sub = G.subgraph(sub_nodes)
        st.success(f"Subgraph: {G_sub.number_of_nodes()} nodes, {G_sub.number_of_edges()} edges")

        # Visualize with plotly
        import plotly.graph_objects as go
        pos = nx.spring_layout(G_sub, seed=42, k=0.5)
        edge_x, edge_y = [], []
        for e in G_sub.edges():
            x0, y0 = pos[e[0]]
            x1, y1 = pos[e[1]]
            edge_x += [x0, x1, None]
            edge_y += [y0, y1, None]

        node_x = [pos[n][0] for n in G_sub.nodes()]
        node_y = [pos[n][1] for n in G_sub.nodes()]
        node_labels = [str(n)[:30] for n in G_sub.nodes()]
        node_colors = []
        for n in G_sub.nodes():
            if str(n).startswith("HP:"):
                node_colors.append("orange")
            elif "ORPHA" in str(n).upper() or "OMIM" in str(n).upper():
                node_colors.append("red")
            elif str(n) == found_node:
                node_colors.append("gold")
            else:
                node_colors.append("steelblue")

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=edge_x, y=edge_y, mode="lines",
                                  line=dict(width=0.5, color="#888"), hoverinfo="none"))
        fig.add_trace(go.Scatter(x=node_x, y=node_y, mode="markers+text",
                                  marker=dict(size=10, color=node_colors, line=dict(width=1)),
                                  text=node_labels, textposition="top center",
                                  textfont=dict(size=8), hoverinfo="text",
                                  name="Nodes"))
        fig.update_layout(
            title=f"Knowledge Graph subgraph around {found_node}",
            showlegend=False, hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            height=600,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        )
        st.plotly_chart(fig, use_container_width=True)

        # Legend
        col_l, col_m, col_r = st.columns(3)
        col_l.markdown("Gold: Query node")
        col_m.markdown("Orange: Phenotype (HP:)")
        col_r.markdown("Red: Disease | Blue: Gene")

        # Neighbor table
        st.subheader(f"Direct Neighbors of {found_node}")
        neighbors_data = []
        for n in G.neighbors(found_node):
            edge_data = G.get_edge_data(found_node, n)
            etype = edge_data.get("edge_type", "unknown") if edge_data else "unknown"
            neighbors_data.append({"Node": str(n), "Edge Type": etype})
        if neighbors_data:
            st.dataframe(pd.DataFrame(neighbors_data).head(50), use_container_width=True)

        # Missing link prediction
        if show_lp and lp is not None and embeddings is not None:
            st.subheader("Predicted Missing Links (Link Prediction)")
            from retrieval.retrieval_engine import retrieve_similar_genes
            from knowledge_graph.build_graph import predict_missing_links

            clf_lp = lp.get("clf")
            scaler_lp = lp.get("scaler")
            if clf_lp and scaler_lp:
                missing = predict_missing_links(found_node, G, embeddings, clf_lp, scaler_lp, top_k=top_k_lp)
                if missing:
                    df_missing = pd.DataFrame(missing, columns=["Predicted Connection", "Score"])
                    df_missing["Score"] = df_missing["Score"].round(4)
                    st.dataframe(df_missing, use_container_width=True)
                else:
                    st.info("No embedding found for this node for link prediction.")
        elif lp is None:
            st.info("Link predictor not built yet. Run `python run_pipeline.py --module kg`")


# ─── Tab 3: Retrieval Engine ────────────────────────────────────────────────────

def tab_retrieval():
    st.header("Phenotype-to-Gene Retrieval Engine")
    st.markdown(
        "Type symptoms in natural language or enter HPO IDs. "
        "Returns ranked genes and diseases by relevance using TF-IDF + cosine similarity."
    )

    tfidf_idx = load_tfidf_index()
    master = load_master_table()
    hpo_lookup = load_hpo_lookup()
    gdi = load_gene_disease_index()

    col1, col2 = st.columns([1, 1])
    with col1:
        st.subheader("Text Search")
        text_query = st.text_area(
            "Describe symptoms:",
            value="aortic root dilatation lens dislocation tall stature scoliosis",
            height=100,
        )
        text_btn = st.button("Search by Text", type="primary")

    with col2:
        st.subheader("HPO ID Search")
        hpo_query = st.text_area(
            "Enter HPO IDs (one per line):",
            value="HP:0001083\nHP:0001182\nHP:0002705",
            height=100,
        )
        hpo_btn = st.button("Search by HPO IDs", type="secondary")

    top_n = st.slider("Number of results:", 5, 50, 20)

    if text_btn and text_query.strip():
        if tfidf_idx is None:
            st.error("Retrieval index not built. Run: `python run_pipeline.py --module retrieval`")
            # Fallback: simple keyword search on master table
            if master is not None:
                st.subheader("Fallback: Keyword search on master table")
                keywords = text_query.lower().split()
                if "hpo_name" in master.columns:
                    mask = master["hpo_name"].str.lower().str.contains(
                        "|".join(keywords[:3]), na=False
                    )
                    results = master[mask][["gene_symbol", "hpo_id", "hpo_name", "disease_id"]].drop_duplicates().head(top_n)
                    st.dataframe(results, use_container_width=True)
            return

        from retrieval.retrieval_engine import retrieve_by_text
        results = retrieve_by_text(
            text_query, tfidf_idx["vectorizer"], tfidf_idx["matrix"],
            tfidf_idx["ids"], tfidf_idx["types"], top_k=top_n
        )

        if results:
            df_r = pd.DataFrame(results)
            genes = df_r[df_r["type"] == "gene"].head(top_n // 2)
            diseases = df_r[df_r["type"] == "disease"].head(top_n // 2)

            col_g, col_d = st.columns(2)
            with col_g:
                st.subheader("Top Genes")
                if not genes.empty:
                    import plotly.express as px
                    fig = px.bar(genes.head(15), x="score", y="id", orientation="h",
                                 title="Gene Relevance Scores", color="score",
                                 color_continuous_scale="Blues")
                    fig.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig, use_container_width=True)
                    st.dataframe(genes[["id", "score"]].rename(columns={"id": "Gene", "score": "Score"}),
                                 use_container_width=True)

            with col_d:
                st.subheader("Top Diseases")
                if not diseases.empty:
                    import plotly.express as px
                    fig2 = px.bar(diseases.head(15), x="score", y="id", orientation="h",
                                  title="Disease Relevance Scores", color="score",
                                  color_continuous_scale="Reds")
                    fig2.update_layout(yaxis=dict(autorange="reversed"))
                    st.plotly_chart(fig2, use_container_width=True)
                    st.dataframe(diseases[["id", "score"]].rename(columns={"id": "Disease", "score": "Score"}),
                                 use_container_width=True)
        else:
            st.warning("No results found.")

    if hpo_btn and hpo_query.strip():
        hpo_ids = [h.strip() for h in hpo_query.strip().split("\n") if h.strip()]
        st.info(f"Searching for {len(hpo_ids)} HPO terms: {', '.join(hpo_ids)}")

        if master is not None and "hpo_id" in master.columns:
            from retrieval.retrieval_engine import retrieve_by_hpo_ids
            results = retrieve_by_hpo_ids(hpo_ids, master, top_k=top_n)

            genes_r = [r for r in results if r["type"] == "gene"]
            diseases_r = [r for r in results if r["type"] == "disease"]

            col_g, col_d = st.columns(2)
            with col_g:
                st.subheader(f"Relevant Genes ({len(genes_r)})")
                if genes_r:
                    df_g = pd.DataFrame(genes_r).rename(columns={"id": "Gene", "score": "HPO Match Score"})
                    df_g["HPO Match Score"] = df_g["HPO Match Score"].round(4)
                    st.dataframe(df_g[["Gene", "HPO Match Score"]], use_container_width=True)

                    # Show diseases linked to top gene
                    if gdi and genes_r:
                        top_gene = genes_r[0]["id"]
                        linked = gdi.get("gene_to_diseases", {}).get(top_gene, [])
                        if linked:
                            st.markdown(f"**Diseases linked to {top_gene}:** {', '.join(linked[:10])}")

            with col_d:
                st.subheader(f"Relevant Diseases ({len(diseases_r)})")
                if diseases_r:
                    df_d = pd.DataFrame(diseases_r).rename(columns={"id": "Disease ID", "score": "HPO Match Score"})
                    df_d["HPO Match Score"] = df_d["HPO Match Score"].round(4)
                    st.dataframe(df_d[["Disease ID", "HPO Match Score"]], use_container_width=True)
        else:
            st.error("Master table not loaded.")

        # Show HPO name lookup
        st.subheader("HPO Term Details")
        hpo_details = [(hpo, hpo_lookup.get(hpo, "Not found")) for hpo in hpo_ids]
        st.dataframe(pd.DataFrame(hpo_details, columns=["HPO ID", "Name"]), use_container_width=True)


# ─── Tab 4: Model Metrics ───────────────────────────────────────────────────────

def tab_metrics():
    st.header("Model Performance Metrics")
    metrics = load_metrics()

    if not metrics:
        st.warning("No metrics found. Run the full pipeline first.")
        return

    # Classifier metrics
    if "classifier" in metrics:
        st.subheader("Disease Classifier Performance")
        clf_data = metrics["classifier"]
        cols = st.columns(len(clf_data))
        for i, m in enumerate(clf_data):
            with cols[i]:
                model_name = m.get("model", f"Model {i+1}")
                st.metric(f"{model_name}", "")
                acc = m.get("accuracy")
                f1 = m.get("f1_macro")
                top5 = m.get("top5_accuracy")
                if acc:
                    st.metric("Accuracy", f"{acc*100:.2f}%")
                if f1:
                    st.metric("F1-Macro", f"{f1*100:.2f}%")
                if top5:
                    st.metric("Top-5 Accuracy", f"{top5*100:.2f}%")

    # Cross-validation
    if "cross_validation" in metrics:
        cv = metrics["cross_validation"]
        st.subheader("3-Fold Cross-Validation (Random Forest)")
        col_a, col_b = st.columns(2)
        col_a.metric("CV F1-Macro (mean)", f"{cv['cv_f1_macro_mean']*100:.2f}%")
        col_b.metric("CV F1-Macro (std)", f"{cv['cv_f1_macro_std']*100:.4f}%")
        if "cv_scores" in cv:
            import plotly.express as px
            fig = px.bar(x=[f"Fold {i+1}" for i in range(len(cv["cv_scores"]))],
                         y=cv["cv_scores"], title="Cross-Validation F1 Scores",
                         color=cv["cv_scores"], color_continuous_scale="Blues")
            st.plotly_chart(fig, use_container_width=True)

    # Link prediction
    if "link_prediction" in metrics:
        lp = metrics["link_prediction"]
        st.subheader("Link Prediction Performance")
        c1, c2, c3 = st.columns(3)
        c1.metric("AUC-ROC", f"{lp.get('auc_roc', 0)*100:.2f}%")
        c2.metric("Avg Precision", f"{lp.get('avg_precision', 0)*100:.2f}%")
        c3.metric("F1 Score", f"{lp.get('f1', 0)*100:.2f}%")

    # Graph stats
    if "graph" in metrics:
        gs = metrics["graph"]
        st.subheader("Knowledge Graph Statistics")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Nodes", f"{gs.get('n_nodes', 0):,}")
        c2.metric("Edges", f"{gs.get('n_edges', 0):,}")
        c3.metric("Avg Degree", f"{gs.get('avg_degree', 0):.1f}")
        c4.metric("Components", f"{gs.get('n_components', 0):,}")

    # Saved plots
    st.subheader("Training Plots")
    plot_cols = st.columns(2)
    plots = [
        ("rf_feature_importance.png", "Feature Importance (RF)"),
        ("confusion_matrix.png", "Confusion Matrix"),
        ("link_prediction_roc.png", "Link Prediction ROC"),
        ("graph_degree_distribution.png", "Graph Degree Distribution"),
    ]
    for i, (fname, title) in enumerate(plots):
        p = PLOTS / fname
        if p.exists():
            from PIL import Image
            with plot_cols[i % 2]:
                st.markdown(f"**{title}**")
                img = Image.open(p)
                st.image(img, use_container_width=True)


# ─── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.title("Rare Disease ML")
        st.markdown("---")
        st.markdown("**Data Sources:**")
        st.markdown("- HPO (Human Phenotype Ontology)")
        st.markdown("- OMIM gene-disease maps")
        st.markdown("- Orphanet associations")
        st.markdown("- ClinVar variants")
        st.markdown("- 293K gene-phenotype pairs")
        st.markdown("---")
        st.markdown("**Models:**")
        st.markdown("- Random Forest (n=300)")
        st.markdown("- XGBoost (n=200)")
        st.markdown("- Node2Vec + LR (link pred)")
        st.markdown("- TF-IDF (retrieval)")
        st.markdown("---")

        rf, _, _, _ = load_classifier()
        kg_loaded = (MODELS / "knowledge_graph.pkl").exists() or (MODELS / "knowledge_graph.gpickle").exists()
        tfidf_loaded = (MODELS / "tfidf_index.pkl").exists()

        st.markdown("**Pipeline Status:**")
        st.markdown(f"{'OK' if rf else 'NOT RUN'} Classifier")
        st.markdown(f"{'OK' if kg_loaded else 'NOT RUN'} Knowledge Graph")
        st.markdown(f"{'OK' if tfidf_loaded else 'NOT RUN'} Retrieval Engine")

        st.markdown("---")
        st.markdown("**Run pipeline:**")
        st.code("python run_pipeline.py")
        st.markdown("---")
        st.caption("MPSTME ML Project 2026")


# ─── Main ────────────────────────────────────────────────────────────────────────

def main():
    sidebar()
    tab1, tab2, tab3, tab4 = st.tabs([
        "Disease Classifier",
        "Knowledge Graph",
        "Retrieval Engine",
        "Model Metrics",
    ])
    with tab1:
        tab_classifier()
    with tab2:
        tab_knowledge_graph()
    with tab3:
        tab_retrieval()
    with tab4:
        tab_metrics()


if __name__ == "__main__":
    main()
