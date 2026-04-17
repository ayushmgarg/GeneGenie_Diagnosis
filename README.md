# GeneGenie — Rare Disease Intelligence System

> **Rub the lamp, skip the diagnostic odyssey.**

ML pipeline for rare disease differential diagnosis, gene prioritisation, and clinical decision support. Integrates 8 biomedical databases (5.7 GB of genomic and phenotypic data).

---

## Key Results

| Component | Metric | Value |
|-----------|--------|-------|
| Classifier V4 (RF) | Within-set Accuracy | **98.94%** |
| Classifier V4 (RF) | Top-5 Accuracy | **~99.5%** |
| Classifier V4 (XGBoost) | Accuracy | **96.72%** |
| Knowledge Graph (Link Prediction) | AUC-ROC | **99.05%** |
| Disease Coverage (Retrieval) | All rare diseases | **12,671** |
| Classifier Scope | Top HPO-annotated | **500** |
| Knowledge Graph | Nodes / Edges | **9,801 / 100K** |
| Total Data Processed | - | **5.7 GB** |

---

## System Architecture

```
Patient Symptoms (HPO IDs)
         |
         +-> [Classifier V4] -----> Ranked differential across 500 diseases
         |   RF + XGBoost + LR      SHAP explanations per prediction
         |   426 features:          IBA panel activation (novel)
         |
         +-> [Similar Disease Engine] -> Jaccard + penetrance similarity
         |   HPO set overlap           All 12,671 diseases
         |
         +-> [TF-IDF Retrieval] ----> Bigram full-text search
                                      All 12,671 diseases indexed

Gene of interest
         |
         +-> [Knowledge Graph] -----> Link prediction (AUC 99%)
             9,801 nodes, 100K edges  Missing gene-disease connections
             Spectral SVD embeddings
```

---

## Data Sources

| # | Dataset | Records | Use |
|---|---------|---------|-----|
| 1 | HPO genes-to-phenotype (JAX) | 293K rows | Gene-HPO-disease triplets |
| 2 | gene2phenotype | 4,573 rows | Gold-standard links |
| 3 | Orphanet (XML + CSV) | 11,456 diseases | Disease classification |
| 4 | OMIM (genemap2 + morbidmap) | 26,724 entries | Gene-disease phenotypes |
| 5 | ClinVar variant_summary.txt | 8.8M variants (3.7 GB) | Pathogenicity scores |
| 6 | gnomAD v4.1 constraint | 18,182 genes | pLI, LOEUF scores |
| 7 | BabySeq + ACMG SF v3.2 | 71 + 500 genes | Gene actionability |
| 8 | NHS PanelApp (10 IBA panels) | 517 green-tier genes | IBA panel features |
| 9 | Biomedical KG (PrimeKG) | 8.1M edges | Biological network |
| 10 | gene_attribute_matrix | 4,554 x 6,178 | Gene phenotype classes |

---

## Novel Contributions

1. **IBA Panel Activation Scores** — Maps patient HPO terms to 13 clinical IBA panels.
   Creates interpretable meta-features: each panel gets a 0-1 match score.

2. **Gene Actionability Index V2** — Integrates BabySeq evidence grade + ACMG SF v3.2
   flag + gnomAD pLI constraint + NHS PanelApp green-tier membership.

3. **HPO Jaccard + Penetrance Similarity** — Disease similarity combining phenotype
   overlap (Jaccard on HPO sets) with evidence quality and penetrance correction.

4. **Unified Multi-source Pipeline** — Single pipeline integrating 10 databases into
   a coherent ML feature set across classification, KG, and retrieval.

---

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run full pipeline
python run_pipeline.py

# Skip 3.7 GB ClinVar for development
python run_pipeline.py --skip-large

# Run specific module
python run_pipeline.py --module classifier_v4

# Launch dashboard
python -m streamlit run src/app/dashboard_v2.py
```

### Raw Data

Large files must be downloaded separately into `data/raw/`:

| File | Source |
|------|--------|
| `genes_to_phenotype_ontology.csv` | [HPO JAX](https://hpo.jax.org/data/annotations) |
| `variant_summary.txt` | [ClinVar FTP](https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/) |
| `gnomad_constraint.tsv` | [gnomAD v4.1 downloads](https://gnomad.broadinstitute.org/downloads) |
| `gene_attribute_matrix.csv` | [Harmonizome](https://maayanlab.cloud/Harmonizome) |
| `kg.csv` | [PrimeKG](https://github.com/mims-harvard/PrimeKG) |
| `en_product1.xml`, `en_product6.xml` | [Orphadata](https://www.orphadata.com) |
| `Homo_sapiens.gene_info` | [NCBI gene FTP](https://ftp.ncbi.nlm.nih.gov/gene/DATA/) |
| `genemap2.txt`, `morbidmap.txt` | [OMIM downloads](https://omim.org/downloads) |

---

## Project Structure

```
geniegenie/
+-- run_pipeline.py           # Master pipeline (all steps)
+-- generate_report.py        # Generates Word project report
+-- requirements.txt
+-- .gitignore
|
+-- src/
|   +-- preprocessing/
|   |   +-- data_loader.py              # Loads all datasets
|   |   +-- merge_datasets.py           # Master table + phenotype matrix
|   |   +-- enrich_features.py          # Gene enrichment (ClinVar+gnomAD+attr)
|   |   +-- clinvar_processor.py        # Streams 3.7GB ClinVar in chunks
|   |   +-- process_external_datasets.py# gnomAD + ACMG SF + PanelApp
|   |   +-- process_babyseq.py          # BabySeq actionability scores
|   |   +-- pediatric_filter.py         # Pediatric-onset disease filter (optional)
|   |
|   +-- classifier/
|   |   +-- train_classifier_v4.py      # Best: 426 features, 98.94% acc
|   |   +-- train_classifier_v3.py      # 408 features, ClinVar+gene attrs
|   |   +-- train_classifier_v2.py      # 300 HPO + noise augmentation
|   |   +-- train_classifier.py         # V1 baseline
|   |
|   +-- knowledge_graph/
|   |   +-- build_graph.py              # NetworkX + spectral embeddings
|   |
|   +-- retrieval/
|   |   +-- retrieval_engine.py         # TF-IDF + HPO lookup
|   |   +-- similar_disease_engine.py   # Jaccard + penetrance similarity
|   |
|   +-- app/
|       +-- dashboard_v2.py             # Streamlit UI (7 tabs)
|       +-- disease_map.py              # UMAP + Louvain clustering
|       +-- generate_figures.py         # SHAP, ablation, graph figures
|       +-- test_end_to_end.py          # 5 clinical scenario tests
|
+-- data/
|   +-- raw/       # Raw databases (excluded from git, download links above)
|   +-- processed/ # Generated by pipeline (excluded from git)
|
+-- outputs/
|   +-- models/    # Trained models (large pkl excluded, see .gitignore)
|   +-- plots/     # Figures (included)
|   +-- results/   # Metrics JSON (included)
|
+-- docs/
    +-- GeneGenie_Project_Report.docx
```

---

## Dashboard Tabs

| Tab | Function |
|-----|----------|
| Diagnose | HPO symptom input → ranked disease predictions + SHAP |
| Similar Diseases | HPO Jaccard similarity with penetrance weighting |
| Disease Map | UMAP of 500 diseases, Louvain cluster explorer |
| Cohort | Browse 12,671-disease corpus, filter by gene/phenotype |
| Knowledge Graph | Graph explorer + link prediction interface |
| Retrieval | Full-text + HPO-ID search across all 12,671 diseases |
| Evaluation | Model metrics, SHAP summary, ablation, data stats |

---

## Accuracy Note

**98.94%** = within-distribution differential diagnosis accuracy:
- 500 diseases, 20 augmented symptom subsets each = 10,000 training samples
- Test set: different random HPO subsets of same 500 diseases (20% holdout)
- Simulates: *"Given partial symptom presentation, rank correct disease #1"*
- **Top-5 (~99.5%)** = clinically relevant metric for differential diagnosis

For diseases outside the 500-class scope, the retrieval engine covers all 12,671.

---

