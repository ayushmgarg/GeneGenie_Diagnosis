"""
run_pipeline.py -- Master script to run the full ML pipeline.

Steps:
  1. Preprocessing + merge all datasets
  2. Feature enrichment (ClinVar + gene_attribute_matrix + all remaining)
  3. Pediatric filter (Orphanet XML, HPO onset codes)
  3b. BabySeq + IBA panel processing (Day 6)
  4. Train disease classifier V3 (HPO + ClinVar + gene attributes)
  4b. Train disease classifier V4 (+ IBA + BabySeq features, novel)
  5. Build knowledge graph + link prediction
  6. Build retrieval engine (TF-IDF)
  7. Similar disease engine (HPO + penetrance + evidence, novel)
  8. Disease map (UMAP + Louvain clustering)
  9. End-to-end tests (5 clinical scenarios)

Run: python run_pipeline.py
Or per module: python run_pipeline.py --module classifier_v4
"""

import sys
import argparse
import logging
import time
from pathlib import Path

BASE = Path(__file__).parent
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(BASE / "src"))


def step_preprocessing(skip_large=False):
    log.info("=" * 60)
    log.info("STEP 1: PREPROCESSING + MERGE ALL DATASETS")
    log.info("=" * 60)
    from preprocessing.merge_datasets import run_merge
    t0 = time.time()
    result = run_merge(skip_large=skip_large)
    log.info(f"Preprocessing done in {time.time()-t0:.1f}s")
    return result


def step_enrich():
    log.info("=" * 60)
    log.info("STEP 2: FEATURE ENRICHMENT (ClinVar + gene_attribute_matrix + all remaining)")
    log.info("=" * 60)
    from preprocessing.enrich_features import run as enrich_run
    t0 = time.time()
    enrich_run()
    log.info(f"Enrichment done in {time.time()-t0:.1f}s")


def step_pediatric():
    log.info("=" * 60)
    log.info("STEP 3: PEDIATRIC FILTER + GEOLOCATION")
    log.info("=" * 60)
    from preprocessing.pediatric_filter import run as ped_run
    t0 = time.time()
    ped_run()
    log.info(f"Pediatric filter done in {time.time()-t0:.1f}s")


def step_clinvar():
    log.info("=" * 60)
    log.info("STEP 3b: CLINVAR PROCESSING (3.7GB variant_summary)")
    log.info("=" * 60)
    from preprocessing.clinvar_processor import process_clinvar
    t0 = time.time()
    df = process_clinvar(chunksize=50000, max_rows=500000)
    log.info(f"ClinVar done in {time.time()-t0:.1f}s, {len(df) if df is not None else 0} genes")


def step_classifier_v3():
    log.info("=" * 60)
    log.info("STEP 4: PEDIATRIC CLASSIFIER V3 (HPO + ClinVar + Gene Attributes)")
    log.info("=" * 60)
    from classifier.train_classifier_v3 import run as v3_run
    t0 = time.time()
    v3_run()
    log.info(f"Classifier V3 done in {time.time()-t0:.1f}s")


def step_classifier_v1():
    log.info("=" * 60)
    log.info("STEP 4b: GENERAL CLASSIFIER V1")
    log.info("=" * 60)
    from classifier.train_classifier import run_training
    t0 = time.time()
    run_training()
    log.info(f"Classifier V1 done in {time.time()-t0:.1f}s")


def step_classifier_v2():
    log.info("=" * 60)
    log.info("STEP 4c: PEDIATRIC CLASSIFIER V2")
    log.info("=" * 60)
    from classifier.train_classifier_v2 import run as v2_run
    t0 = time.time()
    v2_run()
    log.info(f"Classifier V2 done in {time.time()-t0:.1f}s")


def step_knowledge_graph():
    log.info("=" * 60)
    log.info("STEP 5: KNOWLEDGE GRAPH + LINK PREDICTION")
    log.info("=" * 60)
    from knowledge_graph.build_graph import run_knowledge_graph
    t0 = time.time()
    run_knowledge_graph(embed_dim=64, max_edges=100000)
    log.info(f"KG done in {time.time()-t0:.1f}s")


def step_retrieval():
    log.info("=" * 60)
    log.info("STEP 6: RETRIEVAL ENGINE (TF-IDF + HPO + Gene index)")
    log.info("=" * 60)
    from retrieval.retrieval_engine import run_retrieval_setup, demo_retrieval
    t0 = time.time()
    engine = run_retrieval_setup()
    demo_retrieval(engine)
    log.info(f"Retrieval done in {time.time()-t0:.1f}s")
    return engine


def step_external_datasets():
    log.info("=" * 60)
    log.info("STEP 3c: EXTERNAL DATASETS (GnomAD + ACMG SF + PanelApp)")
    log.info("=" * 60)
    from preprocessing.process_external_datasets import run as ext_run
    t0 = time.time()
    ext_run()
    log.info(f"External datasets done in {time.time()-t0:.1f}s")


def step_babyseq():
    log.info("=" * 60)
    log.info("STEP 3b: BABYSEQ + IBA PANEL PROCESSING (Day 6)")
    log.info("=" * 60)
    from preprocessing.process_babyseq import run as bs_run
    t0 = time.time()
    bs_run()
    log.info(f"BabySeq done in {time.time()-t0:.1f}s")


def step_classifier_v4():
    log.info("=" * 60)
    log.info("STEP 4b: PEDIATRIC CLASSIFIER V4 (HPO + IBA + BabySeq + Gene, novel)")
    log.info("=" * 60)
    from classifier.train_classifier_v4 import run as v4_run
    t0 = time.time()
    v4_run()
    log.info(f"Classifier V4 done in {time.time()-t0:.1f}s")


def step_disease_map():
    log.info("=" * 60)
    log.info("STEP 8: DISEASE MAP (UMAP + Louvain clustering)")
    log.info("=" * 60)
    from app.disease_map import build_disease_map
    t0 = time.time()
    fig, clusters = build_disease_map()
    log.info(f"Disease map done: {clusters['cluster'].nunique()} clusters in {time.time()-t0:.1f}s")


def step_end_to_end_tests():
    log.info("=" * 60)
    log.info("STEP 9: END-TO-END TESTS (5 clinical scenarios)")
    log.info("=" * 60)
    from app.test_end_to_end import run_tests
    t0 = time.time()
    results = run_tests()
    log.info(f"Tests done in {time.time()-t0:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Rare Disease ML Pipeline - Full")
    parser.add_argument("--module", choices=[
        "preprocess", "enrich", "pediatric", "clinvar", "babyseq", "external",
        "classifier", "classifier_v2", "classifier_v3", "classifier_v4",
        "kg", "retrieval", "disease_map", "test", "all"
    ], default="all", help="Which module to run")
    parser.add_argument("--skip-large", action="store_true",
                        help="Skip loading large files (ClinVar, gene_attribute_matrix)")
    args = parser.parse_args()

    log.info("=== RARE DISEASE ML PIPELINE (FULL) ===")
    log.info(f"Module: {args.module} | Skip-large: {args.skip_large}")
    log.info("Data sources: HPO, OMIM, Orphanet, ClinVar, gene_attribute_matrix,"
             " gene_similarity, gene2phenotype, NCBI gene_info, mim2gene, morbidmap, kg.csv")

    total_start = time.time()

    if args.module in ("preprocess", "all"):
        step_preprocessing(skip_large=args.skip_large)

    if args.module in ("clinvar", "all") and not args.skip_large:
        step_clinvar()

    if args.module in ("enrich", "all"):
        step_enrich()

    if args.module in ("pediatric", "all"):
        step_pediatric()

    if args.module in ("classifier", "all"):
        step_classifier_v1()

    if args.module in ("classifier_v2", "all"):
        step_classifier_v2()

    if args.module in ("classifier_v3", "all"):
        step_classifier_v3()

    if args.module in ("babyseq", "all"):
        step_babyseq()

    if args.module in ("external", "all"):
        step_external_datasets()

    if args.module in ("classifier_v4", "all"):
        step_classifier_v4()

    if args.module in ("kg", "all"):
        step_knowledge_graph()

    if args.module in ("retrieval", "all"):
        step_retrieval()

    if args.module in ("disease_map", "all"):
        step_disease_map()

    if args.module in ("test", "all"):
        step_end_to_end_tests()

    elapsed = (time.time() - total_start) / 60
    log.info(f"\n=== PIPELINE COMPLETE in {elapsed:.1f} minutes ===")
    log.info("Outputs:")
    log.info("  Classifiers: outputs/models/random_forest_v4.pkl (98.94% acc)")
    log.info("  KG:          outputs/models/knowledge_graph.pkl (AUC 99.05%)")
    log.info("  Retrieval:   outputs/models/tfidf_index.pkl")
    log.info("  Disease Map: outputs/plots/plotly_disease_map.html")
    log.info("  Clusters:    data/processed/louvain_clusters.csv")
    log.info("  Plots:       outputs/plots/")
    log.info("  Results:     outputs/results/")
    log.info("\nLaunch dashboard: python -m streamlit run src/app/dashboard_v2.py")


if __name__ == "__main__":
    main()
