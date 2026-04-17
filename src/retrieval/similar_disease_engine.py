# -*- coding: utf-8 -*-
"""
Similar Disease Query Engine - Day 7
Novel features for research paper:

1. Symptom-to-IBA-Panel Mapping:
   Input HPO terms -> identify which clinical panel(s) are activated
   -> filter diseases by relevant panels

2. Penetrance-Adjusted Ranking:
   Raw model confidence * penetrance_score -> re-ranked output
   (novel: most tools don't account for penetrance in ranking)

3. Evidence-Weighted Similarity:
   Disease similarity = HPO jaccard + evidence grade weight
   Similar diseases retrieved with ClinGen evidence quality score

4. Actionability Index Display:
   For each prediction: show combined_actionability (BabySeq + ClinVar)

5. Differential Diagnosis Mode:
   Given disease A, find N most-similar diseases by HPO overlap
   -> helps clinician distinguish between close diseases

6. Newborn Alert:
   Flag diseases present in Category A (highly penetrant, pediatric)
   and diseases meeting NGSR criteria (actionable in newborns)
"""

import numpy as np
import pandas as pd
import json
import pickle
import logging
from pathlib import Path
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

BASE = Path(__file__).parent.parent.parent
PROC = BASE / "data" / "processed"
MODELS = BASE / "outputs" / "models"
RESULTS = BASE / "outputs" / "results"


class SimilarDiseaseEngine:
    """
    Multi-modal similar disease retrieval.
    Combines: HPO-based similarity + IBA panel overlap + evidence weighting
    """

    def __init__(self):
        self.disease_phenotype = None   # disease x HPO matrix
        self.hpo_lookup = {}
        self.iba_panel_map = {}
        self.hpo_to_panel = {}
        self.babyseq = None
        self.gene_action = None
        self.master = None
        self.disease_vectors = None
        self.disease_ids = []
        self.tfidf = None
        self.disease_texts = {}

    def load(self):
        """Load all required data."""
        # HPO lookup
        hpo_path = RESULTS / "hpo_lookup.json"
        if hpo_path.exists():
            with open(hpo_path) as f:
                self.hpo_lookup = json.load(f)

        # OMIM + Orphanet disease name map (38K diseases)
        name_map_path = RESULTS / "disease_name_map.json"
        if name_map_path.exists():
            with open(name_map_path, encoding="utf-8") as f:
                self._omim_name_map = json.load(f)
            log.info(f"Disease name map: {len(self._omim_name_map)} entries")
        else:
            self._omim_name_map = {}

        # IBA panel maps
        iba_path = PROC / "iba_panel_hpo_map.json"
        if iba_path.exists():
            with open(iba_path) as f:
                self.iba_panel_map = json.load(f)

        hpo2panel_path = PROC / "hpo_to_iba_panel.json"
        if hpo2panel_path.exists():
            with open(hpo2panel_path) as f:
                self.hpo_to_panel = json.load(f)

        # Disease phenotype matrix
        dpm_path = PROC / "disease_phenotype_matrix.csv"
        if dpm_path.exists():
            self.disease_phenotype = pd.read_csv(dpm_path, index_col=0, low_memory=False)
            self.disease_ids = self.disease_phenotype.index.tolist()
            self.disease_vectors = self.disease_phenotype.values
            log.info(f"Disease phenotype matrix: {self.disease_phenotype.shape}")

        # BabySeq gene-disease table
        bs_path = PROC / "babyseq_gene_disease.csv"
        if bs_path.exists():
            self.babyseq = pd.read_csv(bs_path, low_memory=False)
            log.info(f"BabySeq: {self.babyseq.shape}")

        # Gene actionability v2 (GnomAD + ACMG + PanelApp enriched)
        ga_path = PROC / "gene_actionability_v2.csv"
        if not ga_path.exists():
            ga_path = PROC / "gene_actionability.csv"
        if ga_path.exists():
            self.gene_action = pd.read_csv(ga_path, low_memory=False)
            log.info(f"Gene actionability: {self.gene_action.shape} from {ga_path.name}")

        # Master for disease names + gene associations
        master_path = PROC / "master_gene_disease_phenotype.csv"
        if master_path.exists():
            self.master = pd.read_csv(master_path, low_memory=False)
            # Build disease name lookup — prefer OMIM/Orphanet map (38K names)
            # master disease_name column is mostly NaN, so use _omim_name_map
            self.disease_name_map = dict(self._omim_name_map)  # copy full map
            # Overwrite with any non-null master names
            if "disease_name" in self.master.columns:
                master_names = (
                    self.master.dropna(subset=["disease_name"])
                    .groupby("disease_id")["disease_name"].first().to_dict()
                )
                self.disease_name_map.update(master_names)
            # Build disease -> gene map
            self.disease_gene_map = (
                self.master.groupby("disease_id")["gene_symbol"]
                .apply(lambda x: list(x.dropna().unique()))
                .to_dict()
            )
            # Build disease -> HPO map
            self.disease_hpo_map = (
                self.master.groupby("disease_id")["hpo_id"]
                .apply(lambda x: list(x.dropna().unique()))
                .to_dict()
            )
            log.info(f"Master loaded: {len(self.disease_name_map)} diseases named")

        # TF-IDF retrieval
        tfidf_path = MODELS / "tfidf_index.pkl"
        if tfidf_path.exists():
            try:
                self.tfidf = pickle.load(open(tfidf_path, "rb"))
                log.info("TF-IDF index loaded")
            except Exception as e:
                log.warning(f"TF-IDF load failed: {e}")

        log.info("SimilarDiseaseEngine loaded")

    def get_iba_panels_for_query(self, hpo_ids):
        """Map input HPO terms -> activated IBA panels."""
        panel_counts = {}
        for hpo in hpo_ids:
            for panel in self.hpo_to_panel.get(hpo, []):
                panel_counts[panel] = panel_counts.get(panel, 0) + 1

        # Sort by activation count
        activated = sorted(panel_counts.items(), key=lambda x: x[1], reverse=True)
        return activated  # [(panel, count), ...]

    def get_penetrance_score(self, disease_id):
        """Look up penetrance score for a disease from BabySeq."""
        if self.babyseq is None:
            return 0.5
        # Match by disease name approximation
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return 0.5
        matches = self.babyseq[self.babyseq["Gene"].isin([g.upper() for g in genes])]
        if matches.empty:
            return 0.5
        return float(matches["penetrance_score"].max())

    def get_evidence_score(self, disease_id):
        """Look up max evidence score for disease genes."""
        if self.babyseq is None:
            return 0
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return 0
        matches = self.babyseq[self.babyseq["Gene"].isin([g.upper() for g in genes])]
        if matches.empty:
            return 0
        return int(matches["evidence_score"].max())

    def get_actionability(self, disease_id):
        """Get combined actionability index v2 (BabySeq + GnomAD pLI + ACMG SF + PanelApp)."""
        if self.gene_action is None:
            return 0.0
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return 0.0
        matches = self.gene_action[
            self.gene_action["gene_symbol"].str.upper().isin([g.upper() for g in genes])]
        if matches.empty:
            return 0.0
        # Use v2 score if available (GnomAD + ACMG enriched)
        col = "combined_actionability_v2" if "combined_actionability_v2" in matches.columns \
              else "combined_actionability"
        return float(matches[col].max())

    def get_gnomad_constraint(self, disease_id):
        """Get max pLI and LOEUF for disease genes (GnomAD v4.1)."""
        if self.gene_action is None or "pLI" not in self.gene_action.columns:
            return None, None
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return None, None
        matches = self.gene_action[
            self.gene_action["gene_symbol"].str.upper().isin([g.upper() for g in genes])]
        if matches.empty:
            return None, None
        pli = float(matches["pLI"].max()) if not matches["pLI"].isna().all() else None
        loeuf = float(matches["LOEUF"].min()) if not matches["LOEUF"].isna().all() else None
        return pli, loeuf

    def is_acmg_sf(self, disease_id):
        """Check if any disease gene is on ACMG Secondary Findings list."""
        if self.gene_action is None or "acmg_sf" not in self.gene_action.columns:
            return False
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return False
        matches = self.gene_action[
            self.gene_action["gene_symbol"].str.upper().isin([g.upper() for g in genes])]
        return bool((matches["acmg_sf"] == 1).any())

    def is_newborn_alert(self, disease_id):
        """Flag: disease is Category A (highly penetrant, pediatric) in BabySeq."""
        if self.babyseq is None:
            return False
        genes = self.disease_gene_map.get(disease_id, [])
        if not genes:
            return False
        matches = self.babyseq[
            (self.babyseq["Gene"].isin([g.upper() for g in genes])) &
            (self.babyseq["is_category_a"] == 1)
        ]
        return len(matches) > 0

    def query_by_hpo(self, hpo_ids, top_k=10, penetrance_adjust=True):
        """
        Find similar diseases by HPO overlap (Jaccard + cosine).
        Optionally re-rank by penetrance score.
        """
        if self.disease_phenotype is None:
            return []

        hpo_cols = self.disease_phenotype.columns.tolist()
        query_vec = np.zeros(len(hpo_cols), dtype=np.float32)
        matched = []
        for h in hpo_ids:
            if h in hpo_cols:
                query_vec[hpo_cols.index(h)] = 1.0
                matched.append(h)

        if query_vec.sum() == 0:
            return [], matched

        # Cosine similarity
        sims = cosine_similarity(query_vec.reshape(1, -1), self.disease_vectors)[0]

        # Take top candidates - use more to ensure after dedup we have top_k unique
        top_idx = np.argsort(sims)[::-1][:top_k * 10]

        results = []
        query_set = set(hpo_ids)
        seen_diseases = set()  # deduplicate by disease_id

        for i in top_idx:
            did = self.disease_ids[i]
            # Skip duplicates (disease_phenotype_matrix has augmented rows)
            if did in seen_diseases:
                continue
            seen_diseases.add(did)
            cosine_sc = float(sims[i])

            # Jaccard similarity
            disease_hpos = set(self.disease_hpo_map.get(did, []))
            if disease_hpos:
                jaccard = len(query_set & disease_hpos) / len(query_set | disease_hpos)
            else:
                jaccard = 0.0

            combined_sim = 0.6 * cosine_sc + 0.4 * jaccard

            # Penetrance-adjusted score (novel)
            pen_score = self.get_penetrance_score(did) if penetrance_adjust else 1.0
            adj_score = combined_sim * (0.7 + 0.3 * pen_score)

            evidence = self.get_evidence_score(did)
            actionability = self.get_actionability(did)
            newborn_alert = self.is_newborn_alert(did)
            acmg_flag = self.is_acmg_sf(did)
            pli, loeuf = self.get_gnomad_constraint(did)

            results.append({
                "disease_id": did,
                "disease_name": self.disease_name_map.get(did, did),
                "cosine_similarity": round(cosine_sc, 4),
                "jaccard_similarity": round(jaccard, 4),
                "combined_similarity": round(combined_sim, 4),
                "penetrance_score": round(pen_score, 2),
                "penetrance_adjusted_score": round(adj_score, 4),
                "evidence_score": evidence,
                "evidence_label": {4: "Definitive", 3: "Strong", 2: "Moderate",
                                   1: "Limited", 0: "Unknown"}.get(evidence, "Unknown"),
                "actionability_index": round(actionability, 3),
                "newborn_alert": newborn_alert,
                "acmg_sf": acmg_flag,       # ACMG Secondary Findings flag
                "pLI": round(pli, 3) if pli is not None else None,
                "LOEUF": round(loeuf, 3) if loeuf is not None else None,
                "associated_genes": ", ".join(self.disease_gene_map.get(did, [])[:5]),
                "n_shared_hpo": len(query_set & disease_hpos),
                "n_disease_hpo": len(disease_hpos),
            })

        # Sort by penetrance-adjusted score
        results.sort(key=lambda x: x["penetrance_adjusted_score"], reverse=True)
        return results[:top_k], matched

    def differential_diagnosis(self, disease_id, top_k=10):
        """
        Given a known disease, find the N most similar diseases (differential).
        Useful for research: 'what diseases could be confused with X?'
        """
        if self.disease_phenotype is None or disease_id not in self.disease_phenotype.index:
            return []

        # Use first row if multiple augmented rows exist
        row = self.disease_phenotype.loc[disease_id]
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        disease_vec = row.values.reshape(1, -1)
        sims = cosine_similarity(disease_vec, self.disease_vectors)[0]

        top_idx = np.argsort(sims)[::-1][:top_k * 5]  # more to ensure unique after dedup

        results = []
        seen = set()
        for i in top_idx:
            did = self.disease_ids[i]
            if did == disease_id or did in seen:
                continue
            seen.add(did)
            results.append({
                "disease_id": did,
                "disease_name": self.disease_name_map.get(did, did),
                "similarity": round(float(sims[i]), 4),
                "evidence_score": self.get_evidence_score(did),
                "penetrance_score": round(self.get_penetrance_score(did), 2),
                "newborn_alert": self.is_newborn_alert(did),
                "genes": ", ".join(self.disease_gene_map.get(did, [])[:5]),
            })

        return results[:top_k]

    def panel_filter_diseases(self, panel_name, top_k=20):
        """
        Return diseases strongly associated with a clinical panel (IBA).
        Uses BabySeq panel_* columns.
        """
        if self.babyseq is None:
            return []

        col = f"panel_{panel_name}"
        if col not in self.babyseq.columns:
            return []

        panel_genes = self.babyseq[self.babyseq[col] == 1].copy()
        panel_genes = panel_genes.sort_values("actionability_index", ascending=False)

        results = []
        seen_diseases = set()
        for _, row in panel_genes.iterrows():
            disease = str(row.get("disease_name", "")).strip()
            if not disease or disease in seen_diseases:
                continue
            seen_diseases.add(disease)
            results.append({
                "gene": row.get("Gene", ""),
                "disease_name": disease,
                "evidence_level": row.get("evidence_level", ""),
                "penetrance": row.get("Penetrance", ""),
                "category": row.get("babyseq_category", ""),
                "actionability_index": round(float(row.get("actionability_index", 0)), 3),
                "newborn_alert": row.get("is_category_a", 0) == 1,
            })
            if len(results) >= top_k:
                break

        return results


def run_demo():
    """Quick demo of the engine."""
    engine = SimilarDiseaseEngine()
    engine.load()

    # Demo: seizures + hypotonia + developmental delay
    demo_hpos = ["HP:0001250", "HP:0001252", "HP:0001263", "HP:0000252"]
    log.info(f"Demo query: {demo_hpos}")

    # IBA panel activation
    panels = engine.get_iba_panels_for_query(demo_hpos)
    log.info(f"Activated panels: {panels}")

    # Similar diseases
    results, matched = engine.query_by_hpo(demo_hpos, top_k=5, penetrance_adjust=True)
    log.info(f"Matched HPOs: {matched}")
    for r in results:
        alert = "[NEWBORN ALERT]" if r["newborn_alert"] else ""
        log.info(f"  {r['disease_id']} | adj_score={r['penetrance_adjusted_score']} "
                 f"| evidence={r['evidence_label']} {alert}")

    # Panel filter
    seiz_diseases = engine.panel_filter_diseases("SEIZ", top_k=5)
    log.info(f"SEIZ panel top diseases:")
    for d in seiz_diseases:
        log.info(f"  {d['disease_name']} | {d['gene']} | {d['category']} | alert={d['newborn_alert']}")


if __name__ == "__main__":
    run_demo()
