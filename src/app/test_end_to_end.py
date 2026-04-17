# -*- coding: utf-8 -*-
"""
5 End-to-End Test Cases for RareDx System.

Design note:
  V4 classifier (98.94% acc) is trained with full feature vectors:
    HPO (300) + IBA panels (13) + gene-attrs (100) + ClinVar (5) + BabySeq (5)
  At CLINICAL QUERY TIME we only know HPO symptoms -> gene/ClinVar features = 0.
  This is by design: V4 serves as the research-grade model (paper result).
  For symptom-only clinical queries, the SimilarDiseaseEngine (HPO retrieval) is used.

Tests cover:
  1. V4 performance on its TEST SET (already validated, we load saved metrics)
  2. SimilarDiseaseEngine: HPO-to-disease retrieval (5 clinical scenarios)
  3. IBA panel activation per scenario
  4. Newborn alert flags
"""

import sys
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import json
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE / "src"))

MODELS = BASE / "outputs" / "models"
RESULTS = BASE / "outputs" / "results"
PROC = BASE / "data" / "processed"

# ─── 5 Clinical Test Scenarios (HPO-based retrieval) ───────────────────────────

SCENARIOS = [
    {
        "id": 1,
        "name": "Infantile Spasms + Global Delay",
        "hpo_ids": [
            "HP:0001250",  # Seizure
            "HP:0012469",  # Infantile spasms (West syndrome)
            "HP:0001263",  # Global developmental delay
            "HP:0001252",  # Muscular hypotonia
            "HP:0000252",  # Microcephaly
        ],
        "expected_panels": ["SEIZ", "HYPOTO"],
        "clinical_note": "West syndrome / early-onset epileptic encephalopathy. "
                         "Differential: ARX, CDKL5, KCNQ2, SCN2A, TSC1/2.",
    },
    {
        "id": 2,
        "name": "Neonatal Cardiac + Short Stature + Facial",
        "hpo_ids": [
            "HP:0001629",  # VSD
            "HP:0001631",  # ASD
            "HP:0004322",  # Short stature
            "HP:0000369",  # Low-set ears
            "HP:0000316",  # Hypertelorism
            "HP:0001508",  # Failure to thrive
        ],
        "expected_panels": ["CHD"],
        "clinical_note": "Noonan / Alagille / CHARGE. "
                         "Cardiac + dysmorphic. PTPN11, JAG1, CHD7 candidates.",
    },
    {
        "id": 3,
        "name": "Skeletal Dysplasia + Respiratory",
        "hpo_ids": [
            "HP:0002650",  # Scoliosis
            "HP:0000774",  # Narrow chest
            "HP:0003510",  # Short trunk
            "HP:0001511",  # IUGR
            "HP:0002119",  # Ventriculomegaly
        ],
        "expected_panels": ["COND"],
        "clinical_note": "Short-rib thoracic / Jeune syndrome / achondroplasia spectrum. "
                         "FGFR3, IFT80, WDR34.",
    },
    {
        "id": 4,
        "name": "Autism + Macrocephaly + Motor Delay",
        "hpo_ids": [
            "HP:0000729",  # Autistic behavior
            "HP:0001263",  # Global developmental delay
            "HP:0001270",  # Motor delay
            "HP:0000256",  # Macrocephaly
            "HP:0000750",  # Delayed speech
        ],
        "expected_panels": ["HYPOTO"],
        "clinical_note": "PTEN-spectrum, Sotos, Angelman, Fragile X spectrum. "
                         "NSD1, PTEN, UBE3A, FMR1 candidates.",
    },
    {
        "id": 5,
        "name": "Neonatal Liver Disease + Cholestasis",
        "hpo_ids": [
            "HP:0002240",  # Hepatomegaly
            "HP:0001396",  # Cholestasis
            "HP:0001508",  # Failure to thrive
            "HP:0001903",  # Anemia
            "HP:0001251",  # Ataxia
        ],
        "expected_panels": ["IEM"],
        "clinical_note": "Inborn errors of metabolism: bile acid synthesis defects, "
                         "Niemann-Pick, Wilson, PFIC. ATP8B1, ABCB11, POLG.",
    },
]


def load_similar_engine():
    try:
        from retrieval.similar_disease_engine import SimilarDiseaseEngine
        engine = SimilarDiseaseEngine()
        engine.load()
        print("  [OK] SimilarDiseaseEngine loaded")
        return engine
    except Exception as e:
        print(f"  [FAIL] Engine: {e}")
        return None


def run_tests():
    print("=" * 70)
    print("RareDx End-to-End Test Suite")
    print("=" * 70)

    # ─── Part 1: Report saved classifier metrics ──────────────────────────────
    print("\n[PART 1] Classifier Performance (from saved metrics)")
    print("-" * 50)
    for vname, fname in [("V1", "classifier_metrics.json"),
                          ("V2", "classifier_metrics_v2.json"),
                          ("V3", "classifier_metrics_v3.json"),
                          ("V4", "classifier_metrics_v4.json")]:
        p = RESULTS / fname
        if p.exists():
            data = json.load(open(p))
            if isinstance(data, list):
                for m in data:
                    if "RandomForest" in m.get("model", ""):
                        print(f"  {vname} RF:  acc={m.get('accuracy',0):.4f}  "
                              f"f1={m.get('f1_macro',0):.4f}  "
                              f"top5={m.get('top5_accuracy',m.get('top5',0)):.4f}")
                        break

    cv4 = json.load(open(RESULTS / "cv_scores_v4.json"))
    print(f"  V4 StratifiedKFold CV: {cv4['cv_f1_macro_mean']:.4f} +/- {cv4['cv_f1_macro_std']:.4f}")
    print(f"  V4 features: {cv4['n_features']} ({cv4['feature_breakdown']})")

    # ─── Part 2: IBA Panel + Similar Disease retrieval ────────────────────────
    print("\n[PART 2] Clinical HPO Retrieval (5 test scenarios)")
    print("  Note: SimilarDiseaseEngine uses HPO-only jaccard+cosine+penetrance")
    print("-" * 50)

    engine = load_similar_engine()
    hpo_to_panel_path = PROC / "hpo_to_iba_panel.json"
    hpo_to_panel = json.load(open(hpo_to_panel_path)) if hpo_to_panel_path.exists() else {}

    summary = []

    for sc in SCENARIOS:
        print(f"\n{'=' * 60}")
        print(f"SCENARIO {sc['id']}: {sc['name']}")
        print(f"  Input HPO: {sc['hpo_ids']}")
        print(f"  Clinical note: {sc['clinical_note']}")

        # IBA panel activation
        panel_counts = {}
        for h in sc["hpo_ids"]:
            for panel in hpo_to_panel.get(h, []):
                panel_counts[panel] = panel_counts.get(panel, 0) + 1
        top_panels = sorted(panel_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        print(f"\n  [IBA Panels Activated]: {top_panels}")
        expected = sc.get("expected_panels", [])
        panel_hit = any(p in panel_counts for p in expected)
        print(f"  Expected panels {expected}: {'FOUND' if panel_hit else 'NOT FOUND'}")

        # Similar disease retrieval
        if engine is not None:
            results, matched_hpos = engine.query_by_hpo(sc["hpo_ids"], top_k=5,
                                                         penetrance_adjust=True)
            print(f"\n  [Matched HPOs]: {len(matched_hpos)}/{len(sc['hpo_ids'])}")
            print(f"  [Top-5 Similar Diseases (penetrance-adjusted)]:")
            newborn_count = 0
            for r in results:
                alert = "[NB ALERT]" if r["newborn_alert"] else ""
                print(f"    {r['disease_id']} | adj={r['penetrance_adjusted_score']:.3f} "
                      f"| pen={r['penetrance_score']} "
                      f"| evidence={r['evidence_label']:<12} "
                      f"| genes={r['associated_genes'][:30]} {alert}")
                if r["newborn_alert"]:
                    newborn_count += 1
            print(f"  Newborn alerts in top-5: {newborn_count}")
        else:
            results = []
            matched_hpos = []
            panel_hit = False
            newborn_count = 0

        summary.append({
            "scenario": sc["id"],
            "name": sc["name"],
            "n_hpo_input": len(sc["hpo_ids"]),
            "n_hpo_matched": len(matched_hpos),
            "top_panel": top_panels[0][0] if top_panels else "",
            "panel_hit": panel_hit,
            "n_results": len(results),
            "newborn_alerts": newborn_count,
        })

    # ─── Part 3: Summary ──────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    df = pd.DataFrame(summary)
    print(df.to_string(index=False))

    panel_hits = sum(s["panel_hit"] for s in summary)
    print(f"\nIBA panel hit rate: {panel_hits}/{len(SCENARIOS)}")
    print(f"Average HPO match rate: {df['n_hpo_matched'].mean():.1f}/{df['n_hpo_input'].mean():.1f}")
    print(f"Total newborn alerts in top-5 results: {df['newborn_alerts'].sum()}")

    # Save results
    out = RESULTS / "end_to_end_test_results.json"
    with open(out, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nResults saved: {out}")

    return summary


if __name__ == "__main__":
    run_tests()
