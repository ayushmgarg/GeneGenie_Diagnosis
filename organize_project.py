"""
organize_project.py — moves all raw data files into data/raw/
Run once to clean up the project root.
"""

import shutil
from pathlib import Path

BASE = Path(__file__).parent

RAW_DATA_FILES = [
    "Homo_sapiens.gene_info",
    "attribute_list_entries.txt",
    "attribute_similarity_matrix_cosine.txt",
    "diseases_for_HP_0000118",
    "en_product1.xml",
    "en_product6.xml",
    "gene2phenotype.csv",
    "gene_attribute_edges.csv",
    "gene_attribute_edges.txt",
    "gene_attribute_matrix.csv",
    "gene_attribute_matrix.txt",
    "gene_list_terms.txt",
    "gene_similarity_matrix_cosine.csv",
    "gene_similarity_matrix_cosine.txt",
    "genemap2.txt",
    "genes_for_HP_0000118",
    "genes_to_phenotype_ontology.csv",
    "kg.csv",
    "mim2gene.csv",
    "mim2gene.txt",
    "mimTitles.csv",
    "mimTitles.txt",
    "morbidmap.csv",
    "morbidmap.txt",
    "orphanet_diseases.csv",
    "orphanet_genes.csv",
    "phenotype_to_genes_JAX.csv",
    "search_info_C5816885-C1850534-C0234533-C0205858-C0014548+5.txt",
    "search_result_C5816885-C1850534-C0234533-C0205858-C0014548+5.xlsx",
    "variant_summary.txt",
]

DOCS = ["7_Day_ML_Project_Plan.docx", "ML_Project_Summary.docx"]

dest_raw = BASE / "data" / "raw"
dest_docs = BASE / "docs"
dest_raw.mkdir(parents=True, exist_ok=True)
dest_docs.mkdir(parents=True, exist_ok=True)

moved = 0
for fname in RAW_DATA_FILES:
    src = BASE / fname
    dst = dest_raw / fname
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"  Moved: {fname} -> data/raw/")
        moved += 1
    elif dst.exists():
        print(f"  Already in raw: {fname}")

for fname in DOCS:
    src = BASE / fname
    dst = dest_docs / fname
    if src.exists() and not dst.exists():
        shutil.move(str(src), str(dst))
        print(f"  Moved: {fname} -> docs/")

print(f"\nDone. Moved {moved} files to data/raw/")
print("Project structure:")
for p in sorted(BASE.rglob("*")):
    if p.is_dir() and ".git" not in str(p):
        indent = "  " * (len(p.relative_to(BASE).parts) - 1)
        print(f"{indent}[DIR] {p.name}/")
