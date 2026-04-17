# Large Raw Data Files — Download Sources

These files are excluded from the repository due to size. Download each into `data/raw/` before running the pipeline.

| File | Size | Source | Direct Link |
|------|------|--------|-------------|
| `variant_summary.txt` | ~3.7 GB | NCBI ClinVar | https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz |
| `kg.csv` | ~937 MB | PrimeKG (Harvard MIMS) | https://github.com/mims-harvard/PrimeKG |
| `gene_attribute_matrix.csv` / `.txt` | ~250 MB | Harmonizome (Ma'ayan Lab) | https://maayanlab.cloud/Harmonizome/resource/NCBI+Genes |
| `gene_similarity_matrix_cosine.csv` / `.txt` | ~187 MB | Harmonizome | https://maayanlab.cloud/Harmonizome/resource/NCBI+Genes |
| `attribute_similarity_matrix_cosine.txt` | ~344 MB | Harmonizome | https://maayanlab.cloud/Harmonizome/resource/NCBI+Genes |
| `gnomad_constraint.tsv` | ~50 MB | gnomAD v4.1 | https://gnomad.broadinstitute.org/downloads#v4-constraint |
| `Homo_sapiens.gene_info` | ~50 MB | NCBI Gene FTP | https://ftp.ncbi.nlm.nih.gov/gene/DATA/GENE_INFO/Mammalia/Homo_sapiens.gene_info.gz |
| `phenotype_to_genes_JAX.csv` | ~20 MB | HPO Annotations (JAX) | https://hpo.jax.org/data/annotations |
| `genes_to_phenotype_ontology.csv` | ~10 MB | HPO Annotations (JAX) | https://hpo.jax.org/data/annotations |
| `en_product1.xml` | ~15 MB | Orphadata | https://www.orphadata.com/diseases/ |
| `en_product6.xml` | ~30 MB | Orphadata | https://www.orphadata.com/genes/ |
| `gene_attribute_edges.csv` / `.txt` | ~500 KB | Harmonizome | https://maayanlab.cloud/Harmonizome/resource/NCBI+Genes |

## Files Included in Repo (small enough)

| File | Notes |
|------|-------|
| `acmg_sf_v32.csv` | ACMG Secondary Findings v3.2 gene list |
| `gene2phenotype.csv` | Gene2Phenotype curated associations |
| `genemap2.txt` | OMIM gene-disease map |
| `morbidmap.csv` / `.txt` | OMIM morbid map |
| `mim2gene.csv` / `.txt` | OMIM MIM to gene mapping |
| `mimTitles.csv` / `.txt` | OMIM disease titles |
| `orphanet_diseases.csv` | Orphanet disease list (processed) |
| `orphanet_genes.csv` | Orphanet gene list (processed) |
| `panelapp_iba_genes.json` | NHS PanelApp IBA panel genes |
| `attribute_list_entries.txt` | Harmonizome attribute list |
| `gene_list_terms.txt` | Harmonizome gene list |

## Setup Command

After downloading, verify placement:
```bash
ls data/raw/
python run_pipeline.py --module check_inputs
```

## Notes

- ClinVar `variant_summary.txt.gz` must be unzipped after download
- PrimeKG `kg.csv` is available on the Harvard Dataverse: https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IXA7BM
- gnomAD constraint file: download `gnomad.v4.1.constraint_metrics.tsv` and rename to `gnomad_constraint.tsv`
- Harmonizome matrix files: navigate to NCBI Genes resource → download gene-attribute matrix (binary)
