# Clinical-anchored multi-omics survival evaluation

This repository implements the static METABRIC version of a preliminary
four-view CCA–graph survival proposal. Molecular views are mRNA, promoter
methylation and discrete copy-number alterations; the fourth view encodes
baseline clinical covariates. There is no matched imaging or longitudinal
assay component in these experiments.

See **INNOVATION.md** for what changed and **BASELINES.md** for original-method
provenance, scaled reimplementations and explicit task adapters. Numerical
results are measured locally; no literature result is substituted for a run.
The four-page manuscript is a preliminary author-review draft. Responsible
authors must complete identity, affiliation, funding, conflicts and ethics
review/waiver information before submission. A separate appendix is not assumed
to be an allowed extra technical submission attachment.

## Environment

Python 3.11, PyTorch 2.5.1, an RTX 4050 Laptop GPU (6 GB), and the versions in
requirements.txt were used. GPU execution is automatic when CUDA is available.
Use an isolated environment; install PyTorch appropriate to your hardware,
then the listed packages. Install the official IntegrAO package separately:

```text
python -m pip install -r requirements.txt
python -m pip install --no-deps integrao==0.1.3
```

The old exact dependency constraints in IntegrAO's package metadata conflict
with this runtime. The exercised official diffusion/alignment path is tested
with the pinned modern stack; UMAP is not used. No third-party implementation
files are modified. See BASELINES.md for the pandas compatibility choice.

## Reproduce

Obtain the public cBioPortal METABRIC study files in a local folder. Point
`--source` to that folder; the source files are never modified. Required names:
data_clinical_patient.txt, data_clinical_sample.txt,
data_mrna_illumina_microarray.txt, data_methylation_promoters_rrbs.txt and
data_cna.txt. Reference profile metadata identifies expression as log2
intensity, methylation as beta values, and CNA as discrete -2 to 2 calls.

```text
python prepare.py --source /path/to/brca_metabric
python -m pytest -q
python experiment.py --mode cv
python integrao_baseline.py --mode cv
python experiment.py --mode cohort --methods clinical_cox clinical_structured_cox concat_cox gcca_cox original_graph proposed
python integrao_baseline.py --mode cohort
python curate_results.py
python summarize.py
python explain.py
python audit_cox.py
python build_paper.py
```

The dedicated manuscript/ LaTeX project must exist before build_paper.py;
it is included in this release. Compile main.tex and appendix.tex separately
using the official ISBI spconf style included in that folder. Their generated
tables are derived from CSV results; appendix.tex is not appended to main.tex.
The delivered PDFs were compiled with Tectonic 0.17.0. After compiling,
`python validate_release.py` checks result counts, splits, tables and PDF text/structure.
For a code-only smoke check, use `experiment.py --mode smoke`; smoke results
are excluded from the main scientific summaries. Runs resume from local
private predictions; use a fresh project directory if altering the protocol.

## Outputs and privacy boundary

- `results/main_results.csv`: matched five-fold, three-seed discrimination and conditional bootstrap intervals.
- `results/fold_results.csv`: aggregate per-fold/seed metrics, counts, runtime, model size and exact configuration.
- `results/parameters_all_runs.csv`: expanded actual parameter table, including IntegrAO adapters.
- `results/all_fixed_parameters.csv` and `search_candidates.csv`: fixed settings and candidate configurations.
- `results/missing_summary.csv`, `random_missing_summary.csv`: missing-view stress tests.
- `results/cohort_generalization.csv`: within-METABRIC source-group holdouts, not independent external validation.
- `results/paired_comparisons.csv`: paired patient bootstrap differences and Holm-adjusted tests.
- `results/calibration.csv`: aggregate calibration-bin statistics.
- `results/data_audit.json`: feature counts, missingness and source-file SHA-256 hashes.
- `manuscript/main.pdf`, `appendix.pdf`: four-page draft and separate supplement, with editable LaTeX sources.

Raw/preprocessed molecular data, patient-level clinical fields, training
templates, splits, representations and predictions stay in `.private/`, which
is excluded from publication. Public result files contain aggregate statistics
only. The reference PDFs are not redistributed. No credentials are stored.

Intervals resample held-out patients within their folds conditional on fitted
models; they do not capture uncertainty from retraining or a new population.
All methods share the same patient partitions; scaled/adapted baseline results
are not an assertion of a comprehensive state-of-the-art ranking.

Public source data provenance: https://github.com/cBioPortal/datahub/tree/master/public/brca_metabric
and the data papers cited in the manuscript. Dependencies retain their own
licenses. Public visibility alone does not grant a license to the original
code; the repository owner may add one separately.

## Measured outcome

Across five patient folds and three fixed seeds (1,416 patients; 830 events),
the proposed templated-clinical graph achieved Harrell C-index 0.607
(conditional patient-bootstrap 95% CI 0.587–0.625), compared with 0.656
(0.634–0.676) for structured clinical Cox. The structured-clinical graph
reached 0.655 (0.635–0.672). These results do not demonstrate a performance
advantage for the proposed combination. All 300 primary CV runs and 105
within-study source-group holdout runs are reported, including negative results.
