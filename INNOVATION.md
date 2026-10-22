# Method changes and evidence boundaries

The supplied main.pdf/methods.pdf propose four-view anchored pairwise CCA,
clinical text generated from structured fields, patient graphs, and survival
learning. The current implementation tests a static METABRIC adaptation.

1. **Shared coordinate system.** Independently solved CCA projections have
   sign/rotation ambiguities. A single regularized MAXVAR GCCA target aligns
   all four views jointly; clinical view weight is explicit. Weighted GCCA is
   established mathematics, not claimed here as a new estimator.
2. **Missing-aware inductive graph.** Graph construction uses the mean of the
   available view projections; the head retains separate projections and an
   availability mask. Query nodes receive training-reference messages only.
   This design is tested over all 15 nonempty view subsets and random missingness.
3. **Auditable experiment.** Explicit patient alignment, train-only transforms,
   no outcome words in clinical text, source-group holdouts, and paired patient
   bootstrap evaluate whether the combination helps under controlled conditions.

The original temporal claims and miRNA view are not supported by these local
data. CNA replaces miRNA. No simulation is presented as longitudinal evidence.
Clinical narratives are templated covariate strings, not clinical notes; TF-IDF
is not an external foundation model or extra independent clinical information.

The contribution is an evaluated integration and deployment protocol. Whether
there is a useful empirical gain must be read from actual tables, including
negative results. This document is not a claim of first publication, patent
novelty, independent external validation, or clinical readiness.
