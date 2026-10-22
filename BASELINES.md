# Baseline provenance and task adapters

All methods use the same patient splits and train-only preprocessing. Original
paper metrics on other cohorts are never copied into the experimental tables.

| Method | Implementation and material adaptation |
|---|---|
| Unimodal / concatenation Cox | scikit-survival Coxnet, elastic-net l1 ratio 0.1, validation-selected alpha; not unpenalized Cox |
| Random survival forest | scikit-survival, 200 trees, sqrt feature sampling, validation-selected leaf size |
| DeepSurv-style | Independent reimplementation of a two-hidden-layer ReLU/dropout Cox MLP; not original author weights |
| Pairwise CCA + Cox | Ridge CCA between each molecular view and clinical view, arithmetic average clinical projections; reproduces the supplied preliminary design |
| GCCA + Cox | Regularized MAXVAR GCCA with equal view weights and Coxnet head |
| Original pairwise CCA + graph | Supplied preliminary static graph idea, two residual GCN layers and Cox head; no longitudinal information |
| MultiSurv-style (scaled) | Independent scaled implementation: per-view 128-to-hidden encoders, elementwise max fusion, 30 conditional hazards at 12-month intervals, discrete-time negative log likelihood; not a full original architecture reproduction |
| MOGONET-style + Cox | Independent modality-specific two-layer graph encoders with concatenation and a Cox head. The original classification-label VCDN is intentionally absent; this is not original MOGONET |
| Official IntegrAO + Cox adapter | PyPI integrao 0.1.3 network diffusion and unsupervised alignment; 300 alignment epochs, 8-dimensional embedding, 10 neighbors, 20 diffusion steps; validation-selected Coxnet head |

Official IntegrAO is from https://github.com/bowang-lab/IntegrAO (MIT).
It requires old exact package versions. Install the validated dependencies in
requirements.txt first, then `pip install --no-deps integrao==0.1.3`. pandas 2.2.3
is important: its code mutates DataFrame.values, which is read-only in pandas 3.
The installed implementation itself is not edited. No UMAP function is used.
Its fixed original optimizer schedule is retained: Adam learning rate 0.1,
decayed 10-fold each 100 epochs, lower bounded by 0.001; early exaggeration is
removed at epoch 100; hidden dimension 128; alignment loss weight 1.
Train-only diffusion and one-way train-to-query GraphSAGE edges provide an
explicit inductive adapter. Batch independence is asserted during every run.
The graph encoders are trained without survival labels; only Cox head alpha
selection uses validation survival labels. The representation hyperparameters
are fixed rather than searched, so search budgets are not identical across
every baseline. This difference is disclosed rather than hidden.

MultiSurv source and paper: https://github.com/luisvalesilva/multisurv and
https://doi.org/10.1038/s41598-021-92799-4.
MOGONET source and paper: https://github.com/txWang/MOGONET and
https://doi.org/10.1038/s41467-021-23774-w.
DeepSurv paper: https://doi.org/10.1186/s12874-018-0482-1.

The supplied CXR-LT and chest-X-ray rejection papers concern classification;
they are background on robustness, not comparable survival baselines. Learn
Then Test and risk-controlling prediction sets are not applied to censored
survival outcomes here; no finite-sample risk-control guarantee is claimed.
