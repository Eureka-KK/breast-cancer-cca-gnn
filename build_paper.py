"""Build evidence-linked manuscript sources from completed experimental tables."""
import argparse
import json
import shutil
from pathlib import Path
import numpy as np
import pandas as pd
from experiment import ROOT,CFG
from summarize import NAMES


def tex(s):
    s=str(s)
    for a,b in [('&',r'\&'),('%',r'\%'),('_',r'\_'),('#',r'\#')]:s=s.replace(a,b)
    return s


def table(df,columns,labels,fmt=None,size=r'\footnotesize'):
    lines=[r'\begin{center}',size,r'\begin{tabular}{'+('l'+'r'*(len(columns)-1))+r'}\hline',
           ' & '.join(labels)+r'\\\hline']
    for _,row in df.iterrows():
        values=[]
        for c in columns:
            v=row[c]
            if fmt and c in fmt:v=fmt[c](v)
            elif isinstance(v,(float,np.floating)):v=f'{v:.3f}' if np.isfinite(v) else 'NA'
            values.append(tex(v))
        lines.append(' & '.join(values)+r'\\')
    lines.extend([r'\hline\end{tabular}',r'\end{center}'])
    return '\n'.join(lines)


def data():
    r=ROOT/'results'
    summary=pd.read_csv(r/'main_results.csv')
    required=set(NAMES)
    if set(summary.method)!=required:
        raise ValueError(f'Incomplete CV methods: {sorted(required-set(summary.method))}')
    folds=pd.read_csv(r/'fold_results.csv')
    cv=folds[folds.split.str.match(r'^cv\d+$')]
    assert len(cv)==300, f'Expected 300 CV runs, got {len(cv)}'
    audit=json.loads((r/'data_audit.json').read_text())
    return summary,folds,audit


def main_source(summary,folds,audit):
    s=summary.set_index('method');p=s.loc['proposed'];clinical=s.loc['clinical_structured_cox']
    best=summary[~summary.method.isin(['proposed','no_anchor','no_graph','no_mask','no_residual','structured_clinical'])].sort_values('cindex',ascending=False).iloc[0]
    delta=p.cindex-best.cindex
    comparison=(f'The shared-anchor graph achieved a mean C-index of {p.cindex:.3f} '
        f'(conditional 95\\% CI {p.ci_low:.3f}--{p.ci_high:.3f}), compared with '
        f'{best.cindex:.3f} for {tex(best.label)}. The difference was {delta:+.3f}.')
    conclusion=('The fusion configuration improved discrimination over the strongest evaluated baseline, '
                'but the magnitude and paired uncertainty must be interpreted within this single dataset.'
                if delta>0 else
                'The fusion configuration did not outperform the strongest evaluated baseline. '
                'The experiment supports a reproducible evaluation framework rather than a claim of superior prognosis prediction.')
    missing=pd.read_csv(ROOT/'results/missing_summary.csv',dtype={'subset':str})
    missing=missing[missing.method=='proposed'].copy();missing['subset']=missing.subset.str.zfill(4)
    keep=missing[missing.subset.isin(['1111','0111','1011','1101','1110'])].copy()
    keep['label']=keep.subset.map({'1111':'All four views','0111':'Without mRNA','1011':'Without methylation',
                                  '1101':'Without CNA','1110':'Without clinical'})
    abr=summary[summary.method.isin(['proposed','no_anchor','no_graph','no_mask','no_residual','structured_clinical'])].copy()
    abr['short']=abr.method.map({'proposed':'Shared-anchor graph','no_anchor':'Equal view weights','no_graph':'Without graph',
        'no_mask':'Without mask training','no_residual':'Without raw residual','structured_clinical':'Structured clinical'})
    mains=summary[~summary.method.isin(['no_anchor','no_graph','no_mask','no_residual','structured_clinical'])].copy()
    short={'clinical_cox':'Templated clinical Cox','clinical_structured_cox':'Structured clinical Cox','mrna_cox':'mRNA Cox','methylation_cox':'Methylation Cox','cna_cox':'CNA Cox',
        'concat_cox':'Concatenation Cox','rsf':'Survival forest','deepsurv':'DeepSurv-style',
        'pairwise_cox':'Pairwise CCA + Cox','gcca_cox':'GCCA + Cox','original_graph':'Pairwise CCA + GCN',
        'multisurv':'MultiSurv-style (scaled)','mogonet_cox':'MOGONET-style + Cox','integrao_cox':'IntegrAO + Cox adapter',
        'proposed':'Shared-anchor graph'}
    mains['short']=mains.method.map(short)
    mains['interval']=[f'{a:.3f}--{b:.3f}' for a,b in zip(mains.ci_low,mains.ci_high)]
    group=pd.read_csv(ROOT/'results/cohort_generalization.csv')
    gp=group[group.method=='proposed']
    group_sentence=(f'Across five source-group holdouts, the proposed model had mean C-indices ranging '
                    f'from {gp.cindex.min():.3f} to {gp.cindex.max():.3f}.')
    comparison_table=table(mains,['short','cindex','interval'],['Method','C-index','95\% CI'])
    ablation_table=table(abr,['short','cindex'],['Configuration','C-index'])
    missing_table=table(keep,['label','cindex'],['Available input','C-index'])
    primary_comparison=pd.read_csv(ROOT/'results/paired_comparisons.csv')
    row=primary_comparison[primary_comparison.comparator==best.method].iloc[0]
    paired=f'The paired bootstrap difference against the strongest baseline had 95\\% interval {row.ci_low:+.3f} to {row.ci_high:+.3f} (Holm-adjusted $p={row.p_holm:.3f}$).'
    return r'''\documentclass{article}
\usepackage[T1]{fontenc}
\usepackage{spconf,amsmath,graphicx,times}
\ifdefined\pdfinfo\pdfinfo{/Author (EurekaChen) /Title () /Subject () /Keywords ()}
\else\special{pdf:docinfo << /Author (EurekaChen) /Title () /Subject () /Keywords () >>}\fi
\title{Clinical-Anchored Multi-Omics Graph Learning for Breast Cancer Survival: An Audited METABRIC Evaluation}
\name{Author information to be completed}
\address{Affiliation and contact information to be completed}
\begin{document}
\maketitle
\begin{abstract}
Multi-omics survival models require patient-level matching and evaluation that
distinguishes representation learning from information leakage. We adapt a
preliminary four-view canonical-correlation graph model to METABRIC, using
mRNA, promoter methylation, copy-number alterations and baseline clinical
covariates. A regularized generalized canonical correlation analysis (GCCA)
uses one shared latent target with an explicit clinical-view weight; a static
graph combines the available projected views and predicts overall survival.
We evaluate 1,416 patients with 830 deaths using five patient-level held-out
folds, validation-only model selection and three seeds, comparing fifteen
main configurations including survival-specific and adapted multi-omics
baselines. '''+comparison+r''' All fifteen nonempty view subsets, five
component ablations and source-group holdouts are evaluated. '''+conclusion+r'''
There are no repeated visits, paired images or independent external cohort.
\end{abstract}
\begin{keywords}Multi-omics, breast cancer, survival, GCCA, graph learning\end{keywords}
\section{Introduction}
Breast cancer prognosis depends on clinical disease burden and molecular
heterogeneity. METABRIC provides expression, copy-number and clinical
annotations for breast tumours, with additional molecular profiles available
for partially overlapping samples \cite{curtis,pereira}. Combining these
assays may improve individual risk estimates, but increasing model complexity
does not ensure that the additional information generalizes. Matched patient
identifiers, genuinely baseline covariates and a consistent evaluation target
are prerequisites for any meaningful comparison.

Graph-based multi-omics methods offer a natural way to model relationships
between patients. MOGONET uses view-specific graph encoders and cross-view
classification-label interactions \cite{mogonet}; IntegrAO learns aligned
representations from partially overlapping molecular graphs \cite{integrao}.
These contributions do not, by themselves, establish an advantage for a new
Cox survival model. Comparisons must distinguish original task definitions
from downstream adapters, and must keep test patients outside representation
fitting and graph diffusion. Survival-specific baselines, including DeepSurv
and MultiSurv, provide complementary nonlinear approaches \cite{deepsurv,multisurv}.

The supplied preliminary design aligned three molecular views independently
to templated clinical text and then averaged the clinical projections.
Independently fitted canonical coordinates can differ in sign and orientation;
arithmetic averaging does not resolve this ambiguity. We therefore evaluate
a shared-target regularized GCCA formulation, alongside that preliminary
pairwise construction. GCCA is an established estimator \cite{gcca}; the
clinical weight and its combination with missing-aware inductive graph
prediction are implementation choices to test, not claims of a new CCA theory.

Our contribution is an auditable, static survival-learning evaluation:
(i) identifier-based alignment of four METABRIC views; (ii) shared versus
pairwise clinical alignment and a training-reference graph; and (iii) matched
comparisons, component ablations and missing-view stress tests. We explicitly
report negative findings and adapted baseline architectures. This work is a
multi-omics prognosis study, not an imaging-fusion validation or a temporal
disease-progression study.
\clearpage
\section{Methods}
\subsection{Patients, views and preprocessing}
From 2,509 clinical records, 1,980 had positive overall-survival time and a
recognized event/censoring code. Intersecting the three assay sample sets with
primary tumours and valid survival yielded 1,416 unique patients, including
830 deaths. Before training, all views were ordered by the same verified
patient/sample mapping. Duplicate gene symbols were retained as distinct
probe/promoter rows, not collapsed into an unsupported biological identity.

Expression measurements were already log$_2$ intensities and were not logged
again. Promoter methylation beta values were transformed to
$\log_2[\beta/(1-\beta)]$ after clipping to $[10^{-5},1-10^{-5}]$; discrete
copy-number calls were retained. Within each training fold, assay features
with over 20\% missing observations were excluded, 500 or 1,000 high-variance
features were selected, missing entries were median-filled, and features were
standardized and clipped at five training standard deviations. No feature
selection, vocabulary fitting or scaling used validation/test patients.

The clinical view contained age, sex, stage, grade, ER, PR and HER2 status, tumour
size, histological type and menopausal state. Stage was missing in 26.7\% of
the included patients. Outcome, recurrence and treatment fields were excluded.
Fixed field--value strings were encoded with 1--2-gram TF-IDF and training-only
SVD (at most 128 dimensions). These are templated structured covariates,
not free-text notes or an additional information source. A structured
numerical and categorical encoder was evaluated as an ablation.

\subsection{Shared alignment and static graph}
Let $X_v$ be centered training features for view $v$, and let $a_4$ denote the
clinical-view weight; other $a_v$ equal one. We solve the regularized MAXVAR
GCCA problem
\begin{equation}
\min_{G,\{W_v\}}\sum_{v=1}^{4}a_v
  \big(\|X_vW_v-G\|_F^2+n\lambda\|W_v\|_F^2\big),\quad G^\top G=nI.
\end{equation}
The leading eigenvectors of $\sum_v a_vH_v$ give $G$, where
$H_v=X_v(X_v^\top X_v+n\lambda I)^{-1}X_v^\top$; primal or dual solves
are chosen by matrix size. The projections are then fitted to this single
target. The comparator fits three independent molecular--clinical ridge
CCAs and averages the clinical projections as in the preliminary design.

For availability indicators $m_{iv}$, graph features are
$\bar z_i=(\sum_vm_{iv}z_{iv})/(\sum_vm_{iv})$. A symmetric binary cosine
$k$-nearest-neighbour graph is normalized after adding self-loops. Two
residual GCN layers with LayerNorm and dropout produce the graph embedding.
The Cox head receives this embedding, each masked view projection and the
four availability indicators. Independent view masking during training is
conditioned to leave at least one view; an otherwise empty draw retains mRNA.

The training graph contains only training patients. At inference, each query
receives messages from its nearest training nodes and itself; training-node
states and degrees are frozen. Queries have no edges to each other or back
to training nodes. Thus a prediction does not depend on other patients in its
test batch. No visit times, survival times or event indicators enter the graph.

\subsection{Learning and validation}
The neural Cox head minimizes Breslow negative partial log likelihood
\begin{equation}
\mathcal L=-\frac{1}{\sum_i\delta_i}\sum_{i:\delta_i=1}
\left[r_i-\log\sum_{j:t_j\geq t_i}\exp(r_j)\right].
\end{equation}
Five event-stratified patient folds were fixed with seed 42. Each development
set reserved 20\% for validation (approximately 64/16/20\% overall). Up to
twelve configurations per method were evaluated using validation C-index;
the winner was rerun with seeds 42, 2026 and 3407. Neural training used AdamW,
weight decay $10^{-3}$, at most 300 epochs and patience 30. Ablations reused
the selected proposed-model settings. Full search and selected values are
reported in the separate appendix and machine-readable parameter tables.
\clearpage
\section{Experiments and Results}
\subsection{Matched baselines and discrimination}
The main comparison included five unimodal regularized Cox models
(three molecular views and two clinical encodings),
concatenation Cox, a survival forest, a DeepSurv-style Cox MLP, pairwise
CCA--Cox, equal-weight GCCA--Cox, the preliminary pairwise graph, a scaled
MultiSurv-style discrete-time model, a MOGONET-style multi-graph Cox adapter,
official IntegrAO embeddings with an inductive Cox adapter, and the proposed
configuration. The independently reimplemented/scaled baselines are labeled
as such; they are not exact reproductions of every original architecture.
IntegrAO representation settings were fixed, with validation selection only
for its Cox regularization. All other model-selection details are in the appendix.

Table~1 reports the mean test C-index over five folds and three seeds.
Patient bootstrap intervals (1,000 resamples) preserve held-out fold membership
and use the same resampled patients for method comparisons. They condition on
the fitted models and do not include retraining or new-center uncertainty.
\begin{center}\small Table 1. Held-out overall-survival discrimination.\end{center}
'''+comparison_table+r'''
'''+comparison+' '+paired+r'''
The structured clinical-only model reached '''+f'{clinical.cindex:.3f}'+r''', making it an
essential control for any claimed benefit from expensive molecular views.
IPCW concordance, dynamic AUC and Brier scores at one, three and five years
are reported in the appendix. Censoring estimates and Cox baseline hazards
were fitted on training data. Time-dependent evaluation administratively
censored held-out observations beyond training follow-up support.

\subsection{Missing views, ablations and source groups}
All fifteen nonempty availability patterns were evaluated without refitting
the model on the test patients. Table~2 summarizes clinically interpretable
single-view omissions; all combinations and random missingness at 10, 30
and 50\% appear in the appendix. Missingness was simulated independently
and should not be interpreted as a clinical missing-not-at-random process.
\begin{center}\small Table 2. Proposed model under view omission.\end{center}
'''+missing_table+r'''
\begin{center}\small Table 3. Component ablations (fixed selected settings).\end{center}
'''+ablation_table+r'''
The structured-clinical graph variant reached '''+f'{s.loc["structured_clinical"].cindex:.3f}'+r''', close to the
structured clinical Cox point estimate of '''+f'{clinical.cindex:.3f}'+r'''. This comparison does not
establish added value from graph modeling.
The equal-weight ablation removes additional clinical weighting, not the
clinical view itself. If the proposed search selects weight one, that fold's
anchor ablation is identical by construction. This possibility is retained
rather than forcing an artificial distinction. The structured-clinical
comparison tests representation, since both encoders receive the same fields.

'''+group_sentence+r''' These groups are METABRIC source labels and are
not an independent external breast-cancer cohort. All group-specific models
were fitted without the held-out group; results include the clinical,
concatenation, GCCA, original-graph, proposed and IntegrAO controls.
\clearpage
\section{Discussion}
'''+conclusion+r''' The clinical-only control and the explicitly adapted
modern baselines are particularly informative: a flexible graph may preserve
correlated molecular structure without improving the ranking of survival
times. Added model capacity, assay noise, limited sample size and the objective
mismatch between unsupervised correlation and censored outcome prediction
are plausible explanations to investigate, not mechanisms established here.

The shared GCCA target resolves a coordinate consistency problem in the
preliminary pairwise design, but correcting that construction is not proof
of prognostic superiority. Likewise, accurate operation under a specified
view-availability pattern does not establish calibrated uncertainty or an
error-control guarantee. The supplied rejection and Learn Then Test papers
concern different prediction losses; we do not transfer their guarantees to
censored survival outcomes without a corresponding theory and experiment.

Several limitations bound the interpretation. First, selecting complete assays
reduces the available survival cohort from 1,980 to 1,416 patients and may
introduce selection bias. Second, validation and test patients originate from
one study, and the additional source-group holdouts do not replace external
validation. Third, clinical strings are an encoding of measured fields; they
cannot establish a language-model or clinical-note benefit. Fourth, there are
no repeated visits and no matched patient images. Accordingly, the temporal
and imaging extensions in the preliminary proposal remain unevaluated.

The survival head makes a proportional-hazards modeling assumption, and
Breslow-derived survival probabilities may be miscalibrated even when ranking
performance is acceptable. Calibration and out-of-fold Kaplan--Meier groups
are therefore reported separately. Bootstrap intervals are conditional on
this experiment, while subgroup analyses and simulated missingness are
exploratory. Architecture-scaled baseline reimplementations are useful
controls but do not establish a complete state-of-the-art ranking.

\section{Conclusion}
We provide patient-aligned four-view METABRIC preprocessing, shared and
pairwise canonical-correlation models, an inductive static graph, and a
reproducible survival benchmark with missing-view evaluation. The results
support only the performance comparisons measured here. Independent
external validation, genuinely longitudinal assays and paired imaging would
be required before extending the scientific or clinical claims.

\section{Compliance with Ethical Standards}
This is a secondary analysis of publicly distributed, de-identified METABRIC
research data. No new patient data were collected. The responsible authors
must confirm their institution's review/waiver determination before any
submission; no approval identifier or exemption is invented here.

\section{Acknowledgments and Disclosure}
This preliminary draft requires author, affiliation, funding and conflict-of-interest
confirmation. Codex assisted with implementation, experiment execution and
draft text throughout the methods/results manuscript; accountable human
authors must verify the analyses, references and conclusions before submission.
Code and aggregate tables are prepared for a public repository; patient-level
clinical data and predictions are excluded. The separate appendix is an
author-review supplement, not an assumed permitted extra technical submission page.

\begin{thebibliography}{10}\small
\bibitem{curtis} C. Curtis et al., ``The genomic and transcriptomic architecture
of 2,000 breast tumours reveals novel subgroups,'' \emph{Nature}, vol. 486,
pp. 346--352, 2012.
\bibitem{pereira} B. Pereira et al., ``The somatic mutation profiles of 2,433
breast cancers refines their genomic and transcriptomic landscapes,''
\emph{Nature Communications}, vol. 7, art. 11479, 2016.
\bibitem{mogonet} T. Wang et al., ``MOGONET integrates multi-omics data using
graph convolutional networks allowing patient classification and biomarker
identification,'' \emph{Nature Communications}, vol. 12, art. 3445, 2021.
\bibitem{integrao} S. Ma et al., ``Moving towards genome-wide data integration
for patient stratification with Integrate Any Omics,'' \emph{Nature Machine
Intelligence}, vol. 7, pp. 29--42, 2025.
\bibitem{deepsurv} J. L. Katzman et al., ``DeepSurv: personalized treatment
recommender system using a Cox proportional hazards deep neural network,''
\emph{BMC Medical Research Methodology}, vol. 18, art. 24, 2018.
\bibitem{multisurv} L. A. Vale-Silva and K. Rohr, ``Long-term cancer survival
prediction using multimodal deep learning,'' \emph{Scientific Reports},
vol. 11, art. 13505, 2021.
\bibitem{gcca} J. R. Kettenring, ``Canonical analysis of several sets of
variables,'' \emph{Biometrika}, vol. 58, pp. 433--451, 1971.
\end{thebibliography}
\end{document}
'''


def appendix_source(summary,folds,audit):
    parts=[r'''\documentclass[10pt]{article}
\usepackage[T1]{fontenc}\usepackage{times}
\usepackage[margin=0.8in]{geometry}
\usepackage{amsmath,graphicx,longtable}
\ifdefined\pdfinfo\pdfinfo{/Author (EurekaChen) /Title () /Subject () /Keywords ()}
\else\special{pdf:docinfo << /Author (EurekaChen) /Title () /Subject () /Keywords () >>}\fi
\title{Appendix: Clinical-Anchored Multi-Omics Graph Learning}
\author{Author information to be completed}\date{}
\begin{document}\pagestyle{empty}\maketitle\thispagestyle{empty}
This is a separate author-review supplement. It is not part of the four-page
ISBI manuscript and is not assumed eligible for extra technical submission pages.
\section{Data alignment and evidence boundaries}
The primary cohort contains 1,416 unique primary-tumour patients with three
molecular assays and valid overall survival; 830 died. Clinical records numbered
2,509, and 1,980 had usable survival. Three assay intersections contained 1,417
clinical samples before survival/primary-tumour filtering. The clinical-table
exclusions therefore total 1,093 for the final complete-assay analysis.
Source-file hashes and exact feature counts are in results/data\_audit.json.
No outcomes, recurrence fields, future treatment information or cohort labels
are used as predictive clinical inputs. Cohort labels are reserved for evaluation.

The supplied TCGA/MOGONET matrices lack patient IDs and are not joined by row
order to METABRIC. BUSI and BUSBRA are unrelated patients and are not joined
to molecular records. CNA replaces the preliminary miRNA view. Temporal
graphs, simulated visits, clinical notes and paired imaging are not evaluated.
''']
    counts=pd.read_csv(ROOT/'results/cohort_counts.csv')
    parts.append(table(counts,['COHORT','n','events'],['Source group','Patients','Deaths']))
    feature=pd.DataFrame([{'view':v,'features':a['features'],'missing':a['missing']} for v,a in zip(['mRNA','Methylation','CNA'],audit['views'].values())])
    parts.append(table(feature,['view','features','missing'],['View','Raw features','Missing cells']))
    parts.append(r'\section{All fixed parameters and search settings}')
    parts.append(r'\begin{longtable}{p{.48\textwidth}p{.43\textwidth}}\hline Parameter & Value\\\hline')
    for k,v in CFG.items():parts.append(tex(k)+' & '+tex(json.dumps(v))+r'\\')
    parts.append(r'\hline\end{longtable}')
    parts.append(r'''The twelve candidate dictionaries are in search\_candidates.json;
selected\_parameters.csv stores the actual fold/seed settings. GCCA projection
coordinates are standardized using training projection standard deviations.
Clinical numerical values are inserted into fixed field--value templates;
missing values are ``unknown''. Structured clinical encoding uses training
median filling, numerical missingness indicators, standardization and one-hot
categories with unknown categories ignored. TF-IDF retains tokens containing
digits and decimals; no pretrained language encoder is used.

Coxnet uses l1 ratio 0.1, at most 10,000 iterations and tolerance $10^{-6}$.
Its size is the number of nonzero coefficients. Neural size is the number of
trainable scalar parameters. Survival-forest size is total tree-node count,
which is not directly comparable to neural parameter count. The forest uses
200 trees, square-root feature sampling and four local workers. Its memory-
efficient implementation reconstructs terminal-node bootstrap-weighted
Kaplan--Meier probabilities only at requested horizons; agreement with native
full-memory survival curves is tested. Twelve configurations are an upper budget, not an
exhaustive Cartesian search. Some candidate choices irrelevant to a particular
baseline are retained in the audit record but do not change that model;
duplicate effective forest configurations are evaluated once. The structured
clinical Cox control was added after initial fold inspection to strengthen
the clinical comparison. This is an exploratory, non-preregistered study.

MultiSurv-style encoders have a 128-unit first layer and 32/64-unit output,
elementwise maximum fusion and a same-width hidden prediction layer producing
30 twelve-month hazards. Its scalar C-index risk is negative summed survival
across intervals, whereas time-dependent AUC below uses this same global marker.
Its original pan-cancer architecture and original code are not reproduced exactly.
MOGONET-style encoders use modality-specific two-layer graphs and a concatenated
Cox head; no classification-label VCDN is used. DeepSurv-style denotes a
two-hidden-layer ReLU/dropout MLP trained with Cox loss.

IntegrAO uses the official 0.1.3 implementation with fixed representation
settings: 128 hidden units, 8 output dimensions, 10 neighbors, 20 diffusion
steps, and 300 alignment epochs. Adam uses its original schedule from 0.1,
decayed by ten each 100 epochs to a minimum 0.001; alignment MSE weight is
one and t-SNE early exaggeration is removed at epoch 100. A Coxnet head alpha
is selected from twelve log-spaced values $10^{-3}$ to $10^1$; its Coxnet
head uses tolerance $10^{-7}$ (the other Coxnet baselines use $10^{-6}$). New patients
receive one-way incoming edges from training neighbors only. Evaluation mode
freezes batch normalization; query-batch independence is asserted. These
survival and inductive adapters are explicit departures from the original
downstream task. Exact package versions are in requirements.txt and environment.json.
''')
    parts.append(r'\section{Full discrimination, time-dependent metrics and model size}')
    for columns,labels in [(['label','cindex','ci_low','ci_high'],['Configuration','C-index','CI low','CI high']),
            (['label','ipcw_cindex','auc_12m','auc_36m','auc_60m'],['Configuration','IPCW C','AUC 1y','AUC 3y','AUC 5y']),
            (['label','brier_12m','brier_36m','brier_60m','parameters_median'],['Configuration','BS 1y','BS 3y','BS 5y','Size'])]:
        parts.append(table(summary,columns,labels))
    parts.append(r'''Harrell concordance is the mean across five held-out folds and
three seeds. Intervals use 1,000 patient-level bootstrap resamples separately
within each held-out fold, with identical resamples for all methods. Seed-wise
concordance contributions are averaged within a resample. These intervals are
conditional on fitted models, not nested-bootstrap retraining intervals.
Pairwise differences are two-sided and Holm-adjusted over all comparisons
against the proposed configuration, including ablations. Raw test outcomes
are retained for Harrell concordance. For time-dependent metrics, follow-up
beyond training support is administratively censored just below the largest
training follow-up. Censoring survival at each horizon must exceed 0.05; a
nonestimable metric is NA. Fold-average time-dependent metrics use available
estimable folds. No finite-sample selective-risk guarantee is inferred.
''')
    tests=pd.read_csv(ROOT/'results/paired_comparisons.csv')
    tests['label']=tests.comparator.map(NAMES)
    parts.append(table(tests,['label','delta','ci_low','ci_high','p_holm'],['Comparator','Difference','CI low','CI high','Holm p']))
    parts.append(r'\section{All missing-view combinations}')
    missing=pd.read_csv(ROOT/'results/missing_summary.csv',dtype={'subset':str});missing['subset']=missing.subset.str.zfill(4)
    wide=missing.pivot(index='subset',columns='method',values='cindex').reset_index()
    parts.append(table(wide,['subset','proposed','original_graph','no_mask'],['mRNA/Meth/CNA/Clinical','Proposed','Pairwise graph','No masking']))
    parts.append(r'\begin{center}\includegraphics[width=.9\textwidth]{missing.pdf}\end{center}')
    random=pd.read_csv(ROOT/'results/random_missing_summary.csv');random['label']=random.method.map(NAMES)
    parts.append(table(random,['label','probability','cindex'],['Configuration','Missing probability','C-index']))
    parts.append(r'\section{Source-group holdouts}')
    group=pd.read_csv(ROOT/'results/cohort_generalization.csv');group['label']=group.method.map(NAMES)
    for split,a in group.groupby('split'):
        parts.append(r'\subsection{'+tex(split)+'}')
        parts.append(table(a,['label','n_test','events','cindex','seed_sd'],['Configuration','Patients','Deaths','C-index','Seed SD']))
    parts.append(r'''These are leave-one-source-group-out analyses within METABRIC,
not independently collected external cohorts. Their preprocessing and graph
references exclude the held-out group. Validation is 20\% of remaining patients,
event-stratified with seed 42. The same 12-configuration upper search budget
is used except for the documented fixed IntegrAO representation.
\section{Calibration, risk groups and interpretability limits}
\begin{center}\includegraphics[width=.95\textwidth]{calibration_km.pdf}\end{center}
Each training fold's median predicted risk defines its test high-risk group;
three seed-wise decisions are combined by majority vote. Calibration averages
the three seed-specific survival probabilities within each held-out fold,
then compares prediction quintiles with Kaplan--Meier event probabilities.
These plots are descriptive, and risk-group separation is not a replacement
for discrimination or calibration metrics. GCCA loadings are linear
associations and should not be described as causal biomarkers. The current
study does not establish clinically validated biomarkers or graph attention:
the implemented graph is a GCN, not an attention network.

\section{Relationship to supplied reference PDFs}
The CXR-LT reports (Holste 2024; Dong 2026 overview and full report) concern
long-tailed chest-X-ray classification. They motivate careful cross-domain
testing but do not supply survival comparison numbers. The Aperstein 2025
preprint and 2026 Scientific Reports version concern rejection in
multi-pathology chest-X-ray classification; the later article is not counted
as an independent second study. Geifman and El-Yaniv (2017) concerns selective
classification. Bates et al. (2021) concerns risk-controlling prediction sets;
Angelopoulos et al. (Learn Then Test, arXiv 2021) concerns finite-sample
calibration through hypothesis testing. Their guarantees are not transferred
to censored Cox outcomes in this implementation. No chest-X-ray experiment,
selective survival guarantee, or numerical baseline result is invented.

\section{Per-fold audit and reproducibility}
The complete fold-results CSV contains all patient counts, event counts,
metrics, training seconds, selected epoch, model size and exact configuration.
All forty selected main-CV Cox solutions were re-fitted in a convergence
audit. They reproduced their original C-indices to numerical precision and
emitted no solver warnings. Failed numerical search candidates were not
assigned performance values and were excluded from selection; the attempt
log retains their status. Interrupted, uncompleted fits were rerun.
The following table includes each completed main-CV fold/seed result.
''')
    parts.append(r'\begin{longtable}{llrrr}\hline Configuration & Fold & Seed & C-index & Epoch\\\hline')
    for _,row in folds[folds.split.str.startswith('cv')].sort_values(['method','split','seed']).iterrows():
        parts.append(tex(row.method)+' & '+tex(row.split)+f' & {row.seed} & {row.cindex:.4f} & {row.best_epoch}'+r'\\')
    parts.append(r'\hline\end{longtable}')
    loadings=pd.read_csv(ROOT/'results/canonical_loadings.csv')
    top=loadings[(loadings.fold==0)&(loadings.component==1)&(loadings.loading_rank<=5)]
    parts.append(r'\section{Training-only canonical interpretation}')
    parts.append('The following examples are the five largest absolute projection coefficients for component one of fold zero, separately per molecular view. Structure correlations are Pearson correlations of each standardized input feature with that training canonical score. The complete fold/component table is in canonical\\_loadings.csv. Neither coefficients nor correlations establish causal biomarkers; signs and rotations are not compared across folds.')
    parts.append(table(top,['view','feature','standardized_projection_coefficient','structure_correlation'],['View','Gene/probe row','Coefficient','Correlation']))
    parts.append(r'''Source data, clinical templates and patient-level outcomes/predictions
remain in the ignored local private directory. The public release contains
code, configurations, source hashes and aggregate tables only. Author identity,
institution, funding, conflicts and ethics review/waiver determination remain
for responsible-author completion. AI assistance is disclosed in the main
draft. Neither this appendix nor its page count is assumed compatible with
the conference's technical-page submission restrictions.
\end{document}''')
    return '\n'.join(parts)


def build(print_main=False):
    summary,folds,audit=data()
    source=main_source(summary,folds,audit)
    if print_main:print(source);return
    paper=ROOT/'manuscript'
    if not paper.exists():raise ValueError('Create the dedicated LaTeX project first')
    (paper/'main.tex').write_text(source,encoding='utf-8')
    (paper/'appendix.tex').write_text(appendix_source(summary,folds,audit),encoding='utf-8')
    for name in ['spconf.sty','missing.pdf','calibration_km.pdf']:
        generated=ROOT/'paper'/name
        if generated.exists():shutil.copy2(generated,paper/name)
        elif not (paper/name).exists():raise FileNotFoundError(f'Missing manuscript asset: {name}')
    print('Manuscript and appendix sources generated from verified tables')


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--print-main',action='store_true')
    build(a.parse_args().print_main)
