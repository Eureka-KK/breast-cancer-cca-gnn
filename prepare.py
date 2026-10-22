"""Audit and align local METABRIC files; never export patient-level data publicly."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
VIEW_FILES = ['data_mrna_illumina_microarray.txt',
              'data_methylation_promoters_rrbs.txt', 'data_cna.txt']
FIELDS = ['AGE_AT_DIAGNOSIS', 'SEX', 'TUMOR_STAGE', 'GRADE', 'ER_STATUS',
          'PR_STATUS', 'HER2_STATUS', 'TUMOR_SIZE', 'HISTOLOGICAL_SUBTYPE',
          'INFERRED_MENOPAUSAL_STATE']


def prepare(source):
    private = ROOT / '.private'
    private.mkdir(exist_ok=True)
    (ROOT / 'results').mkdir(exist_ok=True)
    pt = pd.read_csv(source / 'data_clinical_patient.txt', sep='\t', comment='#')
    sm = pd.read_csv(source / 'data_clinical_sample.txt', sep='\t', comment='#')
    assert not pt.PATIENT_ID.duplicated().any(), 'Duplicate clinical patient IDs'
    assert not sm.SAMPLE_ID.duplicated().any(), 'Duplicate sample IDs'
    sample_sets = [set(pd.read_csv(source / f, sep='\t', nrows=0).columns)
                   & set(sm.SAMPLE_ID) for f in VIEW_FILES]
    clinical = sm.merge(pt, on='PATIENT_ID', validate='many_to_one')
    valid_os = (pd.to_numeric(clinical.OS_MONTHS, errors='coerce').gt(0)
                & clinical.OS_STATUS.isin(['0:LIVING', '1:DECEASED']))
    complete = clinical.SAMPLE_ID.isin(set.intersection(*sample_sets))
    primary = clinical.SAMPLE_TYPE.astype(str).str.lower().eq('primary')
    # Public METABRIC uses Primary, verified rather than assuming all samples primary.
    if not primary.any():
        raise ValueError('Primary-tumour coding requires inspection')
    selected = clinical.loc[valid_os & complete & primary].sort_values('PATIENT_ID')
    assert not selected.PATIENT_ID.duplicated().any(), 'Multiple primary samples per patient'
    ids = selected.SAMPLE_ID.tolist()
    audit = dict(clinical_patients=len(pt), clinical_samples=len(sm),
                 valid_survival_samples=int(valid_os.sum()),
                 all_three_omics_samples=int(complete.sum()),
                 complete_primary_survival_patients=len(selected),
                 events=int(selected.OS_STATUS.eq('1:DECEASED').sum()),
                 excluded_after_clinical=len(clinical)-len(selected),
                 clinical_missing_fraction=selected[FIELDS].isna().mean().to_dict(),
                 views={}, source_sha256={})
    for v, name in enumerate(VIEW_FILES):
        print('Reading view', v+1, flush=True)
        df = pd.read_csv(source/name, sep='\t', dtype={i:'float32' for i in ids},
                         usecols=['Hugo_Symbol'] + ids)
        # Unique row keys preserve multiple probes/promoters without averaging biology.
        names = [f'{str(s)}|row{j}' for j, s in enumerate(df.Hugo_Symbol)]
        x = df[ids].to_numpy(dtype=np.float32).T.copy()
        finite = np.isfinite(x)
        vals = x[finite]
        audit['views'][str(v)] = dict(features=x.shape[1], missing=int((~finite).sum()),
                                      min=float(vals.min()), max=float(vals.max()),
                                      duplicate_gene_symbols=int(df.Hugo_Symbol.duplicated().sum()))
        if v == 0:
            audit['views'][str(v)]['transform'] = 'Already log2 intensity; no second log'
        if v == 1:
            if vals.min() < 0 or vals.max() > 1:
                raise ValueError('Methylation outside beta range')
            x = np.log2(np.clip(x, 1e-5, 1-1e-5)/np.clip(1-x, 1e-5, 1-1e-5))
            audit['views'][str(v)]['transform'] = 'beta to M; epsilon=1e-5'
        if v == 2:
            assert set(np.unique(vals)).issubset({-2,-1,0,1,2})
            audit['views'][str(v)]['transform'] = 'Discrete -2 to 2 calls retained'
        np.save(private/f'view{v}.npy', x)
        (private/f'features{v}.json').write_text(json.dumps(names), encoding='utf-8')
        del df, x, vals
    # Only approved baseline covariates. Local row numbers, no public identifiers.
    selected[FIELDS].to_csv(private/'clinical.csv', index=False)
    np.savez(private/'outcomes.npz',
             time=selected.OS_MONTHS.to_numpy(float),
             event=selected.OS_STATUS.eq('1:DECEASED').to_numpy(bool),
             cohort=selected.COHORT.to_numpy(float))
    selected.groupby('COHORT').agg(n=('PATIENT_ID','size'),
        events=('OS_STATUS',lambda s:s.eq('1:DECEASED').sum())).to_csv(ROOT/'results/cohort_counts.csv')
    for name in VIEW_FILES+['data_clinical_patient.txt','data_clinical_sample.txt']:
        h=hashlib.sha256()
        with (source/name).open('rb') as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b''): h.update(chunk)
        audit['source_sha256'][name]=h.hexdigest()
    (ROOT/'results/data_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print('Aligned patients:',len(selected),'events:',audit['events'],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--source',type=Path,required=True)
    prepare(a.parse_args().source)
