"""Non-image verification of completed experimental records and final PDFs."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import fitz
from experiment import ROOT,CFG
from summarize import NAMES


def validate():
    r=ROOT/'results';d=pd.read_csv(r/'fold_results.csv')
    assert len(d)==405 and not d.duplicated(['split','method','seed']).any()
    cv=d[d.split.str.match(r'^cv\d+$')];groups=d[d.split.str.match(r'^cohort\d+$')]
    assert len(cv)==300 and len(groups)==105
    assert set(cv.method)==set(NAMES)
    assert (cv.groupby('method').size()==15).all()
    outcome=np.load(ROOT/'.private/outcomes.npz')
    test_rows=[]
    for i in range(5):
        split=np.load(ROOT/f'.private/split_cv{i}.npz')
        tr,va,te=[set(split[k]) for k in ['train','valid','test']]
        assert not(tr&va or tr&te or va&te)
        assert len(tr|va|te)==1416
        test_rows.extend(te)
    assert len(test_rows)==len(set(test_rows))==1416
    for c in range(1,6):
        split=np.load(ROOT/f'.private/split_cohort{c}.npz')
        assert (outcome['cohort'][split['test']]==c).all()
        assert (outcome['cohort'][split['train']]!=c).all()
    summary=pd.read_csv(r/'main_results.csv')
    assert len(summary)==20
    means=cv.groupby('method').cindex.mean()
    for _,row in summary.iterrows():
        assert abs(row.cindex-means[row.method])<1e-9
        assert 0<=row.ci_low<=row.cindex<=row.ci_high<=1
    for name,n,extra,keys in [('missing_subsets.csv',675,450,['split','method','seed','subset']),('random_missing.csv',135,90,['split','method','seed','probability'])]:
        missing=pd.read_csv(r/name,dtype={'subset':str})
        assert len(missing[missing.split.str.match(r'^cv[0-4]$')])==n
        assert len(missing[missing.split.str.match(r'^cohort[1-5]$')])==extra
        assert len(missing)==n+extra and not missing.duplicated(keys).any()
    assert len(pd.read_csv(r/'parameters_all_runs.csv'))==405
    solver=pd.read_csv(r/'cox_solver_audit.csv')
    assert len(solver)==40 and solver.warning_count.sum()==0
    assert solver.absolute_cindex_difference.max()<1e-9
    coefficients=pd.read_csv(r/'canonical_loadings.csv')
    assert np.isfinite(coefficients.structure_correlation).all()
    assert coefficients.structure_correlation.abs().max()<=1.00001
    papers={}
    for name in ['main.pdf','appendix.pdf']:
        doc=fitz.open(ROOT/'manuscript'/name)
        assert len(doc)>0 and all(len(p.get_text().strip())>30 for p in doc)
        assert doc.metadata['author']=='EurekaChen'
        assert doc.metadata.get('title','')==''
        if name=='main.pdf':assert len(doc)==4
        papers[name]={'pages':len(doc),'author':doc.metadata['author'],
                      'text_characters':sum(len(p.get_text()) for p in doc)}
    report=dict(status='passed',cv_runs=300,source_group_runs=105,
        methods=20,patients=1416,deaths=830,missing_subset_runs=675,
        random_missing_runs=135,bootstrap_replicates=1000,
        patient_split_disjoint=True,aggregate_results_match_fold_records=True,
        paper_checks=papers,visual_review='Not performed; structure and text checks only')
    (r/'validation.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':validate()
