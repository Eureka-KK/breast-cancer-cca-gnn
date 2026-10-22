"""Re-fit selected main-CV Cox solutions and report convergence/reproducibility."""
import argparse
import json
import warnings
import numpy as np
import pandas as pd
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from threadpoolctl import threadpool_limits
from fusion import Preprocessor,Alignment,cindex
from experiment import ROOT,CFG


def main(only_fold=None):
    raw=[np.load(ROOT/f'.private/view{i}.npy',mmap_mode='r') for i in range(3)]
    clinical=pd.read_csv(ROOT/'.private/clinical.csv')
    outcome=np.load(ROOT/'.private/outcomes.npz');t=outcome['time'];e=outcome['event']
    result=pd.read_csv(ROOT/'results/fold_results.csv')
    methods=['clinical_cox','clinical_structured_cox','mrna_cox','methylation_cox',
             'cna_cox','concat_cox','pairwise_cox','gcca_cox']
    records=[]
    with threadpool_limits(2):
        for fold in range(5):
            if only_fold is not None and fold!=only_fold:continue
            split=np.load(ROOT/f'.private/split_cv{fold}.npz');train=split['train'];test=split['test']
            cache={}
            for method in methods:
                rows=result[(result.split==f'cv{fold}')&(result.method==method)&(result.seed==42)]
                if not len(rows):continue
                c=json.loads(rows.iloc[0].config);key=(c['features'],method=='clinical_structured_cox')
                if key not in cache:
                    pre=Preprocessor(*key).fit(raw,clinical,train)
                    cache[key]=[pre.transform(raw,clinical,r) for r in [train,test]]
                z,q=cache[key]
                if method in ['pairwise_cox','gcca_cox']:
                    a=Alignment(c['rank'],c['ridge'],1. if method=='gcca_cox' else c['anchor'],method=='pairwise_cox').fit(z)
                    z,q=a.transform(z),a.transform(q)
                index={'clinical_cox':3,'clinical_structured_cox':3,'mrna_cox':0,'methylation_cox':1,'cna_cox':2}
                x=z[index[method]] if method in index else np.concatenate(z,1)
                query=q[index[method]] if method in index else np.concatenate(q,1)
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    model=CoxnetSurvivalAnalysis(l1_ratio=.1,alphas=[c['alpha']],
                        fit_baseline_model=False,max_iter=10000,tol=1e-6).fit(x,Surv.from_arrays(e[train],t[train]))
                score=cindex(t[test],e[test],model.predict(query))
                records.append(dict(split=f'cv{fold}',method=method,alpha=c['alpha'],
                    reproduced_cindex=score,original_cindex=rows.iloc[0].cindex,
                    absolute_cindex_difference=abs(score-rows.iloc[0].cindex),
                    warning_count=len(captured),warnings='; '.join(str(w.message)[:300] for w in captured)))
    dest=ROOT/'results/cox_solver_audit.csv'
    frame=pd.DataFrame(records)
    if only_fold is not None and dest.exists():
        old=pd.read_csv(dest);old=old[old.split!=f'cv{only_fold}'];frame=pd.concat([old,frame],ignore_index=True)
    frame.to_csv(dest,index=False)
    print('Cox solutions audited',len(frame),'maximum C-index difference',frame.absolute_cindex_difference.max(),
          'warnings',frame.warning_count.sum())


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--fold',type=int)
    main(a.parse_args().fold)
