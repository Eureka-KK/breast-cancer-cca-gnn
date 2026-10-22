"""Aggregate only de-identified summaries; conditional patient bootstrap, no refits."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test
from threadpoolctl import threadpool_limits
from experiment import ROOT,CFG

NAMES={'clinical_cox':'Templated clinical Cox','clinical_structured_cox':'Structured clinical Cox','mrna_cox':'mRNA Cox','methylation_cox':'Methylation Cox',
       'cna_cox':'CNA Cox','concat_cox':'Concatenation Cox','rsf':'Random survival forest',
       'deepsurv':'DeepSurv-style MLP','pairwise_cox':'Pairwise CCA + Cox','gcca_cox':'GCCA + Cox',
       'original_graph':'Original pairwise CCA + graph','multisurv':'MultiSurv-style (scaled)',
       'mogonet_cox':'MOGONET-style + Cox','integrao_cox':'Official IntegrAO + Cox adapter',
       'proposed':'Shared-anchor GCCA + graph','no_anchor':'Equal GCCA weights',
       'no_graph':'Without graph','no_mask':'Without view masking',
       'no_residual':'Without raw residual','structured_clinical':'Structured clinical input'}


def comparable(time,event):
    return event[:,None] & ((time[:,None]<time[None,:])|
                ((time[:,None]==time[None,:]) & ~event[None,:]))


def bootstrap_method(method,mode='cv',draws=1000):
    values=[];point=[]
    splits=[f'cv{i}' for i in range(5)] if mode=='cv' else [f'cohort{i}' for i in range(1,6)]
    for split in splits:
        paths=[ROOT/f'.private/pred_{split}_{method}_{s}.npz' for s in CFG['seeds']]
        if not all(p.exists() for p in paths):return None,None
        arrays=[np.load(p) for p in paths]
        t=arrays[0]['time'];e=arrays[0]['event'];n=len(t)
        comp=comparable(t,e).astype(float)
        credit=np.zeros((n,n))
        for a in arrays:
            diff=a['risk'][:,None]-a['risk'][None,:]
            credit+=((diff>1e-8)+.5*(np.abs(diff)<=1e-8))*comp/len(arrays)
        rng=np.random.default_rng(42+sum(map(ord,split)))
        weights=rng.multinomial(n,np.full(n,1/n),size=draws).astype(float)
        den=np.einsum('ij,ij->i',weights@comp,weights)
        num=np.einsum('ij,ij->i',weights@credit,weights)
        values.append(num/den)
        point.append(credit.sum()/comp.sum())
    return float(np.mean(point)),np.mean(values,axis=0)


def summarize():
    result=ROOT/'results';fig=ROOT/'paper';fig.mkdir(exist_ok=True)
    d=pd.read_csv(result/'fold_results.csv')
    cv=d[d.split.str.match(r'^cv\d+$')]
    rows=[];boot={}
    with threadpool_limits(2):
        for method in cv.method.unique():
            point,b=bootstrap_method(method,draws=CFG['bootstrap'])
            if point is None:continue
            boot[method]=b;sub=cv[cv.method==method]
            lo,hi=np.quantile(b,[.025,.975])
            row=dict(method=method,label=NAMES[method],cindex=point,ci_low=lo,ci_high=hi,
                     run_sd=sub.cindex.std(),fold_seed_runs=len(sub),parameters_median=sub.parameters.median())
            for name in ['ipcw_cindex','auc_12m','auc_36m','auc_60m','brier_12m','brier_36m','brier_60m']:
                row[name]=sub[name].mean()
            rows.append(row)
    summary=pd.DataFrame(rows)
    if summary.empty:return
    summary.sort_values('cindex',ascending=False).to_csv(result/'main_results.csv',index=False)
    tests=[]
    if 'proposed' in boot:
        for method,b in boot.items():
            if method=='proposed':continue
            delta=boot['proposed']-b
            p=min(1.,2*min((np.sum(delta<=0)+1)/(len(delta)+1),(np.sum(delta>=0)+1)/(len(delta)+1)))
            tests.append(dict(comparator=method,delta=float(np.mean(delta)),
                ci_low=float(np.quantile(delta,.025)),ci_high=float(np.quantile(delta,.975)),p=p))
        tests=sorted(tests,key=lambda r:r['p']);previous=0
        for i,row in enumerate(tests):
            previous=max(previous,min(1.,row['p']*(len(tests)-i)));row['p_holm']=previous
        pd.DataFrame(tests).to_csv(result/'paired_comparisons.csv',index=False)
    # Cohort-level results: labels are source groups, not external cohorts.
    group=d[d.split.str.startswith('cohort')]
    if len(group):
        group.groupby(['method','split']).agg(cindex=('cindex','mean'),
            seed_sd=('cindex','std'),n_test=('n_test','first'),events=('events','first')).reset_index().to_csv(result/'cohort_generalization.csv',index=False)
    for src,dest,keys in [('missing_subsets.csv','missing_summary.csv',['method','subset']),
                          ('random_missing.csv','random_missing_summary.csv',['method','probability'])]:
        if (result/src).exists():
            a=pd.read_csv(result/src,dtype={'subset':str});a=a[a.split.str.startswith('cv')]
            a.groupby(keys).agg(cindex=('cindex','mean'),sd=('cindex','std'),runs=('cindex','size')).reset_index().to_csv(result/dest,index=False)
    # Compact actual-result figures. No synthetic data or photorealistic image generation.
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':8,'axes.spines.top':False,'axes.spines.right':False})
    main=summary[~summary.method.isin(['no_anchor','no_graph','no_mask','no_residual','structured_clinical'])].sort_values('cindex')
    f,ax=plt.subplots(figsize=(6.8,3.6))
    ax.errorbar(main.cindex,range(len(main)),xerr=[main.cindex-main.ci_low,main.ci_high-main.cindex],fmt='o',color='black',capsize=2,ms=3)
    ax.set_yticks(range(len(main)),main.label);ax.set_xlabel('Harrell C-index (conditional patient-bootstrap 95% CI)')
    ax.axvline(.5,color='.7',lw=.7);f.tight_layout();f.savefig(fig/'comparison.pdf');plt.close(f)
    if (result/'missing_summary.csv').exists():
        a=pd.read_csv(result/'missing_summary.csv',dtype={'subset':str});a=a[a.method=='proposed'].copy()
        if len(a):
            a=a.sort_values('subset');f,ax=plt.subplots(figsize=(6.8,2.5))
            ax.bar(np.arange(len(a)),a.cindex,color='.45')
            ax.set_xticks(np.arange(len(a)),a.subset.astype(str).str.zfill(4),rotation=45)
            ax.set_ylim(.3,max(.8,a.cindex.max()+.03));ax.set_ylabel('C-index');ax.set_xlabel('Available views: mRNA / methylation / CNA / clinical (1 = available)')
            f.tight_layout();f.savefig(fig/'missing.pdf');plt.close(f)
    # Pooled held-out risk groups using majority vote over train-defined seed thresholds.
    if 'proposed' in boot:
        tables=[]
        for i in range(5):
            arrays=[np.load(ROOT/f'.private/pred_cv{i}_proposed_{s}.npz') for s in CFG['seeds']]
            tables.append(pd.DataFrame({'time':arrays[0]['time'],'event':arrays[0]['event'],
                'high':np.stack([x['high_risk'] for x in arrays]).mean(0)>=.5,
                's36':np.stack([x['survival'][:,1] for x in arrays]).mean(0),
                's60':np.stack([x['survival'][:,2] for x in arrays]).mean(0)}))
        oof=pd.concat(tables,ignore_index=True)
        test=logrank_test(oof.loc[oof.high,'time'],oof.loc[~oof.high,'time'],
                         oof.loc[oof.high,'event'],oof.loc[~oof.high,'event'])
        (result/'risk_groups.json').write_text(json.dumps({'high_n':int(oof.high.sum()),
            'low_n':int((~oof.high).sum()),'logrank_p':float(test.p_value),
            'threshold':'each training fold median; seed-majority vote'},indent=2))
        f,axes=plt.subplots(1,2,figsize=(6.8,2.6))
        for high,label,style in [(False,'Lower predicted risk','-'),(True,'Higher predicted risk','--')]:
            a=oof[oof.high==high];km=KaplanMeierFitter().fit(a.time,a.event,label=label)
            km.plot_survival_function(ax=axes[0],ci_show=False,color='black',ls=style)
        axes[0].set_xlabel('Months');axes[0].set_ylabel('Overall survival');axes[0].set_xlim(0,240)
        axes[0].set_title(f'Out-of-fold groups, log-rank p={test.p_value:.2g}')
        calibration=[]
        for horizon,column,marker in [(36,'s36','o'),(60,'s60','s')]:
            bins=pd.qcut(1-oof[column],5,labels=False,duplicates='drop')
            x=[];y=[]
            for b in sorted(bins.unique()):
                a=oof[bins==b];km=KaplanMeierFitter().fit(a.time,a.event)
                predicted=float(1-a[column].mean());observed=float(1-km.predict(horizon))
                x.append(predicted);y.append(observed)
                calibration.append(dict(horizon_months=horizon,bin=int(b),n=len(a),
                    predicted_death_probability=predicted,km_death_probability=observed))
            axes[1].plot(x,y,marker=marker,color='black',label=f'{horizon} months',ls='-' if horizon==36 else '--')
        axes[1].plot([0,1],[0,1],color='.6',lw=.7);axes[1].set_xlabel('Predicted death probability')
        axes[1].set_ylabel('Kaplan–Meier death probability');axes[1].legend(frameon=False)
        f.tight_layout();f.savefig(fig/'calibration_km.pdf');plt.close(f)
        pd.DataFrame(calibration).to_csv(result/'calibration.csv',index=False)
    # Every fixed setting plus chosen fold/seed settings are available as tables.
    pd.DataFrame([{'parameter':k,'value':json.dumps(v)} for k,v in CFG.items()]).to_csv(result/'all_fixed_parameters.csv',index=False)
    expanded=[]
    for _,row in d.iterrows():
        expanded.append(dict(split=row.split,method=row.method,seed=row.seed,
            best_epoch=row.best_epoch,size=row.parameters,**json.loads(row.config)))
    pd.DataFrame(expanded).to_csv(result/'parameters_all_runs.csv',index=False)
    pd.DataFrame(json.loads((result/'search_candidates.json').read_text())).to_csv(result/'search_candidates.csv',index=False)
    status=dict(completed_cv_methods=summary.method.tolist(),
        expected_cv_methods=20,completed_cv_runs=len(cv),expected_cv_runs=300,
        completed_cohort_runs=len(group),expected_cohort_runs=105,
        bootstrap_replicates=CFG['bootstrap'],
        uncertainty_scope='Conditional on fitted models; stratified by held-out fold; no model refits')
    (result/'completion.json').write_text(json.dumps(status,indent=2))
    print(summary[['label','cindex','ci_low','ci_high']].round(4).to_string(index=False))


if __name__=='__main__':summarize()
