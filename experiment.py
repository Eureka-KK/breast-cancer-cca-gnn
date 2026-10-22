"""Resumable patient-level nested holdout evaluation. Private artifacts stay local."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
os.environ.setdefault('OPENBLAS_NUM_THREADS','2')
import argparse
import csv
import gc
import itertools
import json
import platform
import time
import traceback
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedKFold,train_test_split
from sksurv.util import Surv
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.ensemble import RandomSurvivalForest
from sksurv.metrics import concordance_index_ipcw,cumulative_dynamic_auc,brier_score
from sksurv.nonparametric import CensoringDistributionEstimator
from threadpoolctl import threadpool_limits
from fusion import Alignment,Preprocessor,NeuralFit,cindex
from forest_survival import ForestSurvival

ROOT=Path(__file__).resolve().parent
CFG=json.loads((ROOT/'config.json').read_text())
BASELINES=['clinical_cox','clinical_structured_cox','mrna_cox','methylation_cox','cna_cox','concat_cox',
           'rsf','deepsurv','pairwise_cox','gcca_cox','original_graph',
           'multisurv','mogonet_cox','proposed']
ABLATIONS=['no_anchor','no_graph','no_mask','no_residual','structured_clinical']


def candidates():
    rng=np.random.default_rng(42)
    out=[dict(features=1000,rank=8,ridge=.1,anchor=2.,k=10,hidden=32,
              lr=.01,mask=.2,alpha=.01,leaf=10)]
    for j in range(CFG['trials']-1):
        out.append(dict(features=int(rng.choice(CFG['feature_counts'])),
            rank=int(rng.choice(CFG['ranks'])),ridge=float(rng.choice(CFG['ridges'])),
            anchor=float(rng.choice(CFG['anchors'])),k=int(rng.choice(CFG['neighbors'])),
            hidden=int(rng.choice(CFG['hidden_sizes'])),lr=float(rng.choice(CFG['learning_rates'])),
            mask=float(rng.choice(CFG['mask_rates'])),alpha=float(np.logspace(-3,0,12)[j]),
            leaf=int(rng.choice([5,10,20]))))
    for x in out:x.update(epochs=CFG['epochs'],patience=CFG['patience'])
    return out


def append_csv(path,row):
    exists=path.exists()
    with path.open('a',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=list(row));
        if not exists:w.writeheader()
        w.writerow(row)


def breslow_survival(time,event,risk,query_risk,times):
    offset=float(np.mean(risk))
    r=np.clip(risk-offset,-40,40)
    unique=np.unique(time[event])
    increments=np.array([np.sum(event & (time==t))/np.exp(r[time>=t]).sum() for t in unique])
    cumulative=np.cumsum(increments)
    at=np.searchsorted(unique,np.asarray(times),side='right')-1
    base=np.where(at>=0,cumulative[np.maximum(at,0)],0.)
    return np.exp(-np.exp(np.clip(query_risk-offset,-40,40))[:,None]*base[None,:])


def metrics(train_time,train_event,test_time,test_event,risk,surv):
    tr=Surv.from_arrays(train_event,train_time);te=Surv.from_arrays(test_event,test_time)
    out={'cindex':cindex(test_time,test_event,risk)}
    cen=CensoringDistributionEstimator().fit(tr)
    tau=min(np.quantile(train_time,.9),np.max(test_time)-1e-4)
    try:out['ipcw_cindex']=float(concordance_index_ipcw(tr,te,risk,tau=tau)[0])
    except ValueError:out['ipcw_cindex']=None
    for j,t in enumerate([12.,36.,60.]):
        name=f'{int(t)}m'
        if not test_time.min()<t<min(test_time.max(),train_time.max()) or cen.predict_proba([t])[0]<=.05:
            out[f'auc_{name}']=None;out[f'brier_{name}']=None;continue
        # Samples beyond training support can be treated as administratively censored
        # just before the largest train follow-up; original outcomes remain for Harrell C.
        cap=np.nextafter(train_time.max(),-np.inf)
        tt=np.minimum(test_time,cap);ee=test_event & (test_time<=cap)
        yy=Surv.from_arrays(ee,tt)
        try:out[f'auc_{name}']=float(cumulative_dynamic_auc(tr,yy,risk,[t])[0][0])
        except ValueError:out[f'auc_{name}']=None
        try:out[f'brier_{name}']=float(brier_score(tr,yy,surv[:,j:j+1],[t])[1][0])
        except ValueError:out[f'brier_{name}']=None
    return out


class Fit:
    def __init__(self,method,z,t,e,vz,vt,ve,cfg,seed):
        self.method=method;self.nparams=0;self.epoch=0
        self.neural=None
        if (method.endswith('_cox') and method!='mogonet_cox') or method in ['gcca_cox','pairwise_cox']:
            self.model=CoxnetSurvivalAnalysis(l1_ratio=CFG['cox_l1_ratio'],alphas=[cfg['alpha']],
                fit_baseline_model=False,max_iter=CFG['cox_max_iterations'],tol=CFG['cox_tolerance'])
            a=self.flatten(z)
            with warnings.catch_warnings():
                warnings.simplefilter('ignore');self.model.fit(a,Surv.from_arrays(e,t))
            self.nparams=int(np.count_nonzero(self.model.coef_))
        elif method=='rsf':
            self.model=RandomSurvivalForest(n_estimators=CFG['rsf_trees'],min_samples_leaf=cfg['leaf'],
                max_features='sqrt',n_jobs=CFG['rsf_jobs'],random_state=seed,low_memory=True)
            a=np.concatenate(z,1)
            self.model.fit(a,Surv.from_arrays(e,t))
            self.survival=ForestSurvival(self.model,a,t,e,np.arange(1,31)*12)
            self.nparams=sum(x.tree_.node_count for x in self.model.estimators_)
        else:
            kind={'multisurv':'multisurv','mogonet_cox':'mogonet',
                  'deepsurv':'deep','no_graph':'mlp'}.get(method,'graph')
            rate=0 if method in ['original_graph','no_mask','deepsurv','mogonet_cox'] else cfg['mask']
            self.neural=NeuralFit(z,t,e,vz,vt,ve,cfg,seed,kind,rate,method!='no_residual')
            self.nparams=self.neural.nparams;self.epoch=self.neural.epoch

    def flatten(self,z):
        index={'clinical_cox':3,'clinical_structured_cox':3,'mrna_cox':0,'methylation_cox':1,'cna_cox':2}
        if self.method in index:return z[index[self.method]]
        return np.concatenate(z,1)

    def predict(self,z,mask=None):
        if self.neural:return self.neural.predict(z,mask)
        if mask is not None:z=[x*mask[:,i:i+1] for i,x in enumerate(z)]
        if self.method=='rsf':
            a=np.concatenate(z,1)
            surv=self.survival.predict(a)
            return self.model.predict(a),surv
        return self.model.predict(self.flatten(z)),None


def run_split(label,train,valid,test,raw,clinical,t,e,methods,trials):
    assert not(set(train)&set(valid) or set(train)&set(test) or set(valid)&set(test))
    np.savez(ROOT/f'.private/split_{label}.npz',train=train,valid=valid,test=test)
    cache={};align_cache={}
    def obtain(method,c):
        key=(c['features'],method in ['structured_clinical','clinical_structured_cox'])
        if key not in cache:
            pre=Preprocessor(*key).fit(raw,clinical,train)
            cache[key]=[pre.transform(raw,clinical,r) for r in [train,valid,test]]
        zs=cache[key]
        aligned=method in ['pairwise_cox','gcca_cox','original_graph','proposed']+ABLATIONS
        if not aligned:return zs
        anchor=1. if method in ['gcca_cox','no_anchor'] else c['anchor']
        pairwise=method in ['pairwise_cox','original_graph']
        akey=(*key,c['rank'],c['ridge'],anchor,pairwise)
        if akey not in align_cache:
            al=Alignment(c['rank'],c['ridge'],anchor,pairwise).fit(zs[0])
            align_cache[akey]=[al.transform(z) for z in zs]
        return align_cache[akey]
    selected={}
    def relevant(method,c):
        if method=='rsf':return (c['features'],c['leaf'])
        if method in ['clinical_cox','mrna_cox','methylation_cox','cna_cox','concat_cox']:
            return (c['features'],c['alpha'])
        return tuple(c.items())
    for method in methods:
        # Ablations lock the proposed model's selected settings within the same split.
        if method in ABLATIONS:
            selected[method]=dict(selected['proposed'])
            if method=='no_mask':selected[method]['mask']=0.
        else:
            selected_path=ROOT/f'.private/selected_{label}_{method}.json'
            if selected_path.exists():selected[method]=json.loads(selected_path.read_text())
            else:
                best=-float('inf')
                history=pd.read_csv(ROOT/'results/search.csv') if (ROOT/'results/search.csv').exists() else pd.DataFrame()
                seen={}
                for j,c in enumerate(trials):
                    signature=relevant(method,c)
                    if signature in seen:
                        score=seen[signature]
                        if score>best:best=score;selected[method]=dict(c)
                        continue
                    if len(history):
                        old=history[(history.split==label)&(history.method==method)&(history.trial==j)&(history.status=='ok')]
                        old=old[old.config.apply(lambda value:json.loads(value)==c)]
                        if len(old):
                            score=float(old.iloc[-1].validation_cindex)
                            seen[signature]=score
                            if score>best:best=score;selected[method]=dict(c)
                            continue
                    start=time.monotonic()
                    try:
                        z,vz,_=obtain(method,c)
                        fit=Fit(method,z,t[train],e[train],vz,t[valid],e[valid],c,42)
                        score=cindex(t[valid],e[valid],fit.predict(vz)[0])
                        seen[signature]=score
                        if score>best:best=score;selected[method]=dict(c)
                        append_csv(ROOT/'results/search.csv',dict(split=label,method=method,trial=j,
                            status='ok',validation_cindex=score,seconds=round(time.monotonic()-start,3),config=json.dumps(c)))
                        del fit
                    except Exception as ex:
                        append_csv(ROOT/'results/search.csv',dict(split=label,method=method,trial=j,
                            status=type(ex).__name__,validation_cindex='',seconds=round(time.monotonic()-start,3),config=json.dumps(c)))
                        print('Trial failed',label,method,j,type(ex).__name__,str(ex)[:130],flush=True)
                if method not in selected:
                    print('All trials failed',label,method,flush=True);continue
                selected_path.write_text(json.dumps(selected[method],indent=2))
        c=selected[method]
        for seed in CFG['seeds']:
            path=ROOT/f'.private/pred_{label}_{method}_{seed}.npz'
            if path.exists():continue
            start=time.monotonic()
            try:
                z,vz,tz=obtain(method,c)
                fit=Fit(method,z,t[train],e[train],vz,t[valid],e[valid],c,seed)
                risk,discrete=fit.predict(tz)
                trainrisk,_=fit.predict(z)
                if fit.neural and fit.neural.kind in ['graph','mogonet']:
                    # Reference predictions must use training topology, not query attachment.
                    with torch.no_grad():
                        rr,_=fit.neural.model(fit.neural.z,torch.ones((len(train),4),device=fit.neural.device))
                        trainrisk=rr.cpu().numpy()
                surv=breslow_survival(t[train],e[train],trainrisk,risk,[12,36,60])
                if discrete is not None:surv=discrete[:,[0,2,4]]
                met=metrics(t[train],e[train],t[test],e[test],risk,surv)
                np.savez(path,rows=test,time=t[test],event=e[test],risk=risk,survival=surv,
                    percentile=np.searchsorted(np.sort(trainrisk),risk,side='right')/len(trainrisk),
                    threshold=np.median(trainrisk),high_risk=risk>np.median(trainrisk))
                append_csv(ROOT/'results/fold_results.csv',dict(split=label,method=method,seed=seed,
                    n_train=len(train),n_valid=len(valid),n_test=len(test),events=int(e[test].sum()),
                    **met,parameters=fit.nparams,best_epoch=fit.epoch,
                    seconds=round(time.monotonic()-start,3),config=json.dumps(c)))
                append_csv(ROOT/'results/selected_parameters.csv',dict(split=label,method=method,seed=seed,**c))
                if method in ['proposed','original_graph','no_mask']:
                    for bits in itertools.product([0,1],repeat=4):
                        if not any(bits):continue
                        mask=np.tile(bits,(len(test),1)).astype('float32')
                        rp,_=fit.predict(tz,mask)
                        append_csv(ROOT/'results/missing_subsets.csv',dict(split=label,method=method,seed=seed,
                            subset=''.join(map(str,bits)),cindex=cindex(t[test],e[test],rp)))
                    for probability in [.1,.3,.5]:
                        rng=np.random.default_rng(seed+int(probability*100))
                        mask=(rng.random((len(test),4))>=probability).astype('float32')
                        empty=mask.sum(1)==0;mask[empty,0]=1
                        rp,_=fit.predict(tz,mask)
                        append_csv(ROOT/'results/random_missing.csv',dict(split=label,method=method,seed=seed,
                            probability=probability,cindex=cindex(t[test],e[test],rp)))
                print(label,method,seed,'C=',round(met['cindex'],4),'sec=',round(time.monotonic()-start,1),flush=True)
                del fit;gc.collect()
                if torch.cuda.is_available():torch.cuda.empty_cache()
            except Exception as ex:
                append_csv(ROOT/'results/failures.csv',dict(split=label,method=method,seed=seed,
                           error=type(ex).__name__,detail=str(ex)[:250]))
                print('FAILED',label,method,seed,type(ex).__name__,str(ex)[:150],flush=True)
                traceback.print_exc()
    del cache,align_cache;gc.collect()


def main():
    a=argparse.ArgumentParser();a.add_argument('--mode',choices=['cv','cohort','smoke'],default='cv')
    a.add_argument('--methods',nargs='+');a.add_argument('--fold',type=int)
    args=a.parse_args()
    torch.set_num_threads(CFG['threads'])
    raw=[np.load(ROOT/f'.private/view{i}.npy',mmap_mode='r') for i in range(3)]
    clinical=pd.read_csv(ROOT/'.private/clinical.csv');out=np.load(ROOT/'.private/outcomes.npz')
    t=out['time'];e=out['event'];cohort=out['cohort']
    trials=candidates()
    (ROOT/'results/search_candidates.json').write_text(json.dumps(trials,indent=2))
    env={'python':platform.python_version(),'numpy':np.__version__,'torch':torch.__version__,
         'gpu':torch.cuda.get_device_name() if torch.cuda.is_available() else None,'config':CFG}
    (ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
    methods=args.methods or BASELINES+ABLATIONS
    if args.mode=='smoke':
        trials=trials[:1];trials[0]['epochs']=3;CFG['seeds']=[42]
    if args.mode=='cohort':
        splits=[(int(c),np.flatnonzero(cohort!=c),np.flatnonzero(cohort==c)) for c in np.unique(cohort)]
    else:
        splits=[(i,tr,te) for i,(tr,te) in enumerate(StratifiedKFold(CFG['outer_folds'],shuffle=True,random_state=CFG['split_seed']).split(t,e))]
    with threadpool_limits(CFG['threads']):
        for i,development,test in splits:
            if args.fold is not None and args.fold!=i:continue
            train,valid=train_test_split(development,test_size=CFG['validation_fraction'],random_state=CFG['split_seed'],stratify=e[development])
            label=f'{args.mode}{i}'
            run_split(label,train,valid,test,raw,clinical,t,e,methods,trials)
            if args.mode=='smoke':break


if __name__=='__main__':main()
