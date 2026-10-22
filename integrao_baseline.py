"""Official IntegrAO 0.1.3 + Cox, with one-way inductive query edges.

This is an explicitly documented survival adapter, not the original downstream
classification task. No test covariates enter graph diffusion or model fitting.
"""
import os
os.environ.setdefault('OMP_NUM_THREADS','2')
os.environ.setdefault('MKL_NUM_THREADS','2')
import argparse
import contextlib
import gc
import json
import time
import warnings
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import pairwise_distances
from sklearn.model_selection import StratifiedKFold,train_test_split
from sksurv.linear_model import CoxnetSurvivalAnalysis
from sksurv.util import Surv
from threadpoolctl import threadpool_limits
from fusion import Preprocessor,cindex
from experiment import ROOT,CFG,append_csv,breslow_survival,metrics


def embed_query(model,z,reference,edges,k=10):
    device=next(model.parameters()).device;n=len(reference[0])
    xd={};ed={}
    for i,(q,r,edge) in enumerate(zip(z,reference,edges)):
        distances=pairwise_distances(q,r,metric='euclidean')
        nearest=np.argsort(distances,axis=1)[:,:k]
        # Source=train, target=query. Root term is handled inside GraphSAGE.
        new=torch.tensor(np.stack([nearest.ravel(),np.repeat(np.arange(len(q))+n,k)]),
                         dtype=torch.long,device=device)
        xd[i]=torch.tensor(np.concatenate([r,q],0),dtype=torch.float32,device=device)
        ed[i]=torch.cat([edge,new],1)
    with torch.no_grad():out=model(xd,ed)
    return torch.stack([out[i][n:] for i in range(4)]).mean(0).cpu().numpy()


def run(mode='cv',only_fold=None):
    from integrao.integrater import integrao_integrater
    from integrao.unsupervised_train import tsne_p_deep
    from integrao.dataset import GraphDataset
    warnings.filterwarnings('ignore',category=FutureWarning)
    warnings.filterwarnings('ignore',category=UserWarning)
    torch.set_num_threads(2)
    raw=[np.load(ROOT/f'.private/view{i}.npy',mmap_mode='r') for i in range(3)]
    clinical=pd.read_csv(ROOT/'.private/clinical.csv');out=np.load(ROOT/'.private/outcomes.npz')
    t=out['time'];e=out['event'];cohort=out['cohort']
    if mode=='cv':splits=list(enumerate(StratifiedKFold(5,shuffle=True,random_state=42).split(t,e)))
    else:splits=[(int(c),(np.flatnonzero(cohort!=c),np.flatnonzero(cohort==c))) for c in np.unique(cohort)]
    for fold,(development,test) in splits:
        if only_fold is not None and only_fold!=fold:continue
        train,valid=train_test_split(development,test_size=.2,random_state=42,stratify=e[development])
        label=f'{mode}{fold}'
        pre=Preprocessor(1000).fit(raw,clinical,train)
        z,vz,tz=[pre.transform(raw,clinical,r) for r in [train,valid,test]]
        graph_cache=ROOT/f'.private/integrao_graph_{label}.npz'
        ids=[f'row_{j:04}' for j in range(len(train))]
        with (ROOT/'.private/integrao_console.log').open('a',encoding='utf-8') as log:
            with contextlib.redirect_stdout(log),threadpool_limits(2):
                integrater=integrao_integrater([pd.DataFrame(x,index=ids) for x in z],
                    neighbor_size=10,embedding_dims=8,fusing_iteration=20,alighment_epochs=300)
                if graph_cache.exists():
                    a=np.load(graph_cache);networks=[a[f'arr_{i}'] for i in range(4)]
                else:
                    networks=[x.to_numpy() for x in integrater.network_diffusion()]
                    np.savez(graph_cache,*networks)
                device='cuda' if torch.cuda.is_available() else 'cpu'
                edges=[GraphDataset(10,x,p)[0].edge_index.to(device) for x,p in zip(z,networks)]
        for seed in CFG['seeds']:
            dest=ROOT/f'.private/pred_{label}_integrao_cox_{seed}.npz'
            if dest.exists():continue
            start=time.monotonic();torch.manual_seed(seed);np.random.seed(seed)
            try:
                with (ROOT/'.private/integrao_console.log').open('a',encoding='utf-8') as log:
                    with contextlib.redirect_stdout(log),threadpool_limits(2):
                        _,model=tsne_p_deep(integrater.dicts_commonIndex,integrater.dict_sampleToIndexs,z,
                            P=[p.copy() for p in networks],neighbor_size=10,embedding_dims=8,alighment_epochs=300)
                model.eval()
                with torch.no_grad():
                    tr=model({i:torch.tensor(x,device=device) for i,x in enumerate(z)},dict(enumerate(edges)))
                    tr=torch.stack(list(tr.values())).mean(0).cpu().numpy()
                vv=embed_query(model,vz,z,edges);qq=embed_query(model,tz,z,edges)
                # Exercise the actual official GraphSAGE adapter for batch independence.
                check=embed_query(model,[x[:1] for x in tz],z,edges)
                assert np.allclose(qq[:1],check,atol=2e-5), 'IntegrAO query-batch dependence'
                best=-np.inf;winner=None;alpha=None
                for j,a in enumerate(np.logspace(-3,1,12)):
                    fit=CoxnetSurvivalAnalysis(l1_ratio=.1,alphas=[a],max_iter=10000).fit(tr,Surv.from_arrays(e[train],t[train]))
                    score=cindex(t[valid],e[valid],fit.predict(vv))
                    if score>best:best=score;winner=fit;alpha=float(a)
                risk=winner.predict(qq);trainrisk=winner.predict(tr)
                surv=breslow_survival(t[train],e[train],trainrisk,risk,[12,36,60])
                met=metrics(t[train],e[train],t[test],e[test],risk,surv)
                np.savez(dest,rows=test,time=t[test],event=e[test],risk=risk,survival=surv,
                    percentile=np.searchsorted(np.sort(trainrisk),risk,side='right')/len(trainrisk),
                    threshold=np.median(trainrisk),high_risk=risk>np.median(trainrisk))
                config=dict(features=1000,rank=8,hidden=128,k=10,epochs=300,
                    diffusion_iterations=20,alpha=alpha,cox_l1_ratio=.1,
                    official_package='integrao==0.1.3',inductive_adapter=True)
                append_csv(ROOT/'results/fold_results.csv',dict(split=label,method='integrao_cox',seed=seed,
                    n_train=len(train),n_valid=len(valid),n_test=len(test),events=int(e[test].sum()),**met,
                    parameters=sum(p.numel() for p in model.parameters()),best_epoch=300,
                    seconds=round(time.monotonic()-start,3),config=json.dumps(config)))
                print(label,'integrao_cox',seed,'C=',round(met['cindex'],4),'sec=',round(time.monotonic()-start,1),flush=True)
                del model;gc.collect();torch.cuda.empty_cache()
            except Exception as ex:
                append_csv(ROOT/'results/failures.csv',dict(split=label,method='integrao_cox',seed=seed,
                           error=type(ex).__name__,detail=str(ex)[:250]))
                print(label,'IntegrAO failed',type(ex).__name__,str(ex)[:160],flush=True)


if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--mode',default='cv',choices=['cv','cohort']);a.add_argument('--fold',type=int)
    args=a.parse_args();run(args.mode,args.fold)
