"""Training-only preprocessing, joint GCCA, pairwise anchor CCA and inductive GCN."""
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.linalg as la
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sksurv.metrics import concordance_index_censored
SETTINGS=json.loads((Path(__file__).parent/'config.json').read_text())


def cindex(time, event, risk):
    return float(concordance_index_censored(event.astype(bool), time, risk)[0])


class Preprocessor:
    def __init__(self, features=1000, structured=False):
        self.features=features; self.structured=structured

    def fit(self, views, clinical, train):
        self.transforms=[]
        for x in views:
            a=np.asarray(x[train],dtype=float)
            valid=np.isfinite(a).mean(0)>=1-SETTINGS['feature_max_missing_fraction']
            a[~np.isfinite(a)]=np.nan
            with np.errstate(invalid='ignore'):
                var=np.nanvar(a,axis=0)
            var[~valid | ~np.isfinite(var)]=-np.inf
            ids=np.argsort(var)[::-1][:min(self.features,int(valid.sum()))]
            a=a[:,ids]; med=np.nanmedian(a,axis=0)
            a=np.where(np.isfinite(a),a,med)
            mean=a.mean(0); scale=a.std(0); scale[scale<1e-8]=1
            self.transforms.append((ids,med,mean,scale))
        self.num=['AGE_AT_DIAGNOSIS','TUMOR_STAGE','GRADE','TUMOR_SIZE']
        self.cat=[x for x in clinical.columns if x not in self.num]
        if self.structured:
            a=clinical.iloc[train][self.num].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            self.median=np.nanmedian(a,0)
            self.scaler=StandardScaler().fit(np.where(np.isfinite(a),a,self.median))
            self.encoder=OneHotEncoder(handle_unknown='ignore',sparse_output=False)
            self.encoder.fit(clinical.iloc[train][self.cat].fillna('unknown').astype(str))
        else:
            self.vectorizer=TfidfVectorizer(ngram_range=tuple(SETTINGS['tfidf_ngram_range']),min_df=SETTINGS['tfidf_min_df'],max_features=SETTINGS['tfidf_max_features'],
                                           token_pattern=r'(?u)\b[\w.]+\b')
            a=self.vectorizer.fit_transform(self.narrative(clinical.iloc[train]))
            self.svd=TruncatedSVD(n_components=min(SETTINGS['text_svd_max_dimensions'],a.shape[1]-1,a.shape[0]-1),random_state=SETTINGS['split_seed'])
            z=self.svd.fit_transform(a)
            self.scaler=StandardScaler().fit(z)
        return self

    @staticmethod
    def narrative(clinical):
        return ['. '.join(f'{k.lower()} {str(v) if pd.notna(v) else "unknown"}'
                          for k,v in row.items()) for row in clinical.to_dict('records')]

    def transform(self, views, clinical, rows):
        zs=[]
        for x,(ids,med,mean,scale) in zip(views,self.transforms):
            a=np.asarray(x[np.ix_(rows,ids)],dtype=float)
            bound=SETTINGS['feature_clip_standard_deviations']
            zs.append(np.clip((np.where(np.isfinite(a),a,med)-mean)/scale,-bound,bound).astype('float32'))
        c=clinical.iloc[rows]
        if self.structured:
            a=c[self.num].apply(pd.to_numeric,errors='coerce').to_numpy(float)
            z=np.concatenate([self.scaler.transform(np.where(np.isfinite(a),a,self.median)),
                (~np.isfinite(a)).astype(float),
                self.encoder.transform(c[self.cat].fillna('unknown').astype(str))],1)
        else:
            z=self.scaler.transform(self.svd.transform(self.vectorizer.transform(self.narrative(c))))
        zs.append(z.astype('float32'))
        return zs


def inverse_sqrt(a):
    w,v=la.eigh(a,check_finite=False)
    return (v*(1/np.sqrt(np.maximum(w,1e-10))))@v.T


class Alignment:
    """GCCA solves max trace(G' sum_v a_v H_v G), G'G=I, H_v ridge hats.

    Clinical weight a_4 is an explicit anchor strength. All projections are
    jointly referenced to the same training-only G; not a novel GCCA estimator.
    """
    def __init__(self,rank=8,ridge=.1,anchor=2.,pairwise=False):
        self.rank=rank;self.ridge=ridge;self.anchor=anchor;self.pairwise=pairwise

    def fit(self,z):
        self.mean=[x.mean(0) for x in z]
        xs=[(x-m).astype(float) for x,m in zip(z,self.mean)]
        n=len(xs[0]); r=min(self.rank,min(x.shape[1] for x in xs),n-1)
        self.weights=[]
        if self.pairwise:
            c=xs[-1]; ic=inverse_sqrt(c.T@c/n+self.ridge*np.eye(c.shape[1]))
            anchors=[]
            for x in xs[:-1]:
                ix=inverse_sqrt(x.T@x/n+self.ridge*np.eye(x.shape[1]))
                u,_,vh=la.svd(ix@(x.T@c/n)@ic,full_matrices=False,check_finite=False)
                self.weights.append(ix@u[:,:r]);anchors.append(ic@vh.T[:,:r])
            self.weights.append(np.mean(anchors,0))
        else:
            hats=[];maps=[]
            for x in xs:
                # Smaller of primal and dual avoids high-dimensional inversion.
                if x.shape[1] <= n:
                    b=la.solve(x.T@x+n*self.ridge*np.eye(x.shape[1]),x.T,assume_a='pos',check_finite=False)
                else:
                    b=la.solve(x@x.T+n*self.ridge*np.eye(n),x,assume_a='pos',check_finite=False).T
                maps.append(b);hats.append(x@b)
            h=sum(w*a for w,a in zip([1,1,1,self.anchor],hats))
            _,g=la.eigh(h,subset_by_index=[n-r,n-1],check_finite=False)
            g=g[:,::-1]*np.sqrt(n)
            self.weights=[b@g for b in maps]
        projected=[x@w for x,w in zip(xs,self.weights)]
        self.scale=[np.maximum(a.std(0),1e-6) for a in projected]
        return self

    def transform(self,z):
        return [((x-m)@w/s).astype('float32') for x,m,w,s in zip(z,self.mean,self.weights,self.scale)]


def cox_loss(risk,time,event):
    """Exact Breslow risk sets including all observations at each tied time."""
    order=torch.argsort(time,descending=True,stable=True)
    t=time[order]; r=risk[order]; e=event[order]
    _,counts=torch.unique_consecutive(t,return_counts=True)
    ends=torch.cumsum(counts,0)-1
    denominators=torch.repeat_interleave(torch.logcumsumexp(r,0)[ends],counts)
    return ((denominators-r)*e).sum()/e.sum().clamp_min(1)


def cosine_graph(z,k=10):
    a=z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1e-8)
    similarity=a@a.T
    similarity.fill_diagonal_(-float('inf'))
    idx=similarity.topk(min(k,len(z)-1),dim=1).indices
    adj=torch.zeros_like(similarity);adj.scatter_(1,idx,1)
    adj=torch.maximum(adj,adj.T);adj.fill_diagonal_(1)
    degree=adj.sum(1)
    return adj/torch.sqrt(degree[:,None]*degree[None,:]), degree


def query_graph(z,reference,k,degree):
    a=z/torch.linalg.vector_norm(z,dim=1,keepdim=True).clamp_min(1e-8)
    b=reference/torch.linalg.vector_norm(reference,dim=1,keepdim=True).clamp_min(1e-8)
    idx=(a@b.T).topk(min(k,len(reference)),dim=1).indices
    adj=torch.zeros((len(z),len(reference)),device=z.device)
    adj.scatter_(1,idx,1)
    qdegree=adj.sum(1)+1
    return adj/torch.sqrt(qdegree[:,None]*degree[None,:]),1/qdegree


class Network(nn.Module):
    def __init__(self,dims,kind='graph',hidden=32,dropout=.3,k=10,residual=True):
        super().__init__();self.kind=kind;self.k=k;self.residual=residual
        self.dropout=nn.Dropout(dropout)
        if kind=='multisurv':
            self.encoders=nn.ModuleList([nn.Sequential(nn.Linear(d,128),nn.ReLU(),
                nn.Dropout(dropout),nn.Linear(128,hidden),nn.ReLU()) for d in dims])
            self.head=nn.Sequential(nn.Linear(hidden,hidden),nn.ReLU(),nn.Linear(hidden,30))
        elif kind=='deep':
            self.mlp=nn.Sequential(nn.Linear(sum(dims),hidden),nn.ReLU(),nn.Dropout(dropout),
                                   nn.Linear(hidden,hidden),nn.ReLU(),nn.Dropout(dropout),nn.Linear(hidden,1))
        elif kind=='mogonet':
            self.encoders=nn.ModuleList([nn.Linear(d,hidden) for d in dims])
            self.layers=nn.ModuleList([nn.Linear(hidden,hidden) for _ in dims])
            self.head=nn.Linear(4*hidden,1)
        else:
            self.input=nn.Linear(sum(dims)+4,hidden)
            self.layers=nn.ModuleList([nn.Linear(hidden,hidden) for _ in range(2)])
            self.norms=nn.ModuleList([nn.LayerNorm(hidden) for _ in range(2)])
            self.head=nn.Linear(hidden+(sum(dims)+4 if residual else 0),1)

    def forward(self,views,mask,reference=None):
        xs=[x*mask[:,i:i+1] for i,x in enumerate(views)]
        if self.kind=='deep':return self.mlp(torch.cat(xs,1)).flatten(),None
        if self.kind=='multisurv':
            a=torch.stack([m(x) for m,x in zip(self.encoders,xs)],1)
            a=a.masked_fill(mask[:,:,None]==0,-1e6).max(1).values
            return self.head(a),None
        if self.kind=='mogonet':
            hs=[];refs=[]
            for i,(x,enc,layer) in enumerate(zip(xs,self.encoders,self.layers)):
                if reference is None:
                    adj,deg=cosine_graph(x,self.k)
                    h=torch.relu(enc(x)); out=torch.relu(layer(adj@h))
                    refs.append((x,deg,h))
                else:
                    rx,deg,rh=reference[i]
                    adj,selfw=query_graph(x,rx,self.k,deg)
                    h=torch.relu(enc(x));out=torch.relu(layer(adj@rh+selfw[:,None]*h))
                hs.append(self.dropout(out))
            return self.head(torch.cat(hs,1)).flatten(),refs
        raw=torch.cat(xs+[mask],1)
        h=torch.relu(self.input(raw))
        refs=[]
        if self.kind=='graph':
            shared=torch.stack(xs,1).sum(1)/mask.sum(1,keepdim=True).clamp_min(1)
            if reference is None:
                adj,deg=cosine_graph(shared,self.k)
            else:
                rz,deg,rhs=reference
                adj,selfw=query_graph(shared,rz,self.k,deg)
        for j,(layer,norm) in enumerate(zip(self.layers,self.norms)):
            if self.kind=='graph':
                refs.append(h)
                agg=adj@h if reference is None else adj@rhs[j]+selfw[:,None]*h
            else: agg=h
            h=norm(h+self.dropout(torch.relu(layer(agg))))
        return self.head(torch.cat([h,raw],1) if self.residual else h).flatten(),(
            (shared,deg,refs) if self.kind=='graph' and reference is None else None)


def discrete_loss(logits,time,event):
    # Fixed 12-month intervals, right censoring at 360 months; no label-derived boundaries.
    hazard=logits.sigmoid().clamp(1e-6,1-1e-6)
    idx=(time/12).long().clamp(max=29)
    ar=torch.arange(30,device=time.device)[None,:]
    survived=ar<idx[:,None]
    failure=(ar==idx[:,None]) & event.bool()[:,None] & (time<360)[:,None]
    return -(torch.log1p(-hazard)*survived+torch.log(hazard)*failure).sum(1).mean()


class NeuralFit:
    def __init__(self,z,time,event,valid_z,valid_time,valid_event,config,seed=42,
                 kind='graph',mask_rate=.2,residual=True):
        torch.manual_seed(seed);np.random.seed(seed)
        self.device='cuda' if torch.cuda.is_available() else 'cpu'
        self.z=[torch.as_tensor(x,device=self.device) for x in z]
        vz=[torch.as_tensor(x,device=self.device) for x in valid_z]
        t=torch.as_tensor(time,dtype=torch.float32,device=self.device)
        e=torch.as_tensor(event,dtype=torch.float32,device=self.device)
        self.model=Network([x.shape[1] for x in z],kind,config['hidden'],.3,config['k'],residual).to(self.device)
        self.kind=kind
        opt=torch.optim.AdamW(self.model.parameters(),lr=config['lr'],weight_decay=.001)
        best=-np.inf;state=None;stale=0
        full=torch.ones((len(t),4),device=self.device)
        for epoch in range(config.get('epochs',300)):
            self.model.train();opt.zero_grad()
            mask=(torch.rand(full.shape,device=self.device)>=mask_rate).float()
            empty=mask.sum(1)==0;mask[empty,0]=1
            pred,_=self.model(self.z,mask)
            loss=discrete_loss(pred,t,e) if kind=='multisurv' else cox_loss(pred,t,e)
            if not torch.isfinite(loss): raise ValueError('Nonfinite neural loss')
            loss.backward();nn.utils.clip_grad_norm_(self.model.parameters(),5.);opt.step()
            self.model.eval()
            with torch.no_grad():
                _,ref=self.model(self.z,full)
                vp,_=self.model(vz,torch.ones((len(valid_time),4),device=self.device),ref)
                risk=self.logit_risk(vp).cpu().numpy()
            score=cindex(valid_time,valid_event,risk)
            if score>best+1e-5:
                best=score;state=copy.deepcopy(self.model.state_dict());stale=0;self.epoch=epoch+1
            else: stale+=1
            if stale>=config.get('patience',30):break
        self.model.load_state_dict(state);self.model.eval()
        with torch.no_grad(): _,self.reference=self.model(self.z,full)
        self.nparams=sum(p.numel() for p in self.model.parameters())

    def logit_risk(self,p):
        if self.kind=='multisurv':
            return -torch.cumprod(1-p.sigmoid(),1).sum(1)
        return p

    def predict(self,z,mask=None):
        z=[torch.as_tensor(x,device=self.device) for x in z]
        if mask is None:mask=np.ones((len(z[0]),4),dtype='float32')
        with torch.no_grad():
            p,_=self.model(z,torch.as_tensor(mask,dtype=torch.float32,device=self.device),self.reference)
        if self.kind=='multisurv':
            surv=torch.cumprod(1-p.sigmoid(),1).cpu().numpy()
            return self.logit_risk(p).cpu().numpy(),surv
        return p.cpu().numpy(),None
