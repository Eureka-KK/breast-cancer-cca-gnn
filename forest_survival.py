"""Exact terminal-node KM curves for a low-memory scikit-survival forest.

Reconstructs only requested horizons from each tree's bootstrap-weighted leaves,
instead of retaining full node x time arrays. Tested against native forest
survival-function predictions. Training leaf membership never uses query data.
"""
import numpy as np


class ForestSurvival:
    def __init__(self,forest,x,time,event,horizons):
        self.forest=forest;self.horizons=np.asarray(horizons);self.tables=[]
        trainleaves=forest.apply(x)
        for j,indices in enumerate(forest.estimators_samples_):
            weight=np.bincount(indices,minlength=len(time))
            table={}
            for leaf in np.unique(trainleaves[:,j]):
                belongs=(trainleaves[:,j]==leaf)&(weight>0)
                t=time[belongs];e=event[belongs];w=weight[belongs]
                event_times=np.unique(t[e])
                if not len(event_times):table[leaf]=np.ones(len(horizons));continue
                q=np.array([1-w[(t==u)&e].sum()/w[t>=u].sum() for u in event_times])
                survival=np.cumprod(q)
                at=np.searchsorted(event_times,self.horizons,side='right')-1
                table[leaf]=np.where(at>=0,survival[np.maximum(at,0)],1.)
            self.tables.append(table)

    def predict(self,x):
        leaves=self.forest.apply(x)
        result=np.zeros((len(x),len(self.horizons)))
        for j,table in enumerate(self.tables):result+=np.stack([table[l] for l in leaves[:,j]])
        return result/len(self.tables)
