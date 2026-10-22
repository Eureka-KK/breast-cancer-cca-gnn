"""Training-only canonical loadings; associations, not validated biomarkers."""
import json
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from fusion import Preprocessor,Alignment
from experiment import ROOT


def main():
    raw=[np.load(ROOT/f'.private/view{i}.npy',mmap_mode='r') for i in range(3)]
    names=[json.loads((ROOT/f'.private/features{i}.json').read_text()) for i in range(3)]
    clinical=pd.read_csv(ROOT/'.private/clinical.csv');rows=[]
    with threadpool_limits(2):
        for fold in range(5):
            config=json.loads((ROOT/f'.private/selected_cv{fold}_proposed.json').read_text())
            train=np.load(ROOT/f'.private/split_cv{fold}.npz')['train']
            pre=Preprocessor(config['features']).fit(raw,clinical,train)
            z=pre.transform(raw,clinical,train)
            a=Alignment(config['rank'],config['ridge'],config['anchor']).fit(z)
            for v in range(3):
                weights=a.weights[v]/a.scale[v]
                centered=z[v]-a.mean[v]
                scores=centered@weights
                correlations=(centered.T@scores/len(train))/np.maximum(centered.std(0)[:,None],1e-8)
                features=pre.transforms[v][0]
                for c in range(weights.shape[1]):
                    top=np.argsort(np.abs(weights[:,c]))[::-1][:10]
                    for rank,i in enumerate(top,1):
                        rows.append(dict(fold=fold,view=['mRNA','Methylation','CNA'][v],
                            component=c+1,loading_rank=rank,feature=names[v][features[i]],
                            standardized_projection_coefficient=float(weights[i,c]),
                            structure_correlation=float(correlations[i,c]),
                            interpretation='Within-fold association; sign/rotation not comparable across folds'))
    pd.DataFrame(rows).to_csv(ROOT/'results/canonical_loadings.csv',index=False)
    print('Canonical loading rows:',len(rows))


if __name__=='__main__':main()
