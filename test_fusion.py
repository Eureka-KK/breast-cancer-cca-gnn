import numpy as np
import torch
from fusion import cox_loss, Alignment, Network, cindex
from experiment import breslow_survival


def test_breslow_tied_events_matches_explicit_risk_sets():
    t=torch.tensor([2.,2.,3.,5.,5.]);e=torch.tensor([1.,1.,0.,1.,0.])
    r=torch.tensor([.2,-.7,.3,.8,-.1],requires_grad=True)
    expected=sum(-r[i]+torch.logsumexp(r[t>=t[i]],0) for i in range(5) if e[i])/e.sum()
    assert torch.allclose(cox_loss(r,t,e),expected)
    cox_loss(r,t,e).backward();assert torch.isfinite(r.grad).all()


def test_inductive_predictions_do_not_depend_on_query_batch():
    torch.manual_seed(42)
    net=Network([4]*4,hidden=8,k=3).eval()
    train=[torch.randn(20,4) for _ in range(4)]
    query=[torch.randn(5,4) for _ in range(4)]
    with torch.no_grad():
        _,ref=net(train,torch.ones(20,4))
        p,_=net(query,torch.ones(5,4),ref)
        q,_=net([a[:1] for a in query],torch.ones(1,4),ref)
    assert torch.allclose(p[:1],q,atol=1e-6)


def test_all_nonempty_missing_patterns_are_finite():
    import itertools
    net=Network([4]*4,hidden=8,k=3).eval()
    z=[torch.randn(12,4) for _ in range(4)]
    with torch.no_grad():
        _,ref=net(z,torch.ones(12,4))
        for bits in itertools.product([0,1],repeat=4):
            if any(bits):
                p,_=net(z,torch.tensor(bits).float().repeat(12,1),ref)
                assert torch.isfinite(p).all()


def test_alignment_is_training_only_and_shared_dimension():
    rng=np.random.default_rng(7);z=[rng.normal(size=(30,p)) for p in [8,9,10,7]]
    a=Alignment(4).fit(z)
    before=[w.copy() for w in a.weights]
    q=a.transform([x[:3]*100 for x in z])
    assert all(x.shape==(3,4) for x in q)
    assert all(np.array_equal(x,y) for x,y in zip(before,a.weights))


def test_breslow_survival_monotone_and_risk_direction():
    s=breslow_survival(np.array([1.,2.,3.,4.]),np.array([1,0,1,1],bool),
                       np.array([.1,.2,.3,.4]),np.array([0.,1.]),[1.,2.,3.])
    assert np.all(np.diff(s,axis=1)<=0)
    assert np.all(s[0]>=s[1])
    assert cindex(np.array([1.,2.,3.]),np.ones(3,bool),np.array([3.,2.,1.]))==1


def test_bootstrap_comparability_matches_harrell_with_ties():
    from summarize import comparable
    t=np.array([1.,1.,2.,3.,3.,4.]);e=np.array([1,0,1,1,1,0],bool)
    r=np.array([.2,.2,.1,.3,.0,.1]);comp=comparable(t,e)
    diff=r[:,None]-r[None,:]
    explicit=np.sum(((diff>1e-8)+.5*(np.abs(diff)<=1e-8))*comp)/comp.sum()
    assert np.allclose(explicit,cindex(t,e,r))


def test_preprocessor_does_not_fit_test_values_or_vocabulary():
    from fusion import Preprocessor
    from prepare import FIELDS
    import pandas as pd
    rng=np.random.default_rng(8)
    views=[rng.normal(size=(30,40)).astype('float32') for _ in range(3)]
    clinical=pd.DataFrame({field:[str(j%3) for j in range(30)] for field in FIELDS})
    train=np.arange(20);test=np.arange(20,30)
    pre=Preprocessor(10).fit(views,clinical,train)
    before=[tuple(a.copy() for a in item) for item in pre.transforms]
    vocab=dict(pre.vectorizer.vocabulary_)
    clinical.loc[test,'HISTOLOGICAL_SUBTYPE']='UNSEEN_TEST_TOKEN'
    modified=[a.copy() for a in views]
    for a in modified:a[test]=1e6
    result=pre.transform(modified,clinical,test)
    assert all(np.isfinite(a).all() for a in result)
    assert vocab==pre.vectorizer.vocabulary_
    for old,new in zip(before,pre.transforms):
        assert all(np.array_equal(a,b) for a,b in zip(old,new))


def test_low_memory_forest_curves_equal_native_survival():
    from forest_survival import ForestSurvival
    from sksurv.ensemble import RandomSurvivalForest
    from sksurv.util import Surv
    rng=np.random.default_rng(42)
    x=rng.normal(size=(80,5)).astype('float32')
    t=rng.integers(1,40,80).astype(float);e=rng.random(80)>.3
    horizon=np.array([3.,12.,24.])
    forest=RandomSurvivalForest(n_estimators=8,min_samples_leaf=3,random_state=42,n_jobs=1).fit(x,Surv.from_arrays(e,t))
    manual=ForestSurvival(forest,x,t,e,horizon).predict(x[:12])
    native=np.stack([f(horizon) for f in forest.predict_survival_function(x[:12])])
    assert np.allclose(manual,native,atol=1e-10)
