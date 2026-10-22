"""Separate smoke-test artifacts from the completed scientific release."""
import shutil
from pathlib import Path
import pandas as pd
from experiment import ROOT


def main():
    result=ROOT/'results'
    d=pd.read_csv(result/'fold_results.csv')
    scientific=d[d.split.str.match(r'^(cv[0-4]|cohort[1-5])$')]
    assert len(scientific)==405, 'Finish all prescribed runs before curating'
    assert not scientific.duplicated(['split','method','seed']).any()
    archive=ROOT/'.private/raw_run_logs';archive.mkdir(exist_ok=True)
    for name in ['fold_results.csv','selected_parameters.csv','search.csv',
                 'missing_subsets.csv','random_missing.csv']:
        source=result/name
        if not (archive/name).exists():shutil.copy2(source,archive/name)
        frame=pd.read_csv(source,dtype={'subset':str})
        frame=frame[frame.split.str.match(r'^(cv[0-4]|cohort[1-5])$')]
        frame.to_csv(source,index=False)
    print('Scientific release: 405 selected-model runs; smoke artifacts archived privately')


if __name__=='__main__':main()
