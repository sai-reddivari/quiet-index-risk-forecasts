#!/usr/bin/env python3
"""Portable rescoring of shipped, paired 504-session forecasts; never fits models.

python analysis/subgroup_diagnostics.py --output /tmp/subgroup-check
Requires numpy and pandas. The package root defaults to this script's parent.
The output directory must be new or empty. Existing public summaries are read
only, restricted to the four supported runs, and checked numerically.
"""
from __future__ import annotations
import argparse
import gzip
import hashlib
import itertools
import json
from pathlib import Path
import platform
import numpy as np
import pandas as pd

RUNS = ('original_504_semiconductor', 'original_504_semiconductor_industry',
        'revised_semiconductor_main', 'revised_semiconductor_industry_group')
BLOCKS = {'development':12, 'validation':8, 'confirmation':6, 'all':26}
KEY = ['formation_date', 'cik', 'ticker']
LABELS = ['cik', 'ticker', 'issuer_name', 'gics_sector', 'gics_industry_group']
FAMILIES = {'sector':'gics_sector', 'industry_group':'gics_industry_group'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(df, path):
    if path.suffix == '.gz':
        with path.open('wb') as f, gzip.GzipFile(filename='', mode='wb', fileobj=f, mtime=0) as z:
            z.write(df.to_csv(index=False, float_format='%.17g').encode())
    else:
        df.to_csv(path, index=False, float_format='%.17g')


def uncertainty(v):
    """Conditional six-quarter diagnostics, with all 64 and 36 outcomes enumerated."""
    v=np.asarray(v, float)
    assert len(v)==6 and np.isfinite(v).all()
    flips=np.asarray([np.mean(v*s) for s in itertools.product((-1,1), repeat=6)])
    draws=np.asarray([v[[(a+j)%6 for j in range(4)]+[b,(b+1)%6]].mean()
                      for a,b in itertools.product(range(6), repeat=2)])
    adjacent=v.reshape(3,2).mean(axis=1)
    paired=np.asarray([np.mean(adjacent*s) for s in itertools.product((-1,1), repeat=3)])
    lo,hi=np.quantile(draws, [.025,.975], method='linear')
    return dict(metric='quarterly_gain_pct',signflip_two_sided_p=np.mean(np.abs(flips)>=abs(v.mean())-1e-14),
                paired_twoquarter_signflip_p=np.mean(np.abs(paired)>=abs(adjacent.mean())-1e-14),
                circular4_low=lo,circular4_high=hi,circular4_draws=36)


def holm(p):
    p=np.asarray(p); order=np.argsort(p)
    out=np.empty(len(p));out[order]=np.minimum(1,np.maximum.accumulate((len(p)-np.arange(len(p)))*p[order]))
    return out


def quarters(p, column=None):
    keys=['run_id','block','quarter','formation_date']+([column] if column else [])
    rows=[]
    for k,g in p.groupby(keys, sort=True):
        b,e=g.loss_b.mean(),g.loss_e.mean()
        cal=(g.pred_b-g.actual).mean()**2-(g.pred_e-g.actual).mean()**2
        row=dict(zip(keys,k))
        row.update(n=len(g),mse_b=b,mse_e=e,d=b-e,gain_pct=100*(b-e)/b,
                   calibration_pp=100*cal/b,centered_pp=100*(b-e-cal)/b,
                   contribution_pp=g.row_contribution_pp.sum(),contribution_d=g.row_contribution_d.sum(),
                   favorable_companies=int((g.d>0).sum()))
        rows.append(row)
    return pd.DataFrame(rows).rename(columns={column:'group'} if column else {})


def summaries(q, grouped=False):
    keys=['run_id']+(['group'] if grouped else [])
    rows=[]
    for k,g in q.groupby(keys, sort=True):
        prefix=dict(zip(keys,k if isinstance(k,tuple) else (k,)))
        for block,nq in BLOCKS.items():
            z=g if block=='all' else g[g.block==block]
            assert len(z)==nq
            b,e=z.mse_b.mean(),z.mse_e.mean();v=z.gain_pct.to_numpy()
            rows.append(prefix|dict(block=block,quarters=len(z),expected_quarters=nq,
                observations=int(z.n.sum()),n_min=int(z.n.min()),n_max=int(z.n.max()),
                mse_b=b,mse_e=e,mean_d=z.d.mean(),mean_quarterly_gain_pct=v.mean(),
                ratio_mean_loss_gain_pct=100*(b-e)/b,min_quarter_mse_b=z.mse_b.min(),
                favorable_quarters=int((v>0).sum()),worst_quarter_gain_pct=v.min(),best_quarter_gain_pct=v.max(),
                loo_gain_pct_min=min((v.sum()-x)/(len(v)-1) for x in v),
                loo_gain_pct_max=max((v.sum()-x)/(len(v)-1) for x in v),
                calibration_pp=z.calibration_pp.mean(),centered_pp=z.centered_pp.mean(),
                headline_contribution_pp=z.contribution_pp.sum()/nq,
                population_contribution_d=z.contribution_d.sum()/nq))
    return pd.DataFrame(rows)


def compare(generated, reference, keys, checks):
    ref=pd.read_csv(reference)
    ref=ref[ref.run_id.isin(RUNS)]
    z=generated.merge(ref,on=keys,how='outer',suffixes=('_new','_saved'),indicator=True,validate='one_to_one')
    assert z._merge.eq('both').all(), (reference.name,'missing or extra keys')
    cols=[c for c in generated if c not in keys and c in ref and pd.api.types.is_numeric_dtype(generated[c])]
    errors={}
    for c in cols:
        a,b=z[c+'_new'].to_numpy(float),z[c+'_saved'].to_numpy(float)
        np.testing.assert_allclose(a,b,rtol=1e-10,atol=1e-10,equal_nan=True,err_msg=f'{reference.name}:{c}')
        errors[c]=float(np.nanmax(np.abs(a-b)))
    checks.append(dict(reference=reference.name,sha256=sha(reference),rows=len(z),columns=cols,max_absolute_errors=errors))


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--package-root',type=Path,default=Path(__file__).resolve().parents[1])
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args(); root=args.package_root.resolve(); out=args.output.resolve()
    if out.exists() and any(out.iterdir()):
        raise ValueError('--output must be a fresh or empty directory')
    out.mkdir(parents=True,exist_ok=True)
    inp=root/'evidence/subgroups';checks=[]
    provenance=json.loads((inp/'input_provenance.json').read_text())
    for name,entry in provenance['files'].items():
        assert sha(inp/name)==entry['sha256'], ('input hash',name)
    for name,expected in provenance['public_reference_sha256'].items():
        assert sha(root/'evidence'/name)==expected, ('reference hash',name)
    p=pd.read_csv(inp/'paired_forecasts.csv.gz');u=pd.read_csv(inp/'universe.csv')
    assert set(p.run_id)==set(RUNS) and len(u)==u.cik.nunique()==500
    assert not u.duplicated('ticker').any() and not p.duplicated(['run_id']+KEY).any()
    assert p.groupby('run_id').size().eq(12511).all()
    assert np.isfinite(p[['actual','pred_b','pred_e']]).all().all()
    labels=p[LABELS].drop_duplicates().merge(u,on=LABELS,how='left',indicator=True,validate='one_to_one')
    assert labels._merge.eq('both').all()
    assert p.groupby(['run_id','block']).quarter.nunique().to_dict()=={(r,b):n for r in RUNS for b,n in BLOCKS.items() if b!='all'}
    expected_quarter=(pd.to_datetime(p.formation_date).dt.to_period('Q')+1).astype(str)
    assert p.quarter.equals(expected_quarter)
    assert sorted(p.quarter.unique())==list(pd.period_range('2020Q1','2026Q2',freq='Q').astype(str))
    p['loss_b']=(p.actual-p.pred_b)**2;p['loss_e']=(p.actual-p.pred_e)**2;p['d']=p.loss_b-p.loss_e
    p['n_quarter']=p.groupby(['run_id','quarter']).cik.transform('size')
    p['population_mse_b']=p.groupby(['run_id','quarter']).loss_b.transform('mean')
    p['row_contribution_d']=p.d/p.n_quarter
    p['row_contribution_pp']=100*p.d/(p.n_quarter*p.population_mse_b)
    # Same saved targets and issuer keys in all four runs; separate forecasts.
    base=p[p.run_id==RUNS[0]].set_index(KEY).sort_index()
    target_errors={}
    for run in RUNS[1:]:
        z=p[p.run_id==run].set_index(KEY).sort_index()
        assert z.index.equals(base.index)
        err=float(np.abs(z.actual-base.actual).max());assert err<1e-9
        target_errors[run]=err
    aq=quarters(p); ag=summaries(aq);gqs=[];gss=[];unc=[]
    for family,col in FAMILIES.items():
        q=quarters(p,col);s=summaries(q,True)
        counts=u.groupby(col).size()
        q['universe_companies']=q['group'].map(counts)
        q['missing_observations']=q.universe_companies-q.n
        s['universe_companies']=s['group'].map(counts)
        s['possible_observations']=s.universe_companies*s.expected_quarters
        s['missing_observations']=s.possible_observations-s.observations
        q['family']=family;s['family']=family;gqs.append(q);gss.append(s)
        recon=q.groupby(['run_id','quarter'])[['n','contribution_pp','contribution_d']].sum().reset_index().merge(aq,on=['run_id','quarter'],validate='one_to_one',suffixes=('_groups','_all'))
        for c in ['n','contribution_pp','contribution_d']:
            np.testing.assert_allclose(recon[c+'_groups'],recon[c+'_all'],rtol=1e-12,atol=1e-12)
        for run,z in q[q.block=='confirmation'].groupby('run_id'):
            records=[dict(run_id=run,group=group,family=family)|uncertainty(g.sort_values('quarter').gain_pct)
                     for group,g in z.groupby('group')]
            expected=11 if family=='sector' else 25;assert len(records)==expected
            adj=holm([r['signflip_two_sided_p'] for r in records])
            for row,pval in zip(records,adj):row.update(holm_p=pval,family_size=expected)
            unc.extend(records)
    gq=pd.concat(gqs,ignore_index=True);gs=pd.concat(gss,ignore_index=True);gu=pd.DataFrame(unc)
    # Full 500 x 26 coverage for each run includes absent firms explicitly.
    dates=p[['run_id','formation_date','quarter','block']].drop_duplicates()
    grid=dates.merge(u,how='cross')
    cov=grid.merge(p[['run_id']+KEY+['actual','pred_b','pred_e','loss_b','loss_e','d','row_contribution_pp']],
                   on=['run_id']+KEY,how='left',validate='one_to_one',indicator=True)
    cov['covered']=cov._merge.eq('both');cov=cov.drop(columns='_merge')
    assert len(cov)==4*500*26
    companies=[]
    for (run,cik),g in cov.groupby(['run_id','cik'],sort=True):
        pref=g.iloc[0][LABELS].to_dict()|dict(run_id=run)
        for block,nq in BLOCKS.items():
            z=g if block=='all' else g[g.block==block];covered=z[z.covered];n=len(covered)
            b,e=covered.loss_b.mean(),covered.loss_e.mean()
            companies.append(pref|dict(block=block,observations=n,expected_quarters=nq,missing_quarters=nq-n,
                mse_b=b,mse_e=e,mean_d=covered.d.mean(),ratio_loss_gain_pct=100*(b-e)/b if n else np.nan,
                minimum_loss_b=covered.loss_b.min(),favorable_quarters=int((covered.d>0).sum()) if n else np.nan,
                headline_contribution_pp=covered.row_contribution_pp.sum()/nq))
    cs=pd.DataFrame(companies)
    cr=cs.groupby(['run_id','block']).headline_contribution_pp.sum().reset_index().merge(ag,on=['run_id','block'],suffixes=('_companies','_all'))
    np.testing.assert_allclose(cr.headline_contribution_pp_companies,cr.mean_quarterly_gain_pct,rtol=1e-12,atol=1e-12)
    coverage=cov.groupby(['run_id','quarter','block']).covered.agg(['size','sum']).reset_index().rename(columns={'size':'possible_observations','sum':'covered_observations'})
    coverage['missing_observations']=coverage.possible_observations-coverage.covered_observations
    compare(ag,root/'evidence/aggregate_comparisons.csv',['run_id','block'],checks)
    compare(gs,root/'evidence/group_comparisons.csv',['run_id','group','block','family'],checks)
    compare(gu,root/'evidence/group_uncertainty.csv',['run_id','group','family'],checks)
    tables={'all_quarters.csv':aq,'all_summary.csv':ag,'group_quarters.csv':gq,'group_summary.csv':gs,
            'group_uncertainty.csv':gu,'company_summary.csv.gz':cs,'coverage_quarters.csv':coverage,
            'coverage.csv.gz':cov[['run_id']+KEY+['quarter','block','covered']],
            'late_company_quarters.csv.gz':cov[cov.block=='confirmation']}
    for name,table in tables.items():save(table,out/name)
    report=dict(status='passed',method='saved paired forecasts only; no fitting',runs=list(RUNS),
        input_sha256={n:sha(inp/n) for n in ['paired_forecasts.csv.gz','universe.csv']},
        versions=dict(python=platform.python_version(),numpy=np.__version__,pandas=pd.__version__),
        paired_rows=len(p),universe_companies=len(u),coverage_rows=len(cov),
        cross_run_max_target_difference=target_errors,
        group_contributions_reconcile=True,company_contributions_reconcile=True,
        reference_checks=checks,outputs={n:dict(rows=len(t),sha256=sha(out/n)) for n,t in tables.items()})
    (out/'verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(f'PASS: {len(p):,} pairs, four runs, 26 quarters; 11 sectors / 25 industry groups; fixed universe 500.')
    print(f'Compared all supported aggregate/group summaries and conditional uncertainty; outputs: {out}')

if __name__=='__main__':main()
