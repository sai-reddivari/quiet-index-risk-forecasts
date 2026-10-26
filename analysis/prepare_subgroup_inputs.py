#!/usr/bin/env python3
"""Optional maintainer export from canonical snapshots. No fitting or acquisition.

The public subgroup_diagnostics.py needs only the compact shipped CSV inputs;
this export additionally needs the pinned source snapshots and their Git objects.
Paths are supplied via flags, never persisted in public provenance.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import subprocess
import pandas as pd
from subgroup_diagnostics import LABELS, KEY, RUNS, sha, save

ORIGINAL='14c5ba1333c63c6e12c116d468d60434b6aaedcf'
REVISED='4fa7d9f07a427931fd86854c780a589d8499aca0'
SUPPLEMENT='a7cd4db5c39d18b30a2efd707a265d3f23b54615'


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for name in ['original','revised','supplement','git-dir','output']:
        ap.add_argument('--'+name,type=Path,required=True)
    args=ap.parse_args();out=args.output;out.mkdir(parents=True,exist_ok=True)
    sources=[]
    def read(root,commit,rel,**kw):
        path=root/rel
        blob=subprocess.run(['git',f'--git-dir={args.git_dir}','show',f'{commit}:{rel}'],check=True,capture_output=True).stdout
        import hashlib
        canonical=hashlib.sha256(blob).hexdigest();assert sha(path)==canonical,(commit,rel)
        sources.append(dict(study={ORIGINAL:'original',REVISED:'revised',SUPPLEMENT:'supplement'}[commit],commit=commit,path=rel,
                            source_sha256=sha(path),canonical_git_blob_sha256=canonical,canonical_match=True))
        return pd.read_csv(path,**kw)
    reference_hashes={}
    for public,source in [('aggregate_comparisons.csv','all_summary.csv'),('group_comparisons.csv','group_summary.csv'),('group_uncertainty.csv','group_uncertainty.csv')]:
        rel='results/licensed/smh_subgroups/synthesis/'+source
        read(args.supplement,SUPPLEMENT,rel)
        reference_hashes[public]=sha(args.supplement/rel)
    panel_rel='results/licensed/panel/company_panel_504.csv'
    opanel=read(args.original,ORIGINAL,panel_rel,usecols=KEY+LABELS[2:]+['window_sessions','is_representative_security'])
    rpanel=read(args.revised,REVISED,panel_rel,usecols=KEY+LABELS[2:]+['window_sessions','is_representative_security'])
    for panel in [opanel,rpanel]:
        assert panel.is_representative_security.eq(True).all()
        assert panel.window_sessions.eq(504).all() and panel.groupby('formation_date').size().eq(500).all()
        assert not panel.duplicated(KEY).any()
        assert panel.groupby('cik')[LABELS[1:]].nunique().max().max()==1
    universe=opanel[LABELS].drop_duplicates().sort_values('cik')
    pd.testing.assert_frame_equal(universe.reset_index(drop=True),rpanel[LABELS].drop_duplicates().sort_values('cik').reset_index(drop=True))
    raw=read(args.original,ORIGINAL,'results/licensed/forward_tests/predictions_504_residual_volatility.csv')
    pairs=[]
    for comp in ['semiconductor','semiconductor_industry']:
        p=raw[raw.comparison.eq(comp)&raw.outcome.eq('log_forward_residual_volatility')].copy()
        assert len(p)==12511 and p.window_sessions.eq(504).all() and not p.duplicated(KEY).any()
        p=p.merge(opanel[KEY+LABELS[2:]],on=KEY,how='left',validate='one_to_one',indicator=True)
        assert p._merge.eq('both').all()
        p=p.rename(columns={'outcome_block':'block','restricted_prediction':'pred_b','extended_prediction':'pred_e'})
        p['quarter']=(pd.to_datetime(p.formation_date).dt.to_period('Q')+1).astype(str)
        p['run_id']='original_504_'+comp;p['sample_id']=''
        pairs.append(p)
    raw=read(args.revised,REVISED,'extensions/company_risk/results/semiconductor_company_predictions.csv.gz')
    raw=raw[raw.run_id.isin(['semiconductor_main','semiconductor_industry_group'])&raw.model_id.isin(['B','B+S'])].copy()
    keys=['run_id','sample_id','formation_date','outcome_quarter','issuer_id']
    assert not raw.duplicated(keys+['model_id']).any()
    b=raw[raw.model_id.eq('B')].drop(columns='model_id');e=raw[raw.model_id.eq('B+S')].drop(columns='model_id')
    p=b.merge(e,on=keys,how='outer',validate='one_to_one',suffixes=('_b','_e'),indicator=True)
    assert p._merge.eq('both').all();p=p.drop(columns='_merge')
    for col in ['actual_log_vol','actual_variance','outcome_end','block_label','exposure_window','snapshot_weight','score_weight']:
        assert p[col+'_b'].equals(p[col+'_e']),col
    assert p.exposure_window_b.eq(504).all() and p.score_weight_b.eq(1).all()
    p=p.rename(columns={'issuer_id':'cik','outcome_quarter':'quarter','block_label_b':'block',
                        'actual_log_vol_b':'actual','predicted_log_vol_b':'pred_b','predicted_log_vol_e':'pred_e'})
    p=p.merge(rpanel[KEY+LABELS[2:]],on=['cik','formation_date'],how='left',validate='many_to_one',indicator=True)
    assert p._merge.eq('both').all();p=p.drop(columns='_merge')
    feat=read(args.revised,REVISED,'extensions/company_risk/data/processed/formation_measurements.csv',
              usecols=['issuer_id','formation_date','exposure_window','representative_flag','sector','industry_group'])
    feat=feat[feat.exposure_window.eq(504)&feat.representative_flag.eq(True)].rename(columns={'issuer_id':'cik'})
    assert not feat.duplicated(['cik','formation_date']).any()
    p=p.merge(feat,on=['cik','formation_date'],how='left',validate='many_to_one',indicator=True)
    assert p._merge.eq('both').all() and p.gics_sector.eq(p.sector).all() and p.gics_industry_group.eq(p.industry_group).all()
    p['run_id']='revised_'+p.run_id;pairs.append(p)
    cols=['run_id','sample_id']+KEY+['quarter','block']+LABELS[2:]+['actual','pred_b','pred_e']
    paired=pd.concat([p[cols] for p in pairs],ignore_index=True).sort_values(['run_id','quarter','cik'])
    assert set(paired.run_id)==set(RUNS) and paired.groupby('run_id').size().eq(12511).all()
    save(paired,out/'paired_forecasts.csv.gz');save(universe,out/'universe.csv')
    manifest=dict(original_commit=ORIGINAL,revised_commit=REVISED,supplement_commit=SUPPLEMENT,
        method='Re-export canonical saved predictions; paired B/B+S, no model fitting, raw market data, or acquisition.',
        source_files=sources,public_reference_sha256=reference_hashes,selection=dict(window_sessions=504,original_comparisons=['semiconductor','semiconductor_industry'],
            original_outcome='log_forward_residual_volatility',revised_runs=['semiconductor_main','semiconductor_industry_group'],
            revised_models=['B','B+S'],blocks=dict(development=12,validation=8,confirmation=6)),
        joins=dict(original='one-to-one formation_date / cik / ticker to same-window panel',
                   revised_pair='one-to-one run_id / sample_id / formation_date / outcome_quarter / issuer_id',
                   revised_labels='many-to-one cik / formation_date to same-window representative panel; cross-check formation sector/industry labels',
                   identity_checks=['actual_log_vol','actual_variance','outcome_end','block_label','exposure_window','snapshot_weight','score_weight']),
        labels='Retrospective supplied GICS sectors and industry groups, fixed issuer names/tickers; not point-in-time reclassification.',
        universe_companies=len(universe),paired_rows=len(paired),rows_per_run=paired.groupby('run_id').size().to_dict(),
        files={n:dict(sha256=sha(out/n)) for n in ['paired_forecasts.csv.gz','universe.csv']},
        preparation_script_sha256=sha(Path(__file__)),scoring_script_sha256=sha(Path(__file__).with_name('subgroup_diagnostics.py')))
    (out/'input_provenance.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Exported {len(paired):,} paired rows and {len(universe)} fixed issuers; all consumed bytes match canonical Git blobs.')

if __name__=='__main__':main()
