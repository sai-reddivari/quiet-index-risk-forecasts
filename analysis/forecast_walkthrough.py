#!/usr/bin/env python3
"""Reconstruct one saved revised forecast from saved slopes and training inputs.

No regression is refitted. The source export omits intercept/dummy coefficients;
their combined sector offsets follow from the OLS group normal equations with
the saved numeric slopes held fixed. All evaluation predictions are checked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

FORMATION = '2024-12-31'
ISSUER = 19617  # JPM; continuity with the earlier worked example, not a winning row.
RUN = 'semiconductor_main'
KEY = ['issuer_id', 'formation_date', 'exposure_window']
BASE = ['conventional_' + f for f in ['mkt_rf','smb','hml','rmw','cma','mom']]
BASE += ['log_issuer_cap','own_momentum','long_residual_vol_annual',
         'french_market_beta','market_negative_increment','R60',
         'L_amihud252','L_dollar_volume60']


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--package-root', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--inputs', type=Path, help='Defaults to evidence/walkthrough in package')
    parser.add_argument('--output', type=Path, required=True, help='New directory; existing paths refused')
    args = parser.parse_args()
    root = args.package_root.resolve()
    inputs = args.inputs or root/'evidence/walkthrough'
    provenance = json.loads((inputs/'provenance.json').read_text())
    for item in provenance['packaged_inputs']:
        path = inputs/item['path'] if item['location']=='walkthrough' else root/item['path']
        if sha(path) != item['sha256']:
            raise ValueError(f'Input fingerprint mismatch: {item["path"]}')
    ext = root/'reproduction/company-risk/extensions/company_risk'
    features = pd.read_csv(ext/'data/processed/formation_measurements.csv', parse_dates=['formation_date'])
    outcomes = pd.read_csv(ext/'data/processed/company_outcomes.csv', parse_dates=['formation_date','outcome_end'])
    specs = json.loads((ext/'specification/comparison_manifest.json').read_text())['runs']
    spec = next(s for s in specs if s['run_id']==RUN)
    assert spec['models']==['B','B+S','B+V','B+S+V']
    assert spec['exposure_window']==504 and spec['controls']=='sector'
    assert spec['liquidity']=='amihud252_dollar60'
    features['long_residual_vol_annual'] = np.exp(features.long_log_residual_vol)
    frame = features.loc[features.exposure_window.eq(504) & features.representative_flag].merge(
        outcomes, on=KEY, validate='one_to_one', suffixes=('','_outcome'))
    # Eligibility is the declared four-model union, even when displaying B vs B+S.
    union = sorted(set(BASE + ['S','V_annual']))
    eligible = frame[union+['log_residual_vol']].notna().all(axis=1)
    sample = frame.loc[eligible].copy()
    formation = pd.Timestamp(FORMATION)
    train = sample.loc[(sample.formation_date < formation) & (sample.outcome_end <= formation)]
    test = sample.loc[sample.formation_date.eq(formation)].copy().set_index('issuer_id')
    assert train.formation_date.nunique() >= 8 and train.outcome_end.max() <= formation
    assert not test.index.duplicated().any() and test.loc[ISSUER,'security_id']=='JPM'
    saved = pd.read_csv(inputs/'predictions.csv').set_index(['model_id','issuer_id'])
    saved_scores = pd.read_csv(inputs/'quarter_scores.csv').set_index('model_id')
    prep = pd.read_csv(inputs/'preprocessing.csv')
    assert not saved.index.duplicated().any()
    assert set(prep.run_id)=={RUN} and set(prep.formation_date)=={FORMATION}
    assert set(prep.model_id)=={'B','B+S'} and set(saved.index.get_level_values(0))=={'B','B+S'}
    assert not prep.duplicated(['model_id','term']).any()
    summary = {}; contributions = []; recovered = []; full = []
    for model in ['B','B+S']:
        terms = BASE + (['S'] if model=='B+S' else [])
        p = prep.loc[prep.model_id.eq(model)].set_index('term').loc[terms]
        assert len(p)==len(terms)
        assert p.training_count.eq(len(train)).all()
        assert pd.to_datetime(p.training_end).eq(train.outcome_end.max()).all()
        mean, scale = train[terms].mean(), train[terms].std(ddof=0).replace(0,1)
        np.testing.assert_allclose(mean, p.training_mean, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(scale, p.training_scale, atol=1e-12, rtol=1e-12)
        # Use the SAVED training transformations and numeric coefficients.
        z_train = (train[terms]-p.training_mean)/p.training_scale
        z_test = (test[terms]-p.training_mean)/p.training_scale
        beta = p.coefficient.to_numpy()
        numeric_train = z_train.to_numpy() @ beta
        numeric_test = z_test.to_numpy() @ beta
        # OLS with an intercept and sector dummies has zero residual sum per sector:
        # offset_g = mean(y - z @ saved_beta | training sector g).
        # This recovers the omitted intercept + sector coefficient without solving
        # a regression or using any future outcome to reconstruct the predictions.
        offsets = (train.log_residual_vol-numeric_train).groupby(train.sector).mean()
        categories = json.loads(p.categories.iloc[0])
        assert categories==sorted('sector_'+s for s in offsets.index)
        assert set(test.sector).issubset(offsets.index)
        pred = numeric_test + test.sector.map(offsets).to_numpy()
        s = saved.loc[model].reindex(test.index)
        assert set(saved.loc[model].index)==set(test.index)
        assert s.training_count.eq(len(train)).all()
        assert s.training_quarters.eq(train.formation_date.nunique()).all()
        np.testing.assert_allclose(s.actual_log_vol, test.log_residual_vol, atol=1e-12, rtol=0)
        np.testing.assert_allclose(pred, s.predicted_log_vol, atol=1e-9, rtol=0)
        for sector,value in offsets.items():
            recovered.append({'model':model,'sector':sector,'combined_intercept_and_sector_offset':value})
        x = test.loc[ISSUER]; z = z_test.loc[ISSUER]; numeric = z * p.coefficient
        for term in terms:
            contributions.append({'model':model,'term':term,'raw_feature':x[term],
                'training_mean':p.loc[term,'training_mean'],'training_scale':p.loc[term,'training_scale'],
                'standardized_feature':z[term],'forecast_coefficient':p.loc[term,'coefficient'],
                'log_forecast_contribution':numeric[term]})
        actual = float(x.log_residual_vol); yhat = float(pred[test.index.get_loc(ISSUER)])
        residual = train.log_residual_vol-(numeric_train+train.sector.map(offsets).to_numpy())
        smear = float(np.exp(2*residual).mean())
        np.testing.assert_allclose(smear,p.smearing,atol=1e-10,rtol=0)
        mse = float(np.mean((test.log_residual_vol.to_numpy()-pred)**2))
        assert saved_scores.loc[model,'n_companies']==len(test)
        np.testing.assert_allclose(mse,saved_scores.loc[model,'mse_log_vol'],atol=1e-10,rtol=0)
        summary[model]={'actual_log_vol':actual,'predicted_log_vol':yhat,
            'squared_error':(actual-yhat)**2,'exp_predicted_log_vol':float(np.exp(yhat)),
            'actual_annualized_vol':float(x.residual_vol_annual),
            'numeric_contribution_sum':float(numeric.sum()),
            'combined_intercept_and_sector_offset':float(offsets[x.sector]),
            'training_smearing_factor':smear,'quarter_mse':mse,
            'max_saved_prediction_difference':float(np.max(np.abs(pred-s.predicted_log_vol)))}
        full.append(pd.DataFrame({'model':model,'issuer_id':test.index,'actual_log_vol':test.log_residual_vol,
                                  'reconstructed_log_forecast':pred,'saved_log_forecast':s.predicted_log_vol}))
    srow = prep.loc[prep.model_id.eq('B+S') & prep.term.eq('S')].iloc[0]
    x = test.loc[ISSUER]
    result={'study':'revised','run_id':RUN,'issuer_id':ISSUER,'ticker':'JPM',
        'selection':'Same issuer as earlier worked example; first late quarter. Not selected for forecast benefit.',
        'formation_date':FORMATION,'outcome_quarter':x.outcome_quarter,
        'outcome_start':x.outcome_start,'outcome_end':str(x.outcome_end.date()),
        'outcome_days':int(x.n_outcome),'estimation_start':x.estimation_start,'estimation_end':x.estimation_end,
        'conventional_intercept':float(x.conventional_intercept),
        'conventional_loadings':{f:float(x['conventional_'+f]) for f in ['mkt_rf','smb','hml','rmw','cma','mom']},
        'training_rows':len(train),'training_quarters':int(train.formation_date.nunique()),
        'training_last_completed_outcome':str(train.outcome_end.max().date()),'evaluation_companies':len(test),
        'signed_smh_loading':float(x.S),'standardized_smh_loading':float((x.S-srow.training_mean)/srow.training_scale),
        'smh_forecast_coefficient':float(srow.coefficient),
        'smh_term_only':float((x.S-srow.training_mean)/srow.training_scale*srow.coefficient),
        'forecast_difference':summary['B+S']['predicted_log_vol']-summary['B']['predicted_log_vol'],
        'paired_squared_error_improvement':summary['B']['squared_error']-summary['B+S']['squared_error'],
        'quarter_mse_improvement_pct':100*(summary['B']['quarter_mse']-summary['B+S']['quarter_mse'])/summary['B']['quarter_mse'],
        'models':summary,'regression_refits':0,'new_market_data':False,
        'boundary':'Saved prepared measurements and numeric coefficients; sector offsets reconstructed from training-only OLS normal equations. No raw-price target rebuild.'}
    args.output.mkdir(parents=True, exist_ok=False)
    pd.DataFrame(contributions).to_csv(args.output/'feature_contributions.csv',index=False)
    pd.DataFrame(recovered).to_csv(args.output/'sector_offsets.csv',index=False)
    pd.concat(full,ignore_index=True).to_csv(args.output/'quarter_prediction_check.csv',index=False)
    (args.output/'forecast_trace.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(result,indent=2,allow_nan=False))


if __name__=='__main__':
    main()
