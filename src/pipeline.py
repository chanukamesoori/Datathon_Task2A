"""Reproduce audit, chronological selection, untouched holdout, and submission."""
from pathlib import Path
import json
import time
import joblib
import numpy as np
import pandas as pd
from src.data_preparation import ROOT, prepare
from src.forecasting import SPECS, fit, predict, make_series
from src.evaluation import summarize

def calendar_for(weekly, calendar):
    return calendar[calendar.week_start >= weekly.week_start.min()].sort_values('week_start').reset_index(drop=True)

def attach_actual(pred, series):
    labels=pd.concat([f[['week_start','depot','brand','total','chilled']] for f in series.values()])
    long=labels.melt(id_vars=['week_start','depot','brand'],value_vars=['total','chilled'],var_name='target',value_name='actual')
    return pred.merge(long,on=['week_start','depot','brand','target'],validate='one_to_one')

def predict_pair(bundles, series, calendar, origin):
    total=predict(bundles['total'],series,calendar,origin)
    lookup={(r.depot,r.brand,r.horizon):r.predicted for r in total.itertuples()}
    chilled=predict(bundles['chilled'],series,calendar,origin,total_predictions=lookup)
    chilled['predicted']=[min(r.predicted,lookup[(r.depot,r.brand,r.horizon)]) for r in chilled.itertuples()]
    return pd.concat([total,chilled],ignore_index=True)

def infer_artifact(artifact, inputs):
    """Generate forecast rows using saved history and only known future calendar."""
    predictions={}
    for target, selection in artifact['selection'].items():
        frames=[]
        for name in selection:
            frames.append(predict(artifact['bundles'][target][name],artifact['series'],artifact['calendar'],artifact['origin'],total_predictions=predictions.get('lookup')))
        p=frames[0].copy()
        p['predicted']=np.mean([f.predicted.to_numpy() for f in frames],axis=0)
        predictions[target]=p
        if target=='total': predictions['lookup']={(r.depot,r.brand,r.horizon):r.predicted for r in p.itertuples()}
    total=predictions['total'].rename(columns={'predicted':'pred_total_volume_m3'})
    chilled=predictions['chilled'].rename(columns={'predicted':'pred_chilled_volume_m3'})
    merged=total.merge(chilled[['week_start','depot','brand','pred_chilled_volume_m3']],on=['week_start','depot','brand'],how='left')
    merged['pred_chilled_volume_m3']=merged.pred_chilled_volume_m3.fillna(0).clip(lower=0,upper=merged.pred_total_volume_m3)
    merged=merged.merge(artifact['calendar'][['week_start','iso_year','iso_week']],on='week_start',validate='many_to_one')
    keys=['depot','brand','iso_year','iso_week']
    result=inputs.merge(merged[keys+['pred_total_volume_m3','pred_chilled_volume_m3']],on=keys,how='left',validate='one_to_one',sort=False)
    assert result.row_id.equals(inputs.row_id)
    assert np.isfinite(result[['pred_total_volume_m3','pred_chilled_volume_m3']]).all().all()
    return result

def run():
    raw,orders,weekly,cal,audit=prepare()
    series=make_series(weekly)
    calendar=calendar_for(weekly,cal)
    n=weekly.week_start.nunique()
    holdout_origin=n-11
    origins=[holdout_origin-40,holdout_origin-30,holdout_origin-20,holdout_origin-10]
    folds=[]
    for origin in origins:
        print('Validation origin',calendar.iloc[origin].week_start,flush=True)
        for name in SPECS:
            start=time.monotonic()
            bundles={t:fit(name,series,calendar,origin,t) for t in ['total','chilled']}
            pred=attach_actual(predict_pair(bundles,series,calendar,origin),series)
            pred['model']=name
            pred['origin']=calendar.iloc[origin].week_start
            folds.append(pred)
            print(name,round(time.monotonic()-start,1),'seconds',flush=True)
    validation=pd.concat(folds,ignore_index=True)
    comparison=summarize(validation,['model','target'])
    # A single predeclared equal-weight ensemble of the two strongest individual models per target.
    selection={}
    for target in ['total','chilled']:
        ranked=comparison[comparison.target==target].sort_values('WAPE')
        best=ranked.iloc[0].model
        names=ranked.head(2).model.tolist()
        left=validation[(validation.target==target)&(validation.model==names[0])].copy()
        right=validation[(validation.target==target)&(validation.model==names[1])]
        assert left[['origin','week_start','depot','brand']].reset_index(drop=True).equals(right[['origin','week_start','depot','brand']].reset_index(drop=True))
        left['predicted']=(left.predicted.to_numpy()+right.predicted.to_numpy())/2
        left['model']='ensemble_'+target
        ensemble_metrics=summarize(left,['model','target'])
        comparison=pd.concat([comparison,ensemble_metrics],ignore_index=True)
        validation=pd.concat([validation,left],ignore_index=True)
        selection[target]=names if ensemble_metrics.iloc[0].WAPE < ranked.iloc[0].WAPE else [best]
    (ROOT/'models').mkdir(exist_ok=True)
    validation.to_csv(ROOT/'outputs/backtest_predictions.csv',index=False)
    comparison.sort_values(['target','WAPE']).to_csv(ROOT/'outputs/model_comparison.csv',index=False)
    detail=[]
    for groups in [['model','target','origin'],['model','target','depot'],['model','target','brand'],['model','target','horizon']]:
        detail.append(summarize(validation,groups))
    pd.concat(detail,ignore_index=True).to_csv(ROOT/'outputs/backtest_metrics.csv',index=False)
    # Freeze selection before obtaining any holdout performance.
    (ROOT/'outputs/selection.json').write_text(json.dumps(selection,indent=2))
    print('Stage D selected:',selection,flush=True)
    def artifact_at(origin):
        # Persist only history available at this origin. Slicing also makes accidental leakage impossible.
        history={k:f.iloc[:origin+1].copy() for k,f in series.items()}
        bundles={t:{name:fit(name,history,calendar,origin,t) for name in names} for t,names in selection.items()}
        return {'selection':selection,'bundles':bundles,'series':history,'calendar':calendar,'origin':origin,'version':1}
    holdout_artifact=artifact_at(holdout_origin)
    holdout_inputs=weekly[weekly.week_start>calendar.iloc[holdout_origin].week_start][['depot','brand','iso_year','iso_week']].copy().reset_index(drop=True)
    holdout_inputs['row_id']=['H'+str(i) for i in range(len(holdout_inputs))]
    hp=infer_artifact(holdout_artifact,holdout_inputs)
    hp=hp.merge(weekly[['depot','brand','iso_year','iso_week','total','chilled']],on=['depot','brand','iso_year','iso_week'],validate='one_to_one')
    hp.to_csv(ROOT/'outputs/holdout_predictions.csv',index=False)
    hm=[]
    from src.evaluation import metrics
    for t in ['total','chilled']:
        f=hp if t=='total' else hp[hp.brand=='Fresh']
        hm.append({'target':t,**metrics(f[t],f[f'pred_{t}_volume_m3'])})
    holdout_metrics=pd.DataFrame(hm)
    holdout_metrics.to_csv(ROOT/'outputs/holdout_metrics.csv',index=False)
    final=artifact_at(n-1)
    assert calendar.iloc[n:n+10].days.eq(7).all()
    assert raw['task2a_test_inputs'][['iso_year','iso_week']].drop_duplicates().reset_index(drop=True).equals(calendar.iloc[n:n+10][['iso_year','iso_week']].reset_index(drop=True))
    forecast=infer_artifact(final,raw['task2a_test_inputs'])
    joblib.dump(final,ROOT/'models/forecast_artifact.joblib')
    reloaded=joblib.load(ROOT/'models/forecast_artifact.joblib')
    repeated=infer_artifact(reloaded,raw['task2a_test_inputs'])
    np.testing.assert_allclose(forecast[['pred_total_volume_m3','pred_chilled_volume_m3']],repeated[['pred_total_volume_m3','pred_chilled_volume_m3']],rtol=0,atol=0)
    template=raw['submission_task2a']
    submission=template[['row_id']].merge(forecast[['row_id','pred_total_volume_m3','pred_chilled_volume_m3']],on='row_id',how='left',validate='one_to_one',sort=False)
    assert list(submission)==list(template) and submission.row_id.equals(template.row_id)
    assert len(submission)==60 and submission.notna().all().all()
    assert (submission.iloc[:,1:]>=0).all().all()
    assert (submission.pred_chilled_volume_m3<=submission.pred_total_volume_m3).all()
    assert forecast.loc[forecast.brand!='Fresh','pred_chilled_volume_m3'].eq(0).all()
    submission.to_csv(ROOT/'outputs/submission_task2a.csv',index=False)
    forecast.to_csv(ROOT/'outputs/forecasts_with_keys.csv',index=False)
    report={'selection':selection,'holdout_metrics':hm,'validation_origins':[str(calendar.iloc[o].week_start.date()) for o in origins], 'holdout_start':str(calendar.iloc[holdout_origin+1].week_start.date()),'holdout_end':str(calendar.iloc[n-1].week_start.date()),'submission_rows':len(submission),'reload_exact_match':True}
    (ROOT/'outputs/run_summary.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report,indent=2),flush=True)
    return report

if __name__=='__main__': run()
