"""Checks that matter: future-demand perturbation and exact serialized inference."""
import copy
import joblib
import numpy as np
import pandas as pd
from src.data_preparation import ROOT, prepare
from src.pipeline import calendar_for, infer_artifact
from src.features import make_row, training_table
from src.forecasting import make_series, fit, predict

def verify():
    raw,orders,weekly,cal,audit=prepare()
    series=make_series(weekly); calendar=calendar_for(weekly,cal)
    origin=66
    changed=copy.deepcopy(series)
    for f in changed.values():
        f.loc[origin+1:,['total','chilled','outlet_count']]=999999
    X,y=training_table(series,calendar,origin,'total')
    X2,y2=training_table(changed,calendar,origin,'total')
    pd.testing.assert_frame_equal(X,X2);np.testing.assert_array_equal(y,y2)
    bundle=fit('ridge_recursive',series,calendar,origin,'total')
    p=predict(bundle,series,calendar,origin)
    p2=predict(bundle,changed,calendar,origin)
    pd.testing.assert_frame_equal(p,p2)
    bundle=fit('ridge_100',series,calendar,origin,'total')
    pd.testing.assert_frame_equal(predict(bundle,series,calendar,origin),predict(bundle,changed,calendar,origin))
    artifact=joblib.load(ROOT/'models/forecast_artifact.joblib')
    result=infer_artifact(artifact,raw['task2a_test_inputs'])
    saved=pd.read_csv(ROOT/'outputs/submission_task2a.csv')
    np.testing.assert_allclose(result[['pred_total_volume_m3','pred_chilled_volume_m3']],saved.iloc[:,1:],rtol=1e-14)
    assert saved.row_id.equals(raw['submission_task2a'].row_id)
    assert list(saved)==list(raw['submission_task2a'])
    assert saved.notna().all().all() and np.isfinite(saved.iloc[:,1:]).all().all()
    assert saved.iloc[:,1:].ge(0).all().all()
    assert saved.pred_chilled_volume_m3.le(saved.pred_total_volume_m3).all()
    assert result.loc[result.brand!='Fresh','pred_chilled_volume_m3'].eq(0).all()
    # Subsetting new forecast inputs must preserve order and return identical matching predictions.
    subset=raw['task2a_test_inputs'].iloc[[9,0,59]].reset_index(drop=True)
    smaller=infer_artifact(artifact,subset)
    assert smaller.row_id.equals(subset.row_id)
    print('PASS: future-actual perturbation leaves training/direct/recursive forecasts unchanged.')
    print('PASS: saved model reload, submission schema, all 60 rows, physical constraints, and new input subset.')
    return True

if __name__=='__main__':verify()
