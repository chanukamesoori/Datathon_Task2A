"""Measured metrics, including future ground-truth matching."""
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

def metrics(actual, predicted):
    a,p = np.asarray(actual),np.asarray(predicted)
    error = p-a
    denom = np.abs(a)+np.abs(p)
    return {'MAE':float(np.abs(error).mean()), 'RMSE':float(np.sqrt(np.mean(error**2))), 'WAPE':float(100*np.abs(error).sum()/np.abs(a).sum()) if np.abs(a).sum() else np.nan, 'sMAPE':float(200*np.divide(np.abs(error),denom,out=np.zeros_like(error,dtype=float),where=denom>0).mean()), 'R2':float(r2_score(a,p)) if len(a)>1 and np.var(a)>0 else np.nan}

def summarize(predictions, groups):
    result=[]
    for keys, frame in predictions.groupby(groups):
        keys = keys if isinstance(keys,tuple) else (keys,)
        result.append(dict(zip(groups,keys)) | metrics(frame.actual,frame.predicted) | {'n':len(frame)})
    return pd.DataFrame(result)

def evaluate_future(forecasts, ground_truth):
    """Ground truth must supply total and chilled labels; fail on incomplete matching."""
    keys=['depot','brand','iso_year','iso_week']
    actual=pd.read_csv(ground_truth) if isinstance(ground_truth,(str,bytes)) else ground_truth.copy()
    assert not actual.duplicated(keys).any()
    matched=forecasts.merge(actual[keys+['total','chilled']],on=keys,how='left',validate='one_to_one')
    assert matched[['total','chilled']].notna().all().all()
    return pd.DataFrame([{'target':t,**metrics(matched[t],matched[f'pred_{t}_volume_m3'])} for t in ['total','chilled']])
