"""Bounded model search and direct/recursive forecasting."""
import numpy as np
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from catboost import CatBoostRegressor
from lightgbm import LGBMRegressor
from src.features import make_row, training_table

SPECS = {
    'naive':{}, 'moving_average':{}, 'seasonal_naive':{}, 'trend_average':{},
    'ridge_basic':{'kind':'ridge','alpha':10,'enhanced':False},
    'ridge_10':{'kind':'ridge','alpha':10}, 'ridge_100':{'kind':'ridge','alpha':100},
    'random_forest':{'kind':'rf'}, 'hist_gradient':{'kind':'hist'},
    'catboost_depth4':{'kind':'cat','depth':4}, 'catboost_depth6':{'kind':'cat','depth':6},
    'lightgbm_15':{'kind':'lgb','leaves':15}, 'lightgbm_7':{'kind':'lgb','leaves':7},
    'ridge_recursive':{'kind':'ridge','alpha':100,'recursive':True},
    'ridge_by_brand':{'kind':'ridge','alpha':100,'by_brand':True},
    'ridge_chilled_share':{'kind':'ridge','alpha':100,'share':True}
}

def estimator(spec):
    k=spec['kind']
    if k=='ridge': return make_pipeline(StandardScaler(),Ridge(alpha=spec['alpha']))
    if k=='rf': return RandomForestRegressor(n_estimators=120,max_depth=10,min_samples_leaf=12,n_jobs=4,random_state=42)
    if k=='hist': return HistGradientBoostingRegressor(max_iter=160,max_leaf_nodes=15,l2_regularization=5,random_state=42)
    if k=='cat': return CatBoostRegressor(iterations=220,depth=spec['depth'],learning_rate=.05,l2_leaf_reg=8,verbose=False,thread_count=4,random_seed=42,allow_writing_files=False)
    return LGBMRegressor(n_estimators=200,num_leaves=spec['leaves'],learning_rate=.04,min_child_samples=30,reg_lambda=5,verbosity=-1,n_jobs=4,random_state=42)

def fit(name, series, calendar, end, target):
    spec=SPECS[name]
    if 'kind' not in spec: return {'name':name,'target':target}
    share=spec.get('share',False) and target=='chilled'
    X,y=training_table(series,calendar,end,target,spec.get('enhanced',True),spec.get('recursive',False),share)
    models={}
    if spec.get('by_brand'):
        masks={'Fresh':X.brand_Fresh.eq(1),'Style':X.brand_Style.eq(1),'Tech':X.brand_Fresh.eq(0)&X.brand_Style.eq(0)}
        for b,mask in masks.items():
            if mask.any(): models[b]=estimator(spec).fit(X.loc[mask],y[mask])
    else: models['all']=estimator(spec).fit(X,y)
    return {'name':name,'target':target,'models':models,'columns':list(X),'share':share}

def predict(bundle, series, calendar, origin, horizons=10, total_predictions=None):
    name,target=bundle['name'],bundle['target']
    spec=SPECS[name]
    results=[]
    for (depot,brand),frame in series.items():
        if target=='chilled' and brand!='Fresh': continue
        v=frame[target].to_numpy()[:origin+1].tolist()
        if bundle.get('share'): v=(frame.chilled/np.maximum(frame.total,1e-6)).to_numpy()[:origin+1].tolist()
        outlets=frame.outlet_count.to_numpy()[:origin+1]
        for h in range(1,horizons+1):
            t=origin+h
            if name=='naive': p=v[-1]
            elif name=='moving_average': p=np.mean(v[-4:])
            elif name=='seasonal_naive': p=v[t-52] if t>=52 else np.mean(v[-4:])
            elif name=='trend_average': p=np.mean(v[-4:])+h*(np.mean(v[-4:])-np.mean(v[-13:]))/9
            else:
                recursive=spec.get('recursive',False)
                effective_origin=t-1 if recursive else origin
                padded_outlets=np.pad(outlets,(0,max(0,effective_origin+1-len(outlets))),mode='edge')
                row,scale=make_row(v,padded_outlets,effective_origin,t,calendar,depot,brand,spec.get('enhanced',True))
                model=bundle['models'].get(brand,bundle['models'].get('all'))
                p=float(model.predict(pd.DataFrame([row])[bundle['columns']])[0])*scale
                if recursive: v.append(max(0,p))
                if bundle.get('share'): p=np.clip(p,0,1)*total_predictions[(depot,brand,h)]
            results.append({'depot':depot,'brand':brand,'horizon':h,'week_start':calendar.iloc[t].week_start,'target':target,'predicted':max(0,p)})
    return pd.DataFrame(results)

def make_series(weekly):
    return {k:f.sort_values('week_start').reset_index(drop=True) for k,f in weekly.groupby(['depot','brand'])}
