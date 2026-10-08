"""Notebook figures: development-only EDA and honest forecast evaluation."""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from src.features import CAL

def finish(fig, name, root):
    fig.tight_layout()
    (root/'outputs/figures').mkdir(parents=True,exist_ok=True)
    fig.savefig(root/'outputs/figures'/f'{name}.png',dpi=140,bbox_inches='tight')
    plt.show()

def trends(weekly, root, chilled=False):
    data=weekly[weekly.brand=='Fresh'] if chilled else weekly
    groups=list(data.groupby(['depot','brand']))
    fig,axes=plt.subplots(len(groups),1,figsize=(12,2.5*len(groups)),squeeze=False,sharex=True)
    for ax,((depot,brand),f) in zip(axes.flat,groups):
        ax.plot(f.week_start,f['chilled' if chilled else 'total'],lw=1.4)
        ax.set(title=f'{depot} / {brand}',ylabel='mÂ³')
        ax.grid(alpha=.2)
    finish(fig,'fresh_chilled' if chilled else 'weekly_demand',root)

def seasonality(weekly,root):
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,(brand,f) in zip(axes,weekly.groupby('brand')):
        for depot,g in f.groupby('depot'):
            ax.plot(g.groupby('iso_week').total.mean(),label=depot)
        ax.set(title=brand,xlabel='ISO week',ylabel='Mean weekly mÂ³')
        ax.legend()
    finish(fig,'seasonality',root)

def effects(weekly,root):
    data=weekly.copy()
    data['relative_demand']=data.total/data.groupby(['depot','brand']).total.transform('mean')
    fig,axes=plt.subplots(2,3,figsize=(14,8))
    for ax,name in zip(axes.flat,CAL):
        if data[name].nunique()>1:
            groups=data.groupby(name).relative_demand.agg(['mean','count'])
            ax.plot(groups.index,groups['mean'],'o-')
            ax.set(title=f'{name} ({data[name].nunique()} levels)',xlabel='Calendar feature',ylabel='Relative demand')
        else:
            ax.text(.1,.5,f'{name} is constant: {data[name].iloc[0]}',transform=ax.transAxes)
            ax.set_title(name+' (not identifiable)')
    finish(fig,'calendar_effects',root)
    return data[['relative_demand']+CAL].corr(numeric_only=True).iloc[0].drop('relative_demand')

def distributions(weekly,root):
    groups=list(weekly.groupby(['depot','brand']))
    fig,axes=plt.subplots(2,3,figsize=(14,7))
    rows=[]
    for ax,((d,b),f) in zip(axes.flat,groups):
        ax.hist(f.total,bins=18,alpha=.75)
        ax.set(title=d+'/'+b,xlabel='Weekly mÂ³',ylabel='Weeks')
        q1,q3=f.total.quantile([.25,.75]); iqr=q3-q1
        rows.append({'depot':d,'brand':b,'CV':f.total.std()/f.total.mean(),'IQR_outliers':int(((f.total<q1-1.5*iqr)|(f.total>q3+1.5*iqr)).sum())})
    finish(fig,'distributions',root)
    return pd.DataFrame(rows)

def autocorrelation(weekly,root):
    fig,ax=plt.subplots(figsize=(12,4))
    rows=[]
    for (d,b),f in weekly.groupby(['depot','brand']):
        values=[f.total.autocorr(lag) for lag in range(1,54)]
        ax.plot(range(1,54),values,label=d+'/'+b,alpha=.75)
        rows.append({'depot':d,'brand':b,'lag1':values[0],'lag4':values[3],'lag52':values[51]})
    ax.axhline(0,color='black',lw=.5)
    ax.set(xlabel='Lag (weeks)',ylabel='Correlation',title='Demand autocorrelation (descriptive, not causal)')
    ax.legend(ncol=3)
    finish(fig,'autocorrelation',root)
    return pd.DataFrame(rows)

def evaluation_charts(pred,root):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,((d,b),f) in zip(axes.flat,pred[pred.target=='total'].groupby(['depot','brand'])):
        f=f.sort_values('week_start')
        ax.plot(f.week_start,f.actual,label='Actual')
        ax.plot(f.week_start,f.predicted,label='Forecast')
        ax.set(title=d+'/'+b,ylabel='mÂ³')
        ax.tick_params(axis='x',rotation=30)
        ax.legend()
    finish(fig,'validation_actual_predicted',root)
    p=pred.copy();p['error']=p.predicted-p.actual;p['absolute_error']=p.error.abs()
    fig,axes=plt.subplots(2,2,figsize=(13,8))
    for t,f in p.groupby('target'):
        axes[0,0].hist(f.error,bins=25,alpha=.5,label=t)
        axes[0,1].scatter(f.predicted,f.error,s=12,alpha=.6,label=t)
        axes[1,0].plot(f.groupby('horizon').absolute_error.mean(),label=t,marker='o')
    p[p.target=='total'].groupby(['depot','brand']).absolute_error.mean().plot.bar(ax=axes[1,1])
    axes[0,0].set(title='Signed prediction errors',xlabel='Forecast - actual (mÂ³)')
    axes[0,1].axhline(0,color='black');axes[0,1].set(title='Residuals versus fitted demand',xlabel='Forecast mÂ³',ylabel='Error mÂ³')
    axes[1,0].set(title='MAE by forecast horizon',xlabel='Weeks ahead',ylabel='MAE mÂ³')
    axes[1,1].set(title='Total MAE by depot and brand',ylabel='MAE mÂ³')
    for ax in axes.flat:
        if ax!=axes[1,1]:ax.legend()
    finish(fig,'validation_errors',root)

def holdout_charts(hp,root):
    fig,axes=plt.subplots(2,1,figsize=(12,7))
    dates=pd.to_datetime(hp.iso_year.astype(str)+'-W'+hp.iso_week.astype(str).str.zfill(2)+'-1',format='%G-W%V-%u')
    for ax,target in zip(axes,['total','chilled']):
        f=hp.assign(week_start=dates).groupby('week_start')[[target,f'pred_{target}_volume_m3']].sum()
        f.plot(ax=ax,marker='o');ax.set(title='Independent holdout: '+target,ylabel='mÂ³')
    finish(fig,'holdout_actual_predicted',root)

def future_chart(weekly,forecast,root):
    fig,axes=plt.subplots(2,3,figsize=(15,8))
    for ax,((d,b),f) in zip(axes.flat,weekly.groupby(['depot','brand'])):
        ax.plot(f.week_start.tail(30),f.total.tail(30),label='Observed')
        p=forecast[(forecast.depot==d)&(forecast.brand==b)].copy()
        p['date']=pd.to_datetime(p.iso_year.astype(str)+'-W'+p.iso_week.astype(str).str.zfill(2)+'-1',format='%G-W%V-%u')
        ax.plot(p.date,p.pred_total_volume_m3,label='Forecast',marker='o')
        ax.set(title=d+'/'+b,ylabel='mÂ³');ax.tick_params(axis='x',rotation=30);ax.legend()
    finish(fig,'final_forecasts',root)

def importance(artifact,root):
    rows=[]
    for target,bundles in artifact['bundles'].items():
        for name,bundle in bundles.items():
            for brand,model in bundle.get('models',{}).items():
                if hasattr(model,'feature_importances_'): values=model.feature_importances_
                elif hasattr(model,'named_steps'): values=np.abs(model.named_steps['ridge'].coef_)
                else: continue
                values=np.asarray(values,dtype=float)
                values=values/max(values.sum(),1e-12)
                rows.extend({'target':target,'model':name,'brand_model':brand,'feature':col,'importance':float(v)} for col,v in zip(bundle['columns'],values))
    result=pd.DataFrame(rows)
    if len(result):
        fig,axes=plt.subplots(1,2,figsize=(14,5))
        for ax,target in zip(axes,['total','chilled']):
            result[result.target==target].groupby('feature').importance.mean().nlargest(12).sort_values().plot.barh(ax=ax)
            ax.set(title=target+' model importance',xlabel='Mean normalized importance (descriptive)')
        finish(fig,'feature_importance',root)
        result.to_csv(root/'outputs/feature_importance.csv',index=False)
    return result
