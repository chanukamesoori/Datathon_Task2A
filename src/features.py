"""Features are constructed from an explicit origin, never future demand."""
import numpy as np
import pandas as pd

CAL = ['operating','payday','festival_ramp','holiday','monsoon','festival_days']

def make_row(values, outlets, origin, target, calendar, depot, brand, enhanced=True):
    history = np.asarray(values[:origin+1], dtype=float)
    scale = max(history[-13:].mean(), 1e-6)
    c = calendar.iloc[target]
    row = {'horizon':target-origin, 'trend_time':target/52, 'depot_Kandy':int(depot=='Kandy'), 'brand_Fresh':int(brand=='Fresh'), 'brand_Style':int(brand=='Style')}
    for lag in [0,1,2,3,7,12,25]:
        row[f'lag_{lag}'] = history[-1-lag]/scale
    for window in [4,13,26]:
        row[f'mean_{window}'] = history[-window:].mean()/scale
        row[f'std_{window}'] = history[-window:].std()/scale
    row['recent_trend'] = (history[-4:].mean()-history[-13:].mean())/scale
    row['seasonal_lag'] = values[target-52]/scale if 0 <= target-52 <= origin else 1.
    row['seasonal_available'] = int(0 <= target-52 <= origin)
    for name in CAL:
        row[name] = float(c[name])
    for harmonic in range(1, 4 if enhanced else 2):
        for fn, func in [('sin',np.sin),('cos',np.cos)]:
            row[f'{fn}_{harmonic}'] = func(2*np.pi*harmonic*float(c.iso_week)/52.1775)
    if enhanced:
        row['ramp_next'] = float(calendar.iloc[min(target+1,len(calendar)-1)].festival_ramp)
        row['ramp_previous'] = float(calendar.iloc[max(target-1,0)].festival_ramp)
        row['outlets_recent'] = float(np.mean(outlets[max(0,origin-3):origin+1]))
        row['outlets_change'] = row['outlets_recent']/max(float(np.mean(outlets[max(0,origin-12):origin+1])),1)
        for name in CAL + ['seasonal_lag','recent_trend']:
            for b in ['Fresh','Style','Tech']:
                row[f'{name}_{b}'] = row[name]*int(brand==b)
    return row, scale

def training_table(series, calendar, end, target_name, enhanced=True, recursive=False, share=False):
    rows, ys = [], []
    for (depot,brand), frame in series.items():
        if target_name == 'chilled' and brand != 'Fresh':
            continue
        v = frame[target_name].to_numpy()
        if share:
            v = frame.chilled.to_numpy()/np.maximum(frame.total.to_numpy(),1e-6)
        for origin in range(26,end):
            for h in range(1, min(1 if recursive else 10,end-origin)+1):
                row, scale = make_row(v,frame.outlet_count.to_numpy(),origin,origin+h,calendar,depot,brand,enhanced)
                rows.append(row)
                ys.append(v[origin+h]/scale)
    return pd.DataFrame(rows), np.asarray(ys)
