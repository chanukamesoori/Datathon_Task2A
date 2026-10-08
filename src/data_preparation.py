"""Strict order-date aggregation and auditable completeness checks."""
from pathlib import Path
import json
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def load_data():
    names = ['deliveries_train', 'task1_test_inputs', 'calendar', 'task2a_test_inputs', 'submission_task2a']
    return {n: pd.read_csv(ROOT / 'data' / f'{n}.csv') for n in names}

def prepare():
    raw = load_data()
    orders = pd.concat([raw['deliveries_train'], raw['task1_test_inputs']], ignore_index=True)
    required = ['delivery_id','order_date','depot','brand','temp_requirement','order_volume_m3']
    assert not orders[required].isna().any().any(), 'Missing required order data'
    dup = orders[orders.delivery_id.duplicated(False)]
    if len(dup):
        assert not dup.drop_duplicates().delivery_id.duplicated().any(), 'Conflicting duplicate IDs'
    orders = orders.drop_duplicates('delivery_id').copy()
    orders['date'] = pd.to_datetime(orders.order_date)
    assert set(orders.temp_requirement) <= {'ambient','chilled'}
    assert set(orders.dispatch_status) <= {'attempted','deferred','not_run'}
    dispatch = pd.to_datetime(orders.dispatch_date)
    inconsistencies = {
        'attempted_wrong_date': int(((orders.dispatch_status=='attempted') & (dispatch!=orders.date)).sum()),
        'deferred_not_later': int(((orders.dispatch_status=='deferred') & ~(dispatch>orders.date)).sum()),
        'not_run_has_dispatch': int(((orders.dispatch_status=='not_run') & dispatch.notna()).sum())}
    assert not any(inconsistencies.values()), 'Inconsistent dispatch records'
    cal = raw['calendar'].copy()
    cal['date'] = pd.to_datetime(cal.date)
    assert not cal.date.duplicated().any()
    assert not cal.drop(columns='festival').isna().any().any(), 'Missing calendar feature'
    cal['week_start'] = cal.date - pd.to_timedelta(cal.date.dt.dayofweek, unit='D')
    assert (cal.date.dt.isocalendar().year.to_numpy() == cal.iso_year).all()
    assert (cal.date.dt.isocalendar().week.to_numpy() == cal.iso_week).all()
    orders = orders.merge(cal[['date','iso_year','iso_week','week_start']], on='date', how='left', validate='many_to_one')
    assert orders.week_start.notna().all()
    assert orders.order_volume_m3.ge(0).all()
    assert set(orders.depot) == {'Kandy','Peliyagoda'}
    assert set(orders.brand) == {'Fresh','Style','Tech'}
    assert not ((orders.brand != 'Fresh') & (orders.temp_requirement == 'chilled')).any()
    orders['chilled'] = np.where((orders.brand == 'Fresh') & (orders.temp_requirement == 'chilled'), orders.order_volume_m3, 0)
    weekly = orders.groupby(['week_start','depot','brand']).agg(total=('order_volume_m3','sum'), chilled=('chilled','sum'), outlet_count=('outlet_id','nunique')).reset_index()
    calendar_week = cal.groupby('week_start').agg(iso_year=('iso_year','first'), iso_week=('iso_week','first'), days=('date','size'), operating=('is_operating','sum'), payday=('is_payday','sum'), festival_ramp=('festival_ramp','sum'), holiday=('is_holiday','sum'), monsoon=('monsoon','mean'), festival_days=('festival', lambda x: x.notna().sum())).reset_index()
    # Calendar completeness differs from demand completeness: retain a final Sunday only if non-operating.
    complete = []
    for w in weekly.week_start.unique():
        days = cal[cal.week_start == w]
        eligible = days[days.is_operating == 1].date
        if len(days) == 7 and eligible.ge(orders.date.min()).all() and eligible.le(orders.date.max()).all():
            complete.append(w)
    weekly = weekly[weekly.week_start.isin(complete)].merge(calendar_week, on='week_start', validate='many_to_one')
    dates = sorted(weekly.week_start.unique())
    assert len(dates) > 80
    assert (pd.Series(dates).diff().dropna() == pd.Timedelta(days=7)).all(), 'Missing weeks'
    assert weekly.groupby('week_start').size().eq(6).all(), 'Missing depot-brand combination'
    # A missing operating date can indicate incomplete source coverage; never silently fill it.
    missing_dates = cal[(cal.date >= orders.date.min()) & (cal.date <= orders.date.max()) & (cal.is_operating == 1) & ~cal.date.isin(orders.date)]
    assert missing_dates.empty, 'Missing operating dates'
    audit = {n: {'rows':len(d), 'columns':list(d), 'dtypes':d.dtypes.astype(str).to_dict(), 'missing':d.isna().sum().to_dict(), 'duplicate_rows':int(d.duplicated().sum())} for n,d in raw.items()}
    for n in ['deliveries_train','task1_test_inputs']:
        d = raw[n]
        audit[n].update(order_date_start=d.order_date.min(), order_date_end=d.order_date.max(), duplicate_delivery_ids=int(d.delivery_id.duplicated().sum()), statuses=d.dispatch_status.value_counts().to_dict(), temperature=d.temp_requirement.value_counts().to_dict())
    audit['dispatch_inconsistencies'] = inconsistencies
    audit.update(order_start=str(orders.date.min().date()), order_end=str(orders.date.max().date()), duplicate_delivery_ids=int(len(dup)), statuses=orders.dispatch_status.value_counts().to_dict(), depot_brand_counts=orders.groupby(['depot','brand']).size().to_string(), demand_summary=orders.order_volume_m3.describe().to_dict(), calendar_start=str(cal.date.min().date()), calendar_end=str(cal.date.max().date()), complete_weeks=len(dates), excluded_weeks=[str(x) for x in sorted(set(orders.week_start)-set(weekly.week_start))], missing_operating_dates=len(missing_dates), optional_outlets_present=(ROOT/'data/outlets.csv').exists())
    (ROOT/'outputs').mkdir(exist_ok=True)
    (ROOT/'outputs/data_audit.json').write_text(json.dumps(audit, indent=2, default=str))
    return raw, orders, weekly.sort_values(['week_start','depot','brand']), calendar_week, audit

if __name__ == '__main__':
    print(json.dumps(prepare()[-1], indent=2, default=str))
