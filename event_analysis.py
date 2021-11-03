import pandas as pd
from pathlib import Path
import datetime
import numpy as np
import matplotlib.pyplot as plt

# set paths
in_dat_path = Path('./data/PCA分析用_1時間足_20210923.csv')
in_ev_path = Path('./data/economic_calender_US_high.csv')
out_hist_path = Path('./out/event_analysis/histogram.png')
out_bar_path = Path('./out/event_analysis/bar_plot.png')

# load data
dat = pd.read_csv(str(in_dat_path))
event = pd.read_csv(str(in_ev_path))

# get event datetime
is_allday = (event['time'] == 'All Day').values
is_tentative = (event['time'] == 'Tentative').values
is_nan = event['time'].isnull().values
is_time = ~(is_allday + is_tentative + is_nan)
ev_dt = [datetime.datetime.strptime(event['date'][i]+event['time'][i], "%d/%m/%Y%H:%M") for i in range(len(event)) if is_time[i]]
ev_dt = np.array(ev_dt)

# calc data jst
time_dif = {True:datetime.timedelta(hours=6),False:datetime.timedelta(hours=7)}

dat_ent_month = np.array([time[5:7] for time in dat['dttEntTime'].values]).astype(int)
ent_is_summer = (dat_ent_month >= 3) & (dat_ent_month<12)
EntTime_JST=np.array([datetime.datetime.strptime(tmp,"%Y.%m.%d %H:%M:%S")+time_dif[ent_is_summer[i]] for i,tmp in enumerate(dat['dttEntTime'].values)])

dat_ext_month = np.array([time[5:7] for time in dat['dttExtTime'].values]).astype(int)
ext_is_summer = (dat_ext_month >= 3) & (dat_ext_month<12)
ExtTime_JST=np.array([datetime.datetime.strptime(tmp,"%Y.%m.%d %H:%M:%S")+time_dif[ext_is_summer[i]] for i,tmp in enumerate(dat['dttExtTime'].values)])

tmp = pd.DataFrame(data=np.array([EntTime_JST, ExtTime_JST]).T,columns=['EntTime_JST','ExtTime_JST'])
dat_jst = pd.concat([dat,tmp],axis=1)

dat_ent_year = np.array([time[:4] for time in dat['dttEntTime'].values]).astype(int)
is_2005 = dat_ent_year < 2006


# get trade on event
trade_on_event, trade_entry_1h_event = [], []
for i in range(len(dat)):
    if i % 100 == 0:
        print('{}/{}'.format(i,len(dat)))
    ent_dt = dat_jst['EntTime_JST'][i]
    ext_dt = dat_jst['ExtTime_JST'][i]
    trade_on_event.append(bool(((ent_dt < ev_dt) & (ev_dt < ext_dt)).sum()))
    trade_entry_1h_event.append(bool(((ev_dt < ent_dt) & (ent_dt < (ev_dt + datetime.timedelta(hours=1)))).sum()))
trade_on_event = np.array(trade_on_event)
trade_entry_1h_event = np.array(trade_entry_1h_event)
pips_off_event = dat['wkFixPips'].values[~trade_on_event & ~is_2005]
pips_on_event = dat['wkFixPips'].values[trade_on_event]
pips_ent_1h_event = dat['wkFixPips'].values[trade_entry_1h_event]
pips_all = dat['wkFixPips'].values
pips_misc = dat['wkFixPips'].values[~(trade_entry_1h_event + trade_on_event)]

# make histogram
bins = range(-150,150,5)
plt.close('all')
plt.hist(pips_all,bins,label='all',alpha=0.6)
plt.hist(pips_misc,bins,label='other than [1 or 2]',alpha=0.6)
plt.hist(pips_on_event,bins,label='1.[ent -> event -> ext]',alpha=0.6)
plt.hist(pips_ent_1h_event,bins,label='2.[event -> ent -> event+1hour]',alpha=0.6)
plt.grid()
plt.legend()
plt.xlim([-100,150])
out_hist_path.parent.mkdir(parents=True,exist_ok=True)
print('saving: {}'.format(str(out_hist_path)))
plt.savefig(str(out_hist_path))

# make bar plot
plt.close('all')
plt.bar(0,pips_all.mean(),label='all',alpha=0.6)
plt.bar(1,pips_misc.mean(),label='other than [1 or 2]',alpha=0.6)
plt.bar(2,pips_on_event.mean(),label='1.[ent -> event -> ext]',alpha=0.6)
plt.bar(3,pips_ent_1h_event.mean(),label='2.[event -> ent -> event+1hour]',alpha=0.6)
plt.grid(axis='y')
plt.title('averaged pips/trade')
plt.legend()
out_bar_path.parent.mkdir(parents=True,exist_ok=True)
print('saving: {}'.format(str(out_bar_path)))
plt.savefig(str(out_bar_path))


