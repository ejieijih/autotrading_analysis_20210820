from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression

if __name__ == '__main__':

    # set paths
    dir_path = Path('./data')
    csv_path = dir_path / 'パラメーターテスト4.txt'
    out_dir_path = Path('./out')
    out_dir_path.mkdir(parents=True, exist_ok=True)

    # set params
    j_time_frames = ['5M','15M','1H','4H','Daily']
    i_periods = ['10','20','40','80','200']
    indicators = ['wkRSI','wkBB','wkATR','wkSMA']
    y_label = 'wkFixPips'

    # load csv
    df = pd.read_csv(str(csv_path))
    keys = df.keys().values

    # set flags
    is_buy = df['strbuysell'].values == 'close_buy'

    # plot
    for bs, bs_index in zip(['buy','sell'],[is_buy, ~is_buy]):
        for indicator in indicators:
            for j in range(5):
                plt.close('all')
                plt.figure()
                for i in range(5):
                    x = df[indicator+'[{}][{}]'.format(i,j)].values[bs_index]
                    y = df[y_label].values[bs_index]
                    plt.scatter(
                        x, y,
                        label='period='+i_periods[i],
                        marker='+'
                    )
                for i in range(5):
                    x = df[indicator+'[{}][{}]'.format(i,j)].values[bs_index]
                    y = df[y_label].values[bs_index]
                    lr = LinearRegression()
                    lr.fit(x.reshape(-1,1),y)
                    l_x = np.arange(x.min(),x.max(),(x.max()-x.min())/20)
                    l_y = lr.predict(l_x.reshape(-1,1))
                    plt.plot(l_x,l_y*10,label='y_fit x 10 (period={})'.format(i_periods[i]))
                plt.grid()
                plt.xlabel(indicator+' (timeframe={})'.format(j_time_frames[j]))
                plt.ylabel(y_label)
                plt.legend()
                plt.title('Relationship between indicator and gain/loss ({})'.format(bs))
                fname = str(out_dir_path/ ('indicator_vs_gl_{}_{}_{}.png'.format(bs, indicator,j_time_frames[j])))
                print('saving: {}'.format(fname))
                plt.savefig(fname)
    

