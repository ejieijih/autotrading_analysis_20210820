from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':

    # set paths
    dir_path = Path('./data')
    csv_path = dir_path / 'パラメーターテスト2_keys.txt'
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
    for indicator in indicators:
        for j in range(5):
            plt.close('all')
            plt.figure()
            # plt.scatter(theta,y3,label=r'y=sin$\theta$+0.3*N(0,1)')
            # plt.plot(theta,y,label=r'y=sin$\theta$',color='r')
            for i in range(5):
                plt.scatter(
                    df[indicator+'[{}][{}]'.format(i,j)].values[is_buy], df[y_label].values[is_buy],
                    label='period='+i_periods[i],
                    marker='+'
                )
            plt.grid()
            plt.xlabel(indicator+' (timeframe={})'.format(j_time_frames[j]))
            plt.ylabel(y_label)
            plt.legend()
            plt.title('Relationship between oscilator value and realized gain/loss')
            fname = str(out_dir_path/ ('osc_vs_gl_{}_{}.png'.format(indicator,j_time_frames[j])))
            print('saving: {}'.format(fname))
            plt.savefig(fname)
    

