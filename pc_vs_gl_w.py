from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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

    for bs_i, buy_or_sell in enumerate([is_buy, ~is_buy]):

        if bs_i == 0:
            bs = 'buy'
        else:
            bs = 'sell'

        # pca to compress indicator value
        X = df.iloc[:,8:].values
        X = np.array([df.iloc[i,8:].values for i in range(len(df)) if buy_or_sell[i]])
        pca = PCA(n_components=6)
        pca.fit(X)
        Xd = pca.transform(X)
        print('Cumulative contribution rate from 1st to 6th PC:')
        print(np.cumsum(pca.explained_variance_ratio_))

        for xi in range(6):
            # sliding window
            w_width = 10
            w_gp_mean, w_gp_std = [], []
            w_cent = []
            for w_st in np.arange(-70,150,5):
                w_cent.append(w_st + (w_width/2))
                is_w = (Xd[:,xi] >= w_st) & (Xd[:,xi] < w_st+10)
                w_gp_mean.append(np.mean(df[y_label].values[buy_or_sell][is_w]))
                w_gp_std.append(np.std(df[y_label].values[buy_or_sell][is_w]))
            w_cent = np.array(w_cent)
            w_gp_mean = np.array(w_gp_mean)
            w_gp_std = np.array(w_gp_std)
            err = np.array([w_gp_mean + w_gp_std, w_gp_mean-w_gp_std]).T

            # plot
            plt.close('all')
            plt.figure()
            plt.scatter(
                Xd[:,xi],
                df[y_label].values[buy_or_sell],
                label='samples',
                marker='+')
            plt.plot(w_cent,w_gp_mean*10,color='r',label=y_label+'_mean x 10')
            plt.plot(w_cent,err*10,color='r',label='$\pm$std x 10',linestyle='dashed', alpha=0.5)
            plt.grid()
            plt.xlabel('{}th-pcv-projected value'.format(xi+1))
            # plt.ylim([-100,100])
            plt.ylabel(y_label)
            plt.legend()
            plt.title('{}-th-pcv-projected value VS gain/loss ({})'.format(xi+1,bs))
            fname = str(out_dir_path/ 'pc_vs_gl_w_{}_pc{:02}.png'.format(bs,xi))
            print('saving: {}'.format(fname))
            plt.savefig(fname)
