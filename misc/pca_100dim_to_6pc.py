from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

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
    st_indicator = 15

    # load csv
    df = pd.read_csv(str(csv_path))
    keys = df.keys().values

    # set flags
    is_buy = df['strbuysell'].values == 'close_buy'
    is_positive = df['wkFixPips'][is_buy].values > 0

    # pca to compress indicator value
    X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if is_buy[i]])
    X = np.array([X[:,i]/np.std(X[:,i]) for i in range(len(X[0]))]).T
    pca = PCA(n_components=6)
    pca.fit(X)
    Xd = pca.transform(X)
    print('Cumulative contribution rate from 1st to 6th PC:')
    print(np.cumsum(pca.explained_variance_ratio_))

    # plot
    for xi in [0,2,4]:
        yi = xi + 1
        plt.close('all')
        plt.figure()
        plt.scatter(
            Xd[:,xi][is_positive], Xd[:,yi][is_positive],
            label='positive wkFixPips',
            marker='+')
        plt.scatter(
            Xd[:,xi][~is_positive], Xd[:,yi][~is_positive],
            label='negative wkFixPips',
            marker='+')
        plt.grid()
        plt.xlabel('{}th-pcv-projected value'.format(xi+1))
        plt.ylabel('{}th-pcv-projected value'.format(yi+1))
        plt.legend()
        plt.title('Distribution of positive and negative gain/loss on {}, {}-th pc'.format(xi+1,yi+1))
        fname = str(out_dir_path/ ('dist_pn_{:02}_{:02}pc.png'.format(xi+1, yi+1)))
        print('saving: {}'.format(fname))
        plt.savefig(fname)
