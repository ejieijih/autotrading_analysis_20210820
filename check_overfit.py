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
    out_train_path = out_dir_path / 'train'
    out_test_path = out_dir_path / 'test'
    out_pc_path = out_train_path / 'principal_components_100dim_to_6val.csv'
    out_dir_path.mkdir(parents=True, exist_ok=True)
    out_train_path.mkdir(parents=True, exist_ok=True)
    out_test_path.mkdir(parents=True, exist_ok=True)

    # set params
    j_time_frames = ['5M','15M','1H','4H','Daily']
    i_periods = ['10','20','40','80','200']
    indicators = ['wkRSI','wkBB','wkATR','wkSMA']
    y_label = 'wkFixPips'
    indicator_start_label = 'wkSMA[0][0]'

    # load csv
    df = pd.read_csv(str(csv_path))
    keys = df.keys().values
    st_indicator = list(keys).index(indicator_start_label)

    # set flags
    is_buy = df['strbuysell'].values == 'close_buy'
    ent_years = np.array([int(df['Ent_DAY'].values[i][:4]) for i in range(len(df))])
    is_train = ent_years <= 2019
    is_test = ~is_train


    '''
    training
    '''
    pc = []
    for bs_i, buy_or_sell in enumerate([is_buy & is_train, (~is_buy) & is_train]):

        if bs_i == 0:
            bs = 'buy'
        else:
            bs = 'sell'

        # pca to compress indicator value
        X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (buy_or_sell[i])])
        X = np.array([(X[:,i]-np.mean(X[:,i]))/np.std(X[:,i]) for i in range(len(X[0]))]).T
        pca = PCA(n_components=6)
        pca.fit(X)
        Xd = pca.transform(X)
        print('Cumulative contribution rate from 1st to 6th PC:')
        print(np.cumsum(pca.explained_variance_ratio_))

        pc.append(pca.components_)

        for xi in range(6):

            # sliding window
            w_gp_mean, w_gp_std, w_cent, w_num = [], [], [], []
            w_min, w_max = Xd[:,xi].min(), Xd[:,xi].max()
            w_stride = (w_max-w_min)/50
            w_width = w_stride*2
            for w_st in np.arange(w_min,w_max,w_stride):
                w_cent.append(w_st + (w_width/2))
                is_w = (Xd[:,xi] >= w_st) & (Xd[:,xi] < w_st+10)
                w_num.append(is_w.sum())
                w_gp_mean.append(np.mean(df[y_label].values[buy_or_sell][is_w]))
                w_gp_std.append(np.std(df[y_label].values[buy_or_sell][is_w]))
            w_cent = np.array(w_cent)
            w_gp_mean = np.array(w_gp_mean)
            w_gp_std = np.array(w_gp_std)
            w_num = np.array(w_num)
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
            # plt.plot(w_cent,err*10,color='r',label='$\pm$std x 10',linestyle='dashed', alpha=0.5)  
            plt.plot(w_cent, w_num, label='sample_num',linestyle='dashed',color='g',alpha=0.6)
            plt.grid()
            plt.xlabel('{}th-pcv-projected value'.format(xi+1))
            # plt.ylim([-100,100])
            plt.ylabel(y_label)
            plt.legend()
            plt.title('{}-th-pcv-projected value VS gain/loss ({})'.format(xi+1,bs))
            fname = str(out_train_path/ 'pc_vs_gl_w_{}_pc{:02}.png'.format(bs,xi))
            print('saving: {}'.format(fname))
            plt.savefig(fname)

    # save principal components
    pc = np.array(pc).reshape(len(pc[0])*2,-1)
    df_pc= pd.DataFrame(pc.T, 
        columns = ['buy_pc{:02}'.format(num+1) for num in range(6)]+['sell_pc{:02}'.format(num+1) for num in range(6)],
        index = keys[st_indicator:])
    print('saving: {}'.format(str(out_pc_path)))
    df_pc.to_csv(str(out_pc_path))

    # define threshold
    buy_pc_rm_th = [[10,17],[8,10],[8,19],[10,15],[9,15],[5,8]]
    sell_pc_rm_th = [[18,36],[5,12],[3,20],[3,12],[20,20],[1.5,15]]

    '''
    test
    '''
    test_results, trading_results = [], []
    for bs_i, buy_or_sell in enumerate([is_buy & is_test, (~is_buy) & is_test]):

        if bs_i == 0:
            bs = 'buy'
        else:
            bs = 'sell'

        # normalize data: actually the process should use all-period data
        X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (buy_or_sell[i])])
        X = np.array([(X[:,i]-np.mean(X[:,i]))/np.std(X[:,i]) for i in range(len(X[0]))]).T

        # calculate pc values
        PC_val = np.dot(X,pc[:6,:].T)

        # make thresholding flag
        rm_flag = []
        for i in range(len(PC_val)):
            if bs == 'buy':
                pc_rm_th = buy_pc_rm_th
            elif bs == 'sell':
                pc_rm_th = sell_pc_rm_th
            rm_tmp = False
            for pc_rm_th_i in pc_rm_th:
                if (PC_val[i,0] >= pc_rm_th_i[0]) & (PC_val[i,0] < pc_rm_th_i[1]):
                    rm_tmp = True
            rm_flag.append(rm_tmp)        
        rm_flag = np.array(rm_flag)

        # results
        FixPips_sum_before = df[y_label][buy_or_sell].sum()
        FixPips_sum_after = df[y_label][buy_or_sell][~rm_flag].sum()
        trading_results.append([rm_flag.sum(), len(rm_flag)])
        test_results.append([FixPips_sum_before,FixPips_sum_after])

    test_results = np.array(test_results)
    print('buy:[cancel_num, trading_num], sell:[cancel_num, trading_num]')
    print(trading_results)

    plt.close('all')
    plt.figure()
    plt.bar(['buy_before','buy_after','sell_before','sell_after'], [test_results[0,0],0,test_results[1,0],0],alpha=0.6)
    plt.bar(['buy_before','buy_after','sell_before','sell_after'], [0,test_results[0,1],0,test_results[1,1]],alpha=0.6)
    plt.grid(axis='y')
    plt.ylabel('sum of '+y_label)
    plt.legend()
    plt.title('test results (train: ~2019, test:2020~)')
    fname = str(out_dir_path/ 'exp_results.png')
    print('saving: {}'.format(fname))
    plt.savefig(fname)
