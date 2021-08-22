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
    pc_num = 12

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
    for bs, target_index in zip(['buy', 'sell'],[is_buy & is_train, (~is_buy) & is_train]):

        # pca to compress indicator value
        X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
        MN, STD = np.mean(X, axis=0), np.std(X, axis=0)
        X = np.array([(X[:,i]-MN[i])/STD[i] for i in range(len(X[0]))]).T
        pca = PCA(n_components=pc_num)
        pca.fit(X)
        Xd = pca.transform(X)
        print('Cumulative contribution rate from 1st to {}th PC:'.format(pc_num))
        print(np.cumsum(pca.explained_variance_ratio_))

        pc.append(np.concatenate([MN.reshape(1,-1),STD.reshape(1,-1),pca.components_],axis=0))

        for xi in range(pc_num):

            # sliding window
            w_gp_mean, w_gp_std, w_cent, w_num = [], [], [], []
            w_min, w_max = Xd[:,xi].min(), Xd[:,xi].max()
            w_stride = (w_max-w_min)/50
            w_width = w_stride*2
            for w_st in np.arange(w_min,w_max,w_stride):
                is_w = (Xd[:,xi] >= w_st) & (Xd[:,xi] < w_st+w_width)
                w_cent.append(w_st + (w_width/2))
                w_num.append(is_w.sum())
                w_gp_mean.append(np.mean(df[y_label].values[target_index][is_w]))
                w_gp_std.append(np.std(df[y_label].values[target_index][is_w]))
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
                df[y_label].values[target_index],
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
    pc_mat = np.array(pc).reshape(len(pc[1])*2,-1)
    df_pc= pd.DataFrame(pc_mat.T, 
        columns = ['buy_mean', 'buy_std'] + ['buy_pc{:02}'.format(num+1) for num in range(pc_num)] + \
            ['sell_mean', 'sell_std'] + ['sell_pc{:02}'.format(num+1) for num in range(pc_num)],
        index = keys[st_indicator:])
    print('saving: {}'.format(str(out_pc_path)))
    df_pc.to_csv(str(out_pc_path))

    # define threshold
    pc_rm_th = {
        'buy': [[12,21],[9.5,11],[100,100],[100,100],[3.5,4.5],[5.5,7],
                [3,4],[2.5,3],[3.5,4.5],[100,100],[2.5,3],[-3,-2]], 
        'sell': [[27,50],[6,11],[7.5,11],[-7,-2.5],[3,5],[3,5],
                [2,2.5],[-4,-3],[1,2],[1.5,2],[2.5,10],[-3.5,-2]]
        }


    '''
    test
    '''
    use_pc = {'buy':[4], 'sell':[0,4,5]}
    test_results, trading_results = [], []
    for bs, target_index in zip(['buy', 'sell'],[is_buy & is_test, (~is_buy) & is_test]):
    # for bs, target_index in zip(['buy', 'sell'],[is_buy & is_train, (~is_buy) & is_train]):
    
        # normalize data: actually the process should use all-period data
        X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
        MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
        X = np.array([(X[:,i]-MN[i])/STD[i] for i in range(len(X[0]))]).T

        # calculate pc values
        col_st = list(df_pc.keys()).index(bs+'_pc01')
        PC_val = np.dot(X, df_pc.iloc[:,col_st:col_st + pc_num].values)

        # make thresholding flag
        rm_flag = []
        for i in range(len(PC_val)):
            rm_tmp = False
            for j in use_pc[bs]:
                if (PC_val[i,j] >= pc_rm_th[bs][j][0]) & (PC_val[i,j] < pc_rm_th[bs][j][1]):
                    rm_tmp = True
            rm_flag.append(rm_tmp)        
        rm_flag = np.array(rm_flag)

        # results
        FixPips_sum_before = df[y_label][target_index].sum()
        FixPips_sum_after = df[y_label][target_index][~rm_flag].sum()
        trading_results.append([rm_flag.sum(), len(rm_flag)])
        test_results.append([FixPips_sum_before,FixPips_sum_after])

    test_results = np.array(test_results)
    print('buy:[cancel_num, trading_num], sell:[cancel_num, trading_num]')
    print(trading_results)
    print(test_results)

    plt.close('all')
    plt.figure()
    x = ['buy','buy w/ pc{}-th'.format(use_pc['buy']),'sell','sell w/ pc{}-th'.format(use_pc['sell'])]
    plt.bar(x, [test_results[0,0],0,test_results[1,0],0],alpha=0.6)
    plt.bar(x, [0,test_results[0,1],0,test_results[1,1]],alpha=0.6)
    plt.grid(axis='y')
    plt.ylabel('sum of '+y_label)
    plt.legend()
    plt.title('test results (train: ~2019, test:2020~)')
    fname = str(out_dir_path/ 'exp_results.png')
    print('saving: {}'.format(fname))
    plt.savefig(fname)
