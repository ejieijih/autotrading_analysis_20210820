from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# set paths
dir_path = Path('./data')
csv_path = dir_path / 'パラメーターテスト4.txt'
out_dir_path = Path('./out')
out_train_path = out_dir_path / 'train'
out_test_path = out_dir_path / 'test'
out_pc_path = out_train_path / 'principal_components_100dim_to_pcs.csv'
out_dir_path.mkdir(parents=True, exist_ok=True)
out_train_path.mkdir(parents=True, exist_ok=True)
out_test_path.mkdir(parents=True, exist_ok=True)

# load csv
df = pd.read_csv(str(csv_path))
keys = df.keys().values

# set params
j_time_frames = ['5M','15M','1H','4H','Daily']
i_periods = ['10','20','40','80','200']
indicators = ['wkRSI','wkBB','wkATR','wkSMA']
y_label = 'wkFixPips'
pc_num = 20
indicator_start_label = 'wkSMA[0][0]'
st_indicator = list(keys).index(indicator_start_label)
is_buy = df['strbuysell'].values == 'close_buy'
ent_years = np.array([int(df['Ent_DAY'].values[i][:4]) for i in range(len(df))])

# get pca component
pc = []
for bs, target_index in zip(['buy', 'sell'],[is_buy, (~is_buy)]):
    X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
    MN, STD = np.mean(X, axis=0), np.std(X, axis=0)
    X = np.array([(X[:,i]-MN[i])/STD[i] for i in range(len(X[0]))]).T
    pca = PCA(n_components=pc_num)
    pca.fit(X)
    pc.append(np.concatenate([MN.reshape(1,-1),STD.reshape(1,-1),pca.components_],axis=0))

def pca_to_compress_dimension(target_index):

    # get target data
    X_train = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
    MN, STD = np.mean(X_train, axis=0), np.std(X_train, axis=0)

    # normalization
    X_train = np.array([(X_train[:,i]-MN[i])/STD[i] for i in range(len(X_train[0]))]).T

    # get pca-vec-projected-values
    pca = PCA(n_components=pc_num)
    pca.fit(X_train)
    Xd_train = pca.transform(X_train)
    print('Cumulative contribution rate from 1st to {}th PC:'.format(pc_num))
    print(np.cumsum(pca.explained_variance_ratio_))

    return pca, Xd_train, MN, STD

def get_stats_using_sliding_window(Xd_train, pc_i, target_index):

    w_gp_mean, w_gp_std, w_cent, w_num = [], [], [], []
    w_min, w_max = Xd_train[:,pc_i].min(), Xd_train[:,pc_i].max()
    w_stride = (w_max-w_min)/50
    w_width = w_stride*2
    for w_st in np.arange(w_min,w_max,w_stride):
        is_w = (Xd_train[:,pc_i] >= w_st) & (Xd_train[:,pc_i] < w_st+w_width)
        w_cent.append(w_st + (w_width/2))
        w_num.append(is_w.sum())
        tmp = df[y_label].values[target_index][is_w]
        if len(tmp) > 0:
            w_gp_mean.append(np.mean(tmp))
            w_gp_std.append(np.std(tmp))
        else:
            w_gp_mean.append(0)
            w_gp_std.append(0)

    w_cent = np.array(w_cent)
    w_gp_mean = np.array(w_gp_mean)
    w_gp_std = np.array(w_gp_std)
    w_num = np.array(w_num)
    err = np.array([w_gp_mean + w_gp_std, w_gp_mean-w_gp_std]).T

    return w_cent, w_gp_mean, w_num, w_stride, w_min, w_max

def plot_pcprojval_vs_fixpips(Xd_train, pc_i, w_cent, w_gp_mean, w_num, negative_domain):

    plt.close('all')
    plt.figure()
    plt.scatter(
        Xd_train[:,pc_i],
        df[y_label].values[target_index],
        label='samples',
        marker='+')
    plt.plot(w_cent,w_gp_mean*10,color='r',label=y_label+'_mean x 10')
    # plt.plot(w_cent, w_num, label='sample_num',linestyle='dashed',color='g',alpha=0.6)
    plt.plot(np.array(negative_domain).T,np.zeros(np.array(negative_domain).shape).T,label='negative domain',color='b', linewidth=3)
    plt.grid()
    plt.xlabel('{}th-pcv-projected value'.format(pc_i+1))
    # plt.ylim([-100,100])
    plt.ylabel(y_label)
    plt.legend()
    plt.title('{}-th-pcv-projected value VS gain/loss ({})'.format(pc_i+1,bs))
    fname = str(out_train_path/ 'pc_vs_gl_w_{}_pc{:02}.png'.format(bs,pc_i))
    print('saving: {}'.format(fname))
    plt.savefig(fname)

def save_pc_vecs(pc):

    pc_mat = np.array(pc).reshape(len(pc[1])*2,-1)
    df_pc= pd.DataFrame(pc_mat.T, 
        columns = ['buy_mean', 'buy_std'] + ['buy_pc{:02}'.format(num+1) for num in range(pc_num)] + \
            ['sell_mean', 'sell_std'] + ['sell_pc{:02}'.format(num+1) for num in range(pc_num)],
        index = keys[st_indicator:])
    print('saving: {}'.format(str(out_pc_path)))
    df_pc.to_csv(str(out_pc_path))

    return df_pc

if __name__ == '__main__':

    # save principal components
    df_pc = save_pc_vecs(pc)

    test_results, cancel_trade_num = [], []
    for test_year in range(2006,2021):
        print('test year: {}'.format(test_year))
        is_test = ent_years == test_year
        is_train = ~is_test

        test_results_bs, cancel_trade_num_bs = [], []
        for bs, bs_index in zip(['buy', 'sell'],[is_buy, ~is_buy]):

            '''
            train
            '''
            target_index = bs_index & is_train

            # read pca params
            MN, STD = df_pc[bs+'_mean'], df_pc[bs+'_std']
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimesnion
            X_train = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
            X_train = np.array([(X_train[:,i]-MN[i])/STD[i] for i in range(len(X_train[0]))]).T
            Xd_train = np.dot(X_train, df_pc.iloc[:,col_st:col_st + pc_num].values)

            # multiple regression
            xx, y = Xd_train, df[y_label].values[target_index]
            lr = LinearRegression()
            # lr = Ridge(random_state=0)
            # lr = Lasso(random_state=0)
            lr.fit(xx, y)

            '''
            test
            '''
            target_index = bs_index & is_test
            
            # compress dimesnion
            X_test = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
            X_test = np.array([(X_test[:,i]-MN[i])/STD[i] for i in range(len(X_test[0]))]).T
            Xd_test = np.dot(X_test, df_pc.iloc[:,col_st:col_st + pc_num].values)

            # make thresholding flag
            y_pred = lr.predict(Xd_test)
            rm_flag = y_pred < 0

            # results
            FixPips_sum_before = df[y_label][target_index].sum()
            FixPips_sum_after = df[y_label][target_index][~rm_flag].sum()
            test_results_bs.append(FixPips_sum_after - FixPips_sum_before)
            cancel_trade_num_bs.append(rm_flag.sum())

        test_results.append(test_results_bs)
        cancel_trade_num.append(cancel_trade_num_bs)

    test_results = np.array(test_results)
    cancel_trade_num = np.array(cancel_trade_num)
    test_results_mean = np.mean(test_results,axis=0)
    test_results_std = np.std(test_results,axis=0)

    print(test_results_mean)
    print(test_results_std)

    positive_results = (test_results_mean > 0)
    buy_pcs = [i for i in range(len(positive_results)) if positive_results[i,0]]
    sell_pcs = [i for i in range(len(positive_results)) if positive_results[i,1]]

    buy_sum = np.array([tmp[positive_results[:,0]] for tmp in test_results[:,:,0]]).sum(axis=1)
    sell_sum = np.array([tmp[positive_results[:,1]] for tmp in test_results[:,:,1]]).sum(axis=1)

    plt.close('all')
    plt.figure()
    plt.bar(['sell_pc{}'.format(buy_pcs),'buy_pc{}'.format(sell_pcs)],[buy_sum.mean(),sell_sum.mean()],yerr=[buy_sum.std(),sell_sum.std()])
    plt.ylabel('FixPips improvement per year (mean $\pm$ std)')
    plt.grid(axis='y')
    # plt.savefig('out/tmp.png')
    # plt.savefig('out/sliding_window.png')
    plt.savefig('out/linear_regression.png')


    # plt.close('all')
    # plt.figure()
    # x = ['buy','buy w/ pc{}-th'.format(use_pc['buy']),'sell','sell w/ pc{}-th'.format(use_pc['sell'])]
    # plt.bar(x, [test_results[0,0],0,test_results[1,0],0],alpha=0.6)
    # plt.bar(x, [0,test_results[0,1],0,test_results[1,1]],alpha=0.6)
    # plt.grid(axis='y')
    # plt.ylabel('sum of '+y_label)
    # plt.legend()
    # plt.title('test results (train: ~2019, test:2020~)')
    # fname = str(out_dir_path/ 'exp_results.png')
    # print('saving: {}'.format(fname))
    # plt.savefig(fname)