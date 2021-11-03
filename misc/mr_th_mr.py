from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression, Ridge, Lasso

# set input paths
dir_path = Path('./data')
names=['PCA分析用_1時間足_20210923.csv','PCA分析用_5分足_20210923.csv','PCA分析用_1分足_ありあり5sp.csv']
ts_i = 2
csv_path = dir_path / names[ts_i]

# load csv
df = pd.read_csv(str(csv_path))
keys = df.keys().values

# set params
eval_year_range = [2006, 2022]
drive_pips = 10
drive_rate = 1
j_time_frames = ['5M','15M','1H','4H','Daily']
i_periods = ['10','20','40','80','200']
indicators = ['wkRSI','wkBB','wkATR','wkSMA']
y_label = 'wkFixPips'
pc_num = 20
indicator_start_label = 'SMA(5m)(10)'
st_indicator = list(keys).index(indicator_start_label)
is_buy = df['strbuysell'].values == 'close_buy'
ent_years = np.array([int(df['dttEntTime'].values[i][:4]) for i in range(len(df))])
is_2005 = ent_years == 2005

# get pca component
pc = []
for bs, bs_index in zip(['buy', 'sell'],[is_buy, (~is_buy)]):
    # target_index = bs_index
    target_index = bs_index & (~is_2005)
    X = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
    MN, STD = np.mean(X, axis=0), np.std(X, axis=0)
    X_norm = (X - MN)/STD
    pca = PCA(n_components=pc_num)
    pca.fit(X_norm)
    pc.append(np.concatenate([MN.reshape(1,-1),STD.reshape(1,-1),pca.components_],axis=0))

# make pc dataframe
df_pc= pd.DataFrame(np.array(pc).reshape(len(pc[1])*2,-1).T, 
    columns = ['buy_mean', 'buy_std'] + ['buy_pc{:02}'.format(num+1) for num in range(pc_num)] + \
        ['sell_mean', 'sell_std'] + ['sell_pc{:02}'.format(num+1) for num in range(pc_num)],
    index = keys[st_indicator:])


if __name__ == '__main__':

    # set alg
    algs = ['linear_regression','multiple_regression','search_all','ridge','lasso']
    ts_names = ['ts_1h','ts_5m','ts_1m']

    alg = algs[1]

    # set out paths
    out_dir_path = Path('./out') / ts_names[ts_i] / 'mr_th_mr'
    out_train_path = out_dir_path / 'train'
    out_test_path = out_dir_path / 'test'
    out_train_path.mkdir(parents=True, exist_ok=True)
    out_test_path.mkdir(parents=True, exist_ok=True)

    # experiments
    print('start analysis using {}'.format(alg))

    th_mr_0 = 0
    th_mr_1 = 0

    # save pc vectors
    filename = str(out_train_path / 'principal_components_100dim_to_pcs.csv')
    print('saving: {}'.format(filename))
    df_pc.to_csv(filename)

    # cross validation
    FixPips_yearsum_before, FixPips_yearsum_after, test_results, cancel_trade_num, trade_num = [], [], [], [], []
    negative_domains, drive_domains, x_fit, y_fit = [],[],[],[]
    pips_th = []
    mr_coefs = []
    for test_year in range(eval_year_range[0],eval_year_range[1]):
        print('test year: {}'.format(test_year))
        is_test = ent_years == test_year
        is_train = ~is_test
        # is_train = (~is_test) & (~is_2005)

        pc = []
        negative_domains_bs, drive_domains_bs, x_fit_bs, y_fit_bs = [],[],[],[]
        FixPips_yearsum_before_tmp2, FixPips_yearsum_after_tmp2,test_results_tmp2, cancel_trade_num_tmp2, trade_num_tmp2 = [], [], [], [], []
        pips_th_bs = []
        mr_coefs_bs = []
        for bs, bs_index in zip(['buy', 'sell'],[is_buy, ~is_buy]):

            #########
            # train #
            #########

            target_index = bs_index & is_train 

            # read pca params
            MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimension
            X_train = df.iloc[:,st_indicator:].values[target_index,:]
            X_train_norm = (X_train - MN)/STD
            Xd_train = np.dot(X_train_norm, df_pc.iloc[:,col_st:col_st + pc_num].values)
            y_train = df[y_label].values[target_index].reshape(-1,1)

            MR_model = LinearRegression()
            MR_model.fit(Xd_train, y_train)
            mr_coefs_bs.append(np.concatenate([MR_model.coef_[0], MR_model.intercept_]))

            # # thresholding
            # is_mr0_pos = (MR_model.predict(Xd_train) >= th_mr_0).reshape(-1)
            # Xd_mr0_pos, Xd_mr0_neg = Xd_train[is_mr0_pos,:], Xd_train[~is_mr0_pos,:]
            # y_mr0_pos, y_mr0_neg = y_train[is_mr0_pos], y_train[~is_mr0_pos]

            # # second regression
            # MR_model_10, MR_model_11 = LinearRegression(), LinearRegression()
            # MR_model_10.fit(Xd_mr0_pos, y_mr0_pos), MR_model_11.fit(Xd_mr0_neg, y_mr0_neg)

            # thresholding
            # is_mr0_pos = (MR_model.predict(Xd_train) >= th_mr_0).reshape(-1)
            # is_y_pos = (y_train >= 0).reshape(-1)
            # is_fail = (is_mr0_pos & (~is_y_pos)) + ((~is_mr0_pos) & is_y_pos)
            is_fail = ((ent_years == 2017) + (ent_years == 2018))[target_index]

            # second regression
            MR_model_1 = LinearRegression()
            MR_model_1.fit(Xd_train[is_fail,:],y_train[is_fail])


            ########
            # test #
            ########
            target_index = bs_index & is_test

            # read pca params
            MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimension
            X_test = df.iloc[:,st_indicator:].values[target_index,:]
            # X_test = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
            X_test_norm = (X_test - MN)/STD
            Xd_test = np.dot(X_test_norm, df_pc.iloc[:,col_st:col_st + pc_num].values)
            y_test = df[y_label].values[target_index].reshape(-1,1)

            # # thresholding
            # is_mr0_pos = (MR_model.predict(Xd_test) >= th_mr_0).reshape(-1)

            # # second regression
            # y_pred = MR_model.predict(Xd_test)
            # if is_mr0_pos.sum() > 0:
            #     y_pred[is_mr0_pos] = MR_model_10.predict(Xd_test[is_mr0_pos,:])
            # if (~is_mr0_pos).sum() > 0:
            #     y_pred[~is_mr0_pos]  = MR_model_11.predict(Xd_test[~is_mr0_pos,:])
            # rm_flag = y_pred.flatten() < th_mr_1

            y_pred_0 = MR_model.predict(Xd_test)
            y_pred_1 = MR_model_1.predict(Xd_test)
            rm_flag = (y_pred_0.flatten() < th_mr_0) + (y_pred_1.flatten() < th_mr_1)
            # rm_flag = y_pred_1.flatten() < th_mr_1

            # plt.close('all'); plt.scatter(y_test.flatten(),y_pred_0.flatten(),marker='.');plt.xlim([-20, 20]);plt.savefig('tmp.png')

            FixPips_yearsum_before_tmp2.append(df[y_label][is_test & (~bs_index)].sum()+df[y_label][target_index].sum())
            FixPips_yearsum_after_tmp2.append(df[y_label][is_test & (~bs_index)].sum()+df[y_label][target_index][~rm_flag].sum())
            test_results_tmp2.append(df[y_label][target_index][~rm_flag].sum()-df[y_label][target_index].sum())
            cancel_trade_num_tmp2.append(sum(rm_flag))
            trade_num_tmp2.append(sum(~rm_flag))

        mr_coefs.append(mr_coefs_bs)

        # if alg == 'linear_regression':
        negative_domains.append(negative_domains_bs)
        drive_domains.append(drive_domains_bs)
        x_fit.append(x_fit_bs)
        y_fit.append(y_fit_bs)

        pips_th.append(pips_th_bs)

        FixPips_yearsum_before.append(FixPips_yearsum_before_tmp2)
        FixPips_yearsum_after.append(FixPips_yearsum_after_tmp2)
        test_results.append(test_results_tmp2)
        cancel_trade_num.append(cancel_trade_num_tmp2)
        trade_num.append(trade_num_tmp2)

    mr_coefs = np.array(mr_coefs)
    negative_domains = np.array(negative_domains)
    drive_domains = np.array(drive_domains)
    pips_th = np.array(pips_th)
    x_fit = np.array(x_fit)
    y_fit = np.array(y_fit)

    FixPips_yearsum_before = np.array(FixPips_yearsum_before)
    FixPips_yearsum_after = np.array(FixPips_yearsum_after)
    test_results = np.array(test_results)
    cancel_trade_num = np.array(cancel_trade_num)
    trade_num = np.array(trade_num)
    test_results_mean = np.mean(test_results,axis=0)
    test_results_std = np.std(test_results,axis=0)

    print('{}/{} trades are cancelled'.format(cancel_trade_num.sum(),len(df)))

    # save train results
    mr_coefs_mat = mr_coefs.reshape(-1,pc_num+1)
    test_years = np.tile(np.arange(eval_year_range[0],eval_year_range[1]).reshape(-1,1),[1,2]).flatten()
    bs_arr = np.array([['buy','sell'] for i in range(len(mr_coefs))]).flatten()
    mr_coefs_df = pd.DataFrame(
        data = np.concatenate([test_years.reshape(-1,1),bs_arr.reshape(-1,1),mr_coefs_mat],axis=1),
        columns = ['test_year','bs']+['coef_pc{:02}'.format(i+1) for i in range(20)]+['bias'],
    )
    filename = out_train_path / 'mr_coefs.csv'
    print('saving: {}'.format(str(filename)))
    mr_coefs_df.to_csv(str(filename),index=False)

    # save test results
    bs_str = ['buy','sell']
    bs_i = 0
    both_after = FixPips_yearsum_after[:,0] + (FixPips_yearsum_after[:,1]-FixPips_yearsum_before[:,1])
    print('total pips is {}'.format(both_after.sum()))
    plt.close('all')
    plt.figure()
    plt.bar(np.arange(eval_year_range[0],eval_year_range[1])-0.2,FixPips_yearsum_before[:,bs_i],label='before',alpha=0.6,width=0.4)
    plt.bar(np.arange(eval_year_range[0],eval_year_range[1])+0.2,both_after,label='after',alpha=0.6,width=0.4)
    plt.bar(eval_year_range[1]-0.2,FixPips_yearsum_before[:,bs_i].mean(), label='average of before',alpha=0.6,width=0.4)
    plt.bar(eval_year_range[1]+0.2,both_after.mean(), label='average of after',alpha=0.6,width=0.4)
    plt.legend()
    plt.grid(axis='y')
    plt.xticks(ticks=np.arange(eval_year_range[0],eval_year_range[1]+1), labels=[str(i) for i in range(eval_year_range[0],eval_year_range[1])]+['average'], rotation=45)
    plt.title('FixPips sum before/after thresholding mr-projected values on both buy and sell\n(threshold is trained by {} to data other than test year)'.format(alg))
    filename = str(out_test_path / 'results_both.png')
    print('saving: {}'.format(filename))
    plt.savefig(filename)
