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

def experiment(alg, out_train_path, out_test_path):

    th_mr = 0

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

            if alg == 'multiple_regression':
                MR_model = LinearRegression()
                MR_model.fit(Xd_train, y_train)
                mr_coefs_bs.append(np.concatenate([MR_model.coef_[0], MR_model.intercept_]))
            elif alg == 'ridge':
                MR_model = Ridge()
                MR_model.fit(Xd_train, y_train)
                mr_coefs_bs.append(np.concatenate([MR_model.coef_[0], MR_model.intercept_]))
            elif alg == 'lasso':
                MR_model = Lasso()
                MR_model.fit(Xd_train, y_train)
                mr_coefs_bs.append(np.concatenate([MR_model.coef_[0], MR_model.intercept_]))

            else:

                negative_domains_pc, drive_domains_pc, x_fit_pc, y_fit_pc = [],[],[],[]
                pips_th_pc = []
                for pc_i in range(pc_num):

                    if alg == 'search_all':

                        # get search_all
                        x_min = Xd_train[:,pc_i].min()
                        x_max = Xd_train[:,pc_i].max()
                        th_ls = np.arange(x_min,x_max,(x_max-x_min)/50)[:50]
                        pips_th_i = []
                        for th_i in th_ls:
                            pos_dm = Xd_train[:,pc_i] > th_i
                            pips_th_i.append([y_train[pos_dm].mean(),y_train[~pos_dm].mean()])
                        pips_th_pc.append(pips_th_i)

                        # w_gp_mean = np.array(pips_th_i)[1:,1] - np.array(pips_th_i)[:-1,1]
                        w_gp_mean = np.array(pips_th_i)
                        w_cent = th_ls
                        
                        # calc negative_domain
                        tmp = np.array(pips_th_i).T.argmax()
                        pn_i, th_max = tmp // 50, tmp % 50
                        if pn_i == 0:
                            negative_domain = [[x_min,th_ls[th_max]]]
                        else:
                            negative_domain = [[th_ls[th_max],x_max]]
                        drive_domain = [[x_min, x_min]]  # dummy
                        
                    if alg == 'sliding_window':

                        # get statistics using sliding window
                        w_cent, w_gp_mean, w_num, w_stride, w_min, w_max = get_stats_using_sliding_window(Xd_train, pc_i, target_index)

                        # get negative domain
                        negative_domain = get_negative_domain(w_cent, w_gp_mean, w_stride, w_min, w_max)
                        drive_domain = get_negative_domain(w_cent,-(w_gp_mean-drive_pips), w_stride, w_min, w_max)

                    if alg == 'linear_regression':

                        # get regression
                        w_cent, w_gp_mean, w_num, w_stride, w_min, w_max = get_linear_regression(Xd_train, pc_i, target_index)

                        # get negative domain
                        negative_domain = get_negative_domain(w_cent, w_gp_mean, w_stride, w_min, w_max)
                        drive_domain = get_negative_domain(w_cent,-(w_gp_mean-drive_pips), w_stride, w_min, w_max)

                    negative_domains_pc.append(negative_domain)
                    drive_domains_pc.append(drive_domain)
                    x_fit_pc.append(w_cent[:50])
                    y_fit_pc.append(w_gp_mean[:50])

                    # # save plot
                    # if test_year == 2020:
                    #     plot_pcprojval_vs_fixpips(Xd_train, y_train, pc_i, w_cent, w_gp_mean, w_num, negative_domain,test_year,out_train_path,bs)

                negative_domains_bs.append(negative_domains_pc)
                drive_domains_bs.append(drive_domains_pc)
                pips_th_bs.append(pips_th_pc)
                x_fit_bs.append(x_fit_pc)
                y_fit_bs.append(y_fit_pc)

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

            if alg in ['multiple_regression','ridge','lasso']:

                y_pred = MR_model.predict(Xd_test)
                rm_flag = y_pred.flatten() < th_mr
                FixPips_yearsum_before_tmp2.append(df[y_label][is_test & (~bs_index)].sum()+df[y_label][target_index].sum())
                FixPips_yearsum_after_tmp2.append(df[y_label][is_test & (~bs_index)].sum()+df[y_label][target_index][~rm_flag].sum())
                test_results_tmp2.append(df[y_label][target_index][~rm_flag].sum()-df[y_label][target_index].sum())
                cancel_trade_num_tmp2.append(sum(rm_flag))
                trade_num_tmp2.append(sum(~rm_flag))

            else: 

                FixPips_yearsum_before_tmp, FixPips_yearsum_after_tmp, test_results_tmp, cancel_trade_num_tmp = [], [], [], []
                for use_pc in range(pc_num):

                    # make thresholding flag
                    rm_flag,drive_flag = [],[]
                    for i in range(len(Xd_test)):
                        rm_tmp,drive_tmp = False,False
                        for k in range(len(negative_domains_pc[use_pc])):
                            if len(negative_domains_pc[use_pc][k]) > 0:
                                if negative_domains_pc[use_pc][k][0] <= Xd_test[i,use_pc] < negative_domains_pc[use_pc][k][1]:
                                    rm_tmp = True
                            if len(drive_domains_pc[use_pc][k]) > 0:
                                if drive_domains_pc[use_pc][k][0] <= Xd_test[i,use_pc] < drive_domains_pc[use_pc][k][1]:
                                    drive_tmp = True
                                
                        rm_flag.append(rm_tmp)
                        drive_flag.append(drive_tmp)
                    rm_flag = np.array(rm_flag)
                    drive_flag = np.array(drive_flag)

                    # results
                    FixPips_sum_before = df[y_label][is_test & (~bs_index)].sum() + df[y_label][target_index].sum()
                    FixPips_sum_after = df[y_label][is_test & (~bs_index)].sum() + df[y_label][target_index][(~rm_flag)&(~drive_flag)].sum() + drive_rate*df[y_label][target_index][(~rm_flag)&(drive_flag)].sum()
                    FixPips_yearsum_before_tmp.append(FixPips_sum_before)
                    FixPips_yearsum_after_tmp.append(FixPips_sum_after)
                    test_results_tmp.append(FixPips_sum_after-FixPips_sum_before)
                    cancel_trade_num_tmp.append(rm_flag.sum())
                
                FixPips_yearsum_before_tmp2.append(FixPips_yearsum_before_tmp)
                FixPips_yearsum_after_tmp2.append(FixPips_yearsum_after_tmp)
                test_results_tmp2.append(test_results_tmp)
                cancel_trade_num_tmp2.append(cancel_trade_num_tmp)

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

if __name__ == '__main__':

    # set alg
    algs = ['linear_regression','multiple_regression','search_all','ridge','lasso']
    ts_names = ['ts_1h','ts_5m','ts_1m']

    alg = algs[1]

    # set out paths
    out_dir_path = Path('./out') / ts_names[ts_i] / alg
    out_train_path = out_dir_path / 'train'
    out_test_path = out_dir_path / 'test'
    out_train_path.mkdir(parents=True, exist_ok=True)
    out_test_path.mkdir(parents=True, exist_ok=True)

    # experiments
    print('start analysis using {}'.format(alg))
    experiment(alg, out_train_path, out_test_path)

