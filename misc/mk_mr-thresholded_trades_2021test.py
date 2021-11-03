from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pandas.io import json
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

    algs = ['linear_regression','multiple_regression','search_all','ridge','lasso']
    ts_names = ['ts_1h','ts_5m','ts_1m']

    # set alg with 'multiple_regression'
    alg = algs[1]

    # set out paths
    out_dir_path = Path('./out') / ts_names[ts_i] / alg
    out_train_path = out_dir_path / 'train'
    out_test_path = out_dir_path / 'test'
    out_train_path.mkdir(parents=True, exist_ok=True)
    out_test_path.mkdir(parents=True, exist_ok=True)

    # cross validation
    FixPips_yearsum_before, FixPips_yearsum_after, test_results, cancel_trade_num = [], [], [], []
    negative_domains, drive_domains, x_fit, y_fit = [],[],[],[]
    pips_th = []
    mr_coefs = []

    # set test_year with 2021
    # test_year = 2021
    for test_year in range(eval_year_range[0]+1,eval_year_range[1]):

        print('test year: {}'.format(test_year))
        is_test = ent_years == test_year
        is_train = ~is_test
        # is_train = (~is_test) & (~is_2005)

        pc = []
        negative_domains_bs, drive_domains_bs, x_fit_bs, y_fit_bs = [],[],[],[]
        FixPips_yearsum_before_tmp2, FixPips_yearsum_after_tmp2,test_results_tmp2, cancel_trade_num_tmp2 = [], [], [], []
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

            ########
            # test #
            ########
            target_index = bs_index

            # read pca params
            MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimension
            X_test = df.iloc[:,st_indicator:].values[target_index,:]
            X_test_norm = (X_test - MN)/STD
            Xd_test = np.dot(X_test_norm, df_pc.iloc[:,col_st:col_st + pc_num].values)
            y_test = df[y_label].values[target_index].reshape(-1,1)

            if alg in ['multiple_regression','ridge','lasso']:

                y_pred = MR_model.predict(Xd_test)
                rm_flag = y_pred.flatten() < 0

            # extract mr-thresholded trades
            exec('mr_thresholded_trades_{} = df[target_index][~rm_flag]'.format(bs))

        mr_thresholded_trades_total = pd.concat([mr_thresholded_trades_buy, mr_thresholded_trades_sell]).sort_index()

        # trade analysis
        ent_years_trade = np.array([int(time[:4]) for time in mr_thresholded_trades_total['dttEntTime'].values])
        year = list(range(2005,2022))
        trade_num = [(ent_years_trade==year).sum() for year in range(2005,2022)]
        pips_sum = [mr_thresholded_trades_total[y_label].values[ent_years_trade==year].sum() for year in range(2005,2022)]
        averaged_pips = (np.array(pips_sum) / np.array(trade_num)).tolist()
        trade_analysis = pd.DataFrame(
            columns = ['year','trade_num','pips_sum','averaged_pips'],
            data = np.array([year,trade_num,pips_sum,averaged_pips]).T
        )

        # save mr-thresholded trades
        out_path = Path('out/mr-thresholded_trades/test{}/mr-thresholded_trades.csv'.format(test_year))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print('saving: {}'.format(str(out_path)))
        mr_thresholded_trades_total.to_csv(str(out_path))

        # save analysis results
        out_path = Path('out/mr-thresholded_trades/test{}/analysis_results.csv'.format(test_year))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print('saving: {}'.format(str(out_path)))
        trade_analysis.to_csv(str(out_path))

        # save 
        mr_coefs = pd.DataFrame(
            data = np.array(mr_coefs_bs),
            index = ['buy','sell'],
            columns = ['coef_pc{:02}'.format(i+1) for i in range(20)]+['bias']
        )
        out_path = Path('out/mr-thresholded_trades/test{}/mr_coefs.csv'.format(test_year))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        print('saving: {}'.format(str(out_path)))
        mr_coefs.to_csv(str(out_path))
        
    out_path = Path('out/mr-thresholded_trades/pca_coefs.csv')
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print('saving: {}'.format(str(out_path)))
    df_pc.to_csv(str(out_path))


