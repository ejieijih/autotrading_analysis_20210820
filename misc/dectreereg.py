from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier, plot_tree
from sklearn import tree
from IPython.display import Image
import pydotplus


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

if __name__ == '__main__':

    # set alg
    algs = ['linear_regression','multiple_regression','search_all','ridge','lasso']
    ts_names = ['ts_1h','ts_5m','ts_1m']

    alg = algs[1]

    # set out paths
    out_dir_path = Path('./out') / ts_names[ts_i] / 'dectreereg'
    out_train_path = out_dir_path / 'train'
    out_test_path = out_dir_path / 'test'
    out_train_path.mkdir(parents=True, exist_ok=True)
    out_test_path.mkdir(parents=True, exist_ok=True)

    # experiments
    print('start analysis using {}'.format(alg))

    for nodes in [6]:

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

                # compress dimension
                X_train = df.iloc[:,st_indicator:].values[target_index,:]
                # y_train = df[y_label].values[target_index].reshape(-1,1)
                # y_train = df[y_label].values[target_index]
                y_train = (df[y_label].values[target_index] > 0).astype(int)

                # reg = DecisionTreeRegressor(max_leaf_nodes=100)
                reg = DecisionTreeClassifier(max_leaf_nodes=nodes)
                model = reg.fit(X_train, y_train)

                fname = out_train_path/'leafnodes{:03}'.format(nodes) /'dectree_ts{}_{}.png'.format(test_year,bs)
                fname.parent.mkdir(parents=True,exist_ok=True)
                plt.close('all')
                plt.figure(figsize=[15,10])
                plot_tree(model, feature_names=df.keys()[st_indicator:], class_names= ['negative','positive'], filled=True)
                plt.savefig(fname)

                ########
                # test #
                ########
                target_index = bs_index & is_test

                # compress dimension
                X_test = df.iloc[:,st_indicator:].values[target_index,:]
                # y_test = df[y_label].values[target_index].reshape(-1,1)
                # y_test = df[y_label].values[target_index]
                y_test = df[y_label].values[target_index]

                y_pred = model.predict(X_test)
                rm_flag = y_pred.flatten() == 0
                y_test[~rm_flag].sum()

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
        filename = str(out_test_path / 'results_both_{:03}.png'.format(nodes))
        print('saving: {}'.format(filename))
        plt.savefig(filename)
