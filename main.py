from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import yaml
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression


# set input paths
dir_path = Path('./data')
csv_path = dir_path / 'パラメーターテスト4.txt'

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
is_2005 = ent_years == 2005

# get pca component
pc = []
for bs, bs_index in zip(['buy', 'sell'],[is_buy, (~is_buy)]):
    target_index = bs_index
    # target_index = bs_index & (~is_2005)
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

def plot_pcprojval_vs_fixpips(Xd_train, y_train, pc_i, w_cent, w_gp_mean, w_num, negative_domain,test_year,out_train_path,bs):

    fname = out_train_path / ('test_year'+str(test_year)) / 'pc_vs_gl_w_{}_pc{:02}.png'.format(bs,pc_i)
    fname.parent.mkdir(parents=True, exist_ok=True)

    plt.close('all')
    plt.figure()
    plt.hist(Xd_train[:,pc_i],30,alpha=0.6,color='g',label='histgram')
    plt.scatter(
        Xd_train[:,pc_i],
        y_train,
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
    print('saving: {}'.format(str(fname)))
    plt.savefig(str(fname))

def get_linear_regression(Xd_train, pc_i, target_index):

    x, y = Xd_train[:,pc_i].reshape(-1,1), df[y_label].values[target_index]
    lr = LinearRegression()
    lr.fit(x, y)
    w_min, w_max = x.min(),x.max()
    w_stride = (w_max - w_min)/50
    w_cent = np.arange(w_min, w_max, w_stride)
    w_num = len(w_cent)
    w_gp_mean = lr.predict(w_cent.reshape(-1,1)).flatten()

    return w_cent, w_gp_mean, w_num, w_stride, w_min, w_max

def get_negative_domain(w_cent, w_gp_mean, w_stride, w_min, w_max):

    is_positive = (w_gp_mean >= 0).astype(int)
    is_boundary = is_positive[1:] - is_positive[:-1]
    down_pcval = [w_cent[1:][i] for i in range(len(is_boundary)) if is_boundary[i] == -1]
    up_pcval = list(np.array([w_cent[1:][i] for i in range(len(is_boundary)) if is_boundary[i] == 1]) - w_stride)

    negative_domain = [[]]

    if (len(down_pcval) > 0) & (len(up_pcval) > 0):
        if down_pcval[0] > up_pcval[0]:
            if len(down_pcval) == len(up_pcval):
                negative_domain = np.array([down_pcval, up_pcval[1:]+[w_max]]).T.tolist()
            elif len(down_pcval) < len(up_pcval):
                negative_domain = np.array([down_pcval, up_pcval[1:]]).T.tolist()
        else:
            if len(down_pcval) == len(up_pcval):
                negative_domain = np.array([down_pcval, up_pcval]).T.tolist()
            elif len(down_pcval) > len(up_pcval):
                negative_domain = np.array([down_pcval, up_pcval+[w_max]]).T.tolist()
    elif (len(down_pcval) <= 0) & (len(up_pcval) > 0):
        negative_domain = [[w_min, up_pcval[0]]]
    elif (len(down_pcval) > 0) & (len(up_pcval) <= 0):
        negative_domain = [[down_pcval[0], w_max]]

    return negative_domain

def experiment(alg, out_train_path, out_test_path):

    filename = str(out_train_path / 'principal_components_100dim_to_pcs.csv')
    print('saving: {}'.format(filename))
    df_pc.to_csv(filename)

    FixPips_yearsum_before, FixPips_yearsum_after, test_results, cancel_trade_num = [], [], [], []
    for test_year in range(2006,2021):
        print('test year: {}'.format(test_year))
        is_test = ent_years == test_year
        is_train = ~is_test
        # is_train = (~is_test) & (~is_2005)

        pc = []
        negative_domains = {}
        FixPips_yearsum_before_tmp2, FixPips_yearsum_after_tmp2,test_results_tmp2, cancel_trade_num_tmp2 = [], [], [], []
        for bs, bs_index in zip(['buy', 'sell'],[is_buy, ~is_buy]):

            #########
            # train
            #########

            target_index = bs_index & is_train 

            # read pca params
            MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimension
            X_train = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
            X_train_norm = (X_train - MN)/STD
            Xd_train = np.dot(X_train_norm, df_pc.iloc[:,col_st:col_st + pc_num].values)
            y_train = df[y_label].values[target_index].reshape(-1,1)

            if alg == 'multiple_regression':
                MR_model = LinearRegression()
                MR_model.fit(Xd_train, y_train)

                if test_year == 2020:
                    for pc_i in range(pc_num):
                        plt.close('all')
                        plt.figure()
                        plt.scatter(Xd_train[:,pc_i],y_train,label='data',marker='+')
                        plt.scatter(Xd_train[:,pc_i],MR_model.predict(Xd_train),label='predict',marker='+')
                        plt.legend()
                        plt.grid()
                        filepath = out_train_path/str(test_year)/'{}_pc{:02}_vs_fixpips.png'.format(bs,pc_i)
                        filepath.parent.mkdir(parents=True,exist_ok=True)
                        print('print: {}'.format(str(filepath)))
                        plt.savefig(str(filepath))

            else:
                negative_domain_tmp = []
                for pc_i in range(pc_num):

                    if alg == 'sliding_window':

                        # get statistics using sliding window
                        w_cent, w_gp_mean, w_num, w_stride, w_min, w_max = get_stats_using_sliding_window(Xd_train, pc_i, target_index)

                    if alg == 'linear_regression':

                        # get regression
                        w_cent, w_gp_mean, w_num, w_stride, w_min, w_max = get_linear_regression(Xd_train, pc_i, target_index)

                    # get negative domain
                    negative_domain = get_negative_domain(w_cent, w_gp_mean, w_stride, w_min, w_max)
                    negative_domain_tmp.append(negative_domain)

                    # save plot
                    if test_year == 2020:
                        plot_pcprojval_vs_fixpips(Xd_train, y_train, pc_i, w_cent, w_gp_mean, w_num, negative_domain,test_year,out_train_path,bs)

                negative_domains[bs] = negative_domain_tmp

                # save thresholds
                if test_year == 2020:
                    for key in negative_domains.keys():
                        negative_domains[key] = [[[np.array(tmp2).item() for tmp2 in tmp1] for tmp1 in tmp0] for tmp0 in negative_domains[key]]
                    filename = str(out_train_path/('test_year'+str(test_year))/'threshold.yml')
                    print('saving: {}'.format(filename))
                    with open(filename,'w',encoding='utf-8') as f:
                        yaml.dump(negative_domains,f,encoding='utf-8',allow_unicode=True)
                    
                    # negative_domains_df = pd.DataFrame(
                    #     data=np.concatenate([negative_domains['buy'],negative_domains['sell']], axis=0),
                    #     columns=['0_low','0_high'],
                    #     index=sum([['{}-PC{:02}'.format(bs,i) for i in range(pc_num)] for bs in ['buy','sell']],[])
                    # )

            #######
            # test
            #######
            target_index = bs_index & is_test

            # read pca params
            MN, STD = df_pc[bs+'_mean'].values, df_pc[bs+'_std'].values
            col_st = list(df_pc.keys()).index(bs+'_pc01')

            # compress dimension
            X_test = np.array([df.iloc[i,st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
            X_test_norm = (X_test - MN)/STD
            Xd_test = np.dot(X_test_norm, df_pc.iloc[:,col_st:col_st + pc_num].values)
            y_test = df[y_label].values[target_index].reshape(-1,1)

            if alg == 'multiple_regression':

                y_pred = MR_model.predict(Xd_test)
                rm_flag = y_pred.flatten() < 0
                FixPips_yearsum_before_tmp2.append(df[y_label][target_index].sum())
                FixPips_yearsum_after_tmp2.append(df[y_label][target_index][~rm_flag].sum())
                test_results_tmp2.append(df[y_label][target_index][~rm_flag].sum()-df[y_label][target_index].sum())
                cancel_trade_num_tmp2.append(sum(rm_flag))

            else: 

                FixPips_yearsum_before_tmp, FixPips_yearsum_after_tmp, test_results_tmp, cancel_trade_num_tmp = [], [], [], []
                for use_pc in range(pc_num):

                    # make thresholding flag
                    rm_flag = []
                    for i in range(len(Xd_test)):
                        rm_tmp = False
                        for k in range(len(negative_domains[bs][use_pc])):
                            if len(negative_domains[bs][use_pc][k]) > 0:
                                if negative_domains[bs][use_pc][k][0] <= Xd_test[i,use_pc] < negative_domains[bs][use_pc][k][1]:
                                    rm_tmp = True
                        rm_flag.append(rm_tmp)        
                    rm_flag = np.array(rm_flag)

                    # results
                    FixPips_sum_before = df[y_label][target_index].sum()
                    FixPips_sum_after = df[y_label][target_index][~rm_flag].sum()
                    # trading_results.append([rm_flag.sum(), len(rm_flag)])
                    FixPips_yearsum_before_tmp.append(FixPips_sum_before)
                    FixPips_yearsum_after_tmp.append(FixPips_sum_after)
                    test_results_tmp.append(FixPips_sum_after-FixPips_sum_before)
                    cancel_trade_num_tmp.append(rm_flag.sum())
                
                FixPips_yearsum_before_tmp2.append(FixPips_yearsum_before_tmp)
                FixPips_yearsum_after_tmp2.append(FixPips_yearsum_after_tmp)
                test_results_tmp2.append(test_results_tmp)
                cancel_trade_num_tmp2.append(cancel_trade_num_tmp)

        FixPips_yearsum_before.append(FixPips_yearsum_before_tmp2)
        FixPips_yearsum_after.append(FixPips_yearsum_after_tmp2)
        test_results.append(test_results_tmp2)
        cancel_trade_num.append(cancel_trade_num_tmp2)

    FixPips_yearsum_before = np.array(FixPips_yearsum_before)
    FixPips_yearsum_after = np.array(FixPips_yearsum_after)
    test_results = np.array(test_results)
    cancel_trade_num = np.array(cancel_trade_num)
    test_results_mean = np.mean(test_results,axis=0)
    test_results_std = np.std(test_results,axis=0)

    if alg == 'multiple_regression':
        bs_str = ['buy','sell']
        for bs_i in [0,1]:
            plt.close('all')
            plt.figure()
            plt.bar(np.arange(2006,2021)-0.2,FixPips_yearsum_before[:,bs_i],label='before',alpha=0.6,width=0.4)
            plt.bar(np.arange(2006,2021)+0.2,FixPips_yearsum_after[:,bs_i],label='after',alpha=0.6,width=0.4)
            plt.bar(2021-0.2,FixPips_yearsum_before[:,bs_i].mean(), label='average of before',alpha=0.6,width=0.4)
            plt.bar(2021+0.2,FixPips_yearsum_after[:,bs_i].mean(), label='average of after',alpha=0.6,width=0.4)
            plt.legend()
            plt.grid(axis='y')
            plt.xticks(ticks=np.arange(2006,2022), labels=[str(i) for i in range(2006,2021)]+['average'], rotation=45)
            plt.title('FixPips sum on {} before/after thresholding {} pc-projected values \
                \n(threshold is trained by {} to data other than test year)'.format(bs_str[bs_i],pc_num,alg))
            filename = str(out_test_path / 'results_{}.png'.format(bs_str[bs_i]))
            print('saving: {}'.format(filename))
            plt.savefig(filename)

    else:

        # save results
        FixPips_yearsum_df = pd.DataFrame(
            data = np.concatenate([FixPips_yearsum_before[:,:,0].reshape(-1,1),FixPips_yearsum_after.reshape(-1,pc_num)],axis=1),
            columns = ['before']+['PC{:02}-TH'.format(i+1) for i in range(pc_num)],
            index = sum([['{}-{}'.format(year,bs) for bs in ['buy','sell']] for year in range(2006,2021)],[])
        )
        filename = str(out_test_path / 'results_FixPips_yearsum.csv')
        print('saving: {}'.format(filename))
        FixPips_yearsum_df.to_csv(filename,float_format='%.1f')

        positive_index = np.array(np.where(test_results_mean>0)).T

        bs_str = ['buy','sell']
        for bs_i, pc_i in positive_index:
            plt.close('all')
            plt.figure()
            plt.bar(np.arange(2006,2021)-0.2,FixPips_yearsum_before[:,bs_i,pc_i],label='before',alpha=0.6,width=0.4)
            plt.bar(np.arange(2006,2021)+0.2,FixPips_yearsum_after[:,bs_i,pc_i],label='after',alpha=0.6,width=0.4)
            plt.bar(2021-0.2,FixPips_yearsum_before[:,bs_i,pc_i].mean(), label='average of before',alpha=0.6,width=0.4)
            plt.bar(2021+0.2,FixPips_yearsum_after[:,bs_i,pc_i].mean(), label='average of after',alpha=0.6,width=0.4)
            plt.legend()
            plt.grid(axis='y')
            plt.xticks(ticks=np.arange(2006,2022), labels=[str(i) for i in range(2006,2021)]+['average'], rotation=45)
            plt.title('FixPips sum on {} before/after thresholding {}th-pc-projected value \
                \n(threshold is trained by {} to data other than test year)'.format(bs_str[bs_i],pc_i+1,alg))
            filename = str(out_test_path / 'results_{}_pc{:02}.png'.format(bs_str[bs_i],pc_i))
            print('saving: {}'.format(filename))
            plt.savefig(filename)

if __name__ == '__main__':

    # set alg
    algs = ['linear_regression','multiple_regression','sliding_window']
    
    for alg in algs:
    # alg = algs[2]

        # set out paths
        out_dir_path = Path('./out') / alg
        out_train_path = out_dir_path / 'train'
        out_test_path = out_dir_path / 'test'
        out_train_path.mkdir(parents=True, exist_ok=True)
        out_test_path.mkdir(parents=True, exist_ok=True)

        # experiments
        print('start analysis using {}'.format(alg))
        experiment(alg, out_train_path, out_test_path)

