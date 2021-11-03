from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression

class PCA_MR:

    def __init__(self, csv_path, indicator_start_label = 'SMA(1m)(5)'):

        # load csv
        self.df = pd.read_csv(str(csv_path))
        self.keys = self.df.keys().values

        # set params
        self.eval_year_range = [2006, 2022]
        self.y_label = 'wkFixPips'
        self.st_indicator = list(self.keys).index(indicator_start_label)
        self.is_buy = self.df['strbuysell'].values == 'close_buy'
        self.ent_years = np.array([int(self.df['dttEntTime'].values[i][:4]) for i in range(len(self.df))])
    
    def pca(self, pc_num=20):

        # set params
        is_2005 = self.ent_years == 2005
        self.pc_num = pc_num

        # get pca component
        pc = []
        for bs, bs_index in zip(['buy', 'sell'],[self.is_buy, (~self.is_buy)]):
            # target_index = bs_index
            target_index = bs_index & (~is_2005)
            X = np.array([self.df.iloc[i,self.st_indicator:].values for i in range(len(self.df)) if (target_index[i])]).astype(float)
            MN, STD = np.mean(X, axis=0), np.std(X, axis=0)
            X_norm = (X - MN)/STD
            pca = PCA(n_components=self.pc_num)
            pca.fit(X_norm)
            pc.append(np.concatenate([MN.reshape(1,-1),STD.reshape(1,-1),pca.components_],axis=0))

        # make pc dataframe
        self.df_pc= pd.DataFrame(np.array(pc).reshape(len(pc[1])*2,-1).T, 
            columns = ['buy_mean', 'buy_std'] + ['buy_pc{:02}'.format(num+1) for num in range(self.pc_num)] + \
                ['sell_mean', 'sell_std'] + ['sell_pc{:02}'.format(num+1) for num in range(self.pc_num)],
            index = self.keys[self.st_indicator:])

    def experiment(self, out_dir, th_mr = 0):

        # cross validation
        FixPips_yearsum_before, FixPips_yearsum_after, test_results, cancel_trade_num, trade_num = [], [], [], [], []
        negative_domains, drive_domains, x_fit, y_fit = [],[],[],[]
        pips_th = []
        mr_coefs = []
        for test_year in range(self.eval_year_range[0],self.eval_year_range[1]):
            print('test year: {}'.format(test_year))
            is_test = self.ent_years == test_year
            is_train = ~is_test
            # is_train = ~is_test & ~(self.ent_years == 2005)

            pc = []
            negative_domains_bs, drive_domains_bs, x_fit_bs, y_fit_bs = [],[],[],[]
            FixPips_yearsum_before_tmp2, FixPips_yearsum_after_tmp2,test_results_tmp2, cancel_trade_num_tmp2, trade_num_tmp2 = [], [], [], [], []
            pips_th_bs = []
            mr_coefs_bs = []
            for bs, bs_index in zip(['buy', 'sell'],[self.is_buy, ~self.is_buy]):

                #########
                # train #
                #########

                target_index = bs_index & is_train 

                # read pca params
                MN, STD = self.df_pc[bs+'_mean'].values, self.df_pc[bs+'_std'].values
                col_st = list(self.df_pc.keys()).index(bs+'_pc01')

                # compress dimension
                X_train = self.df.iloc[:,self.st_indicator:].values[target_index,:]
                X_train_norm = (X_train - MN)/STD
                Xd_train = np.dot(X_train_norm, self.df_pc.iloc[:,col_st:col_st + self.pc_num].values)
                y_train = self.df[self.y_label].values[target_index].reshape(-1,1)

                MR_model = LinearRegression()
                MR_model.fit(Xd_train, y_train)
                mr_coefs_bs.append(np.concatenate([MR_model.coef_[0], MR_model.intercept_]))

                ########
                # test #
                ########
                target_index = bs_index & is_test

                # read pca params
                MN, STD = self.df_pc[bs+'_mean'].values, self.df_pc[bs+'_std'].values
                col_st = list(self.df_pc.keys()).index(bs+'_pc01')

                # compress dimension
                X_test = self.df.iloc[:,self.st_indicator:].values[target_index,:]
                # X_test = np.array([df.iloc[i,self.st_indicator:].values for i in range(len(df)) if (target_index[i])]).astype(float)
                X_test_norm = (X_test - MN)/STD
                Xd_test = np.dot(X_test_norm, self.df_pc.iloc[:,col_st:col_st + self.pc_num].values)
                y_test = self.df[self.y_label].values[target_index].reshape(-1,1)

                if Xd_test.shape[0] > 0:
                    y_pred = MR_model.predict(Xd_test)
                else:
                    y_pred = np.zeros([0,1])
                rm_flag = y_pred.flatten() < th_mr
                FixPips_yearsum_before_tmp2.append(self.df[self.y_label][is_test & (~bs_index)].sum()+self.df[self.y_label][target_index].sum())
                if Xd_test.shape[0] > 0:
                    FixPips_yearsum_after_tmp2.append(self.df[self.y_label][is_test & (~bs_index)].sum()+self.df[self.y_label][target_index][~rm_flag].sum())
                else:
                    FixPips_yearsum_after_tmp2.append(self.df[self.y_label][is_test & (~bs_index)].sum()+self.df[self.y_label][target_index].sum())
                test_results_tmp2.append(self.df[self.y_label][target_index][~rm_flag].sum()-self.df[self.y_label][target_index].sum())
                cancel_trade_num_tmp2.append(sum(rm_flag))
                trade_num_tmp2.append(sum(~rm_flag))

            mr_coefs.append(mr_coefs_bs)

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

        print('{}/{} trades are cancelled'.format(cancel_trade_num.sum(),len(self.df)))

        # save pc vectors
        filename = str(out_dir / 'pca_coefs.csv')
        print('saving: {}'.format(filename))
        self.df_pc.to_csv(filename)

        # save train results
        mr_coefs_mat = mr_coefs.reshape(-1,self.pc_num+1)
        test_years = np.tile(np.arange(self.eval_year_range[0],self.eval_year_range[1]).reshape(-1,1),[1,2]).flatten()
        bs_arr = np.array([['buy','sell'] for i in range(len(mr_coefs))]).flatten()
        mr_coefs_df = pd.DataFrame(
            data = np.concatenate([test_years.reshape(-1,1),bs_arr.reshape(-1,1),mr_coefs_mat],axis=1),
            columns = ['test_year','bs']+['coef_pc{:02}'.format(i+1) for i in range(self.pc_num)]+['bias'],
        )
        filename = out_dir / 'mr_coefs.csv'
        print('saving: {}'.format(str(filename)))
        mr_coefs_df.to_csv(str(filename),index=False)

        # save test results
        both_before = FixPips_yearsum_before[:,0]
        both_after = FixPips_yearsum_after[:,0] + (FixPips_yearsum_after[:,1]-FixPips_yearsum_before[:,1])
        print('total pips is {}'.format(both_after.sum()))
        plt.close('all')
        plt.figure()
        plt.bar(np.arange(self.eval_year_range[0],self.eval_year_range[1])-0.2,both_before,label='before',alpha=0.6,width=0.4)
        plt.bar(np.arange(self.eval_year_range[0],self.eval_year_range[1])+0.2,both_after,label='after',alpha=0.6,width=0.4)
        plt.bar(self.eval_year_range[1]-0.2,both_before.mean(), label='average of before',alpha=0.6,width=0.4)
        plt.bar(self.eval_year_range[1]+0.2,both_after.mean(), label='average of after',alpha=0.6,width=0.4)
        plt.legend()
        plt.grid(axis='y')
        plt.xticks(ticks=np.arange(self.eval_year_range[0],self.eval_year_range[1]+1), labels=[str(i) for i in range(self.eval_year_range[0],self.eval_year_range[1])]+['average'], rotation=45)
        plt.title('FixPips sum before/after thresholding mr-projected values by {}\n(threshold is trained by MR to data other than test year)'.format(th_mr))
        if th_mr >= 0:
            filename = str(out_dir / f'results_th_{int(th_mr*10//10):}_{int(th_mr*10%10):}.png')
        else:
            filename = str(out_dir / f'results_th_-{int(-th_mr*10//10):}_{int(-th_mr*10%10):}.png')
        print('saving: {}'.format(filename))
        plt.savefig(filename)

        # save results as csv
        before_after = np.array([both_before, both_after]).T
        results_df = pd.DataFrame(
            data = np.concatenate([before_after, before_after.mean(axis=0).reshape(1,-1)],axis=0),
            columns = ['before', f'th={th_mr:.1f}'],
            index = np.arange(self.eval_year_range[0],self.eval_year_range[1]).tolist() + ['average']
        )

        return results_df

def FB_ver02():

    # set paths

    csv_paths = list(Path('./data/フラクタル_ブレイクアウト_PCA重回帰ver02').glob('*足*/*.csv'))
    out_parent_dir = Path('./out/20211020_FB_PCAMR_ver02_out')

    # csv_paths = list(Path('./data/RSI逆張り_PCA重回帰ver01').glob('*.csv'))
    # out_parent_dir = Path('./out/20211027_RSI_PCAMR_ver01_out')


    for csv_path in csv_paths:

        print(f'reading {csv_path}')
        net = PCA_MR(csv_path)

        print('executing PCA')
        net.pca(pc_num=20)

        print('executing MR-experiments')
        out_dir = out_parent_dir / csv_path.parent.name
        out_dir.mkdir(parents=True, exist_ok=True)
        results = []
        for th_mr in [0,0.5,1,1.5]:
            print(f'cross-validation with th={th_mr}')
            results.append(net.experiment(out_dir, th_mr=th_mr))
            print()

        print(f'saving results')
        results_sum = pd.concat([result for result in results],axis=1)
        results_sum = results_sum.loc[:,~results_sum.columns.duplicated()]
        results_sum.to_csv(out_dir/'results.csv')
        print()

def RSI_ver01():

    # set paths
    csv_paths = list(Path('./data/RSI逆張り_PCA重回帰ver01').glob('*.csv'))
    out_parent_dir = Path('./out/20211027_RSI_PCAMR_ver01_out')

    err_i = [3, 5]

    for i, csv_path in enumerate(csv_paths):

        if i in err_i:
            print(f'skip {csv_path}')

        else:
            print(f'reading {csv_path}')
            net = PCA_MR(csv_path)

            print('executing PCA')
            net.pca(pc_num=20)

            print('executing MR-experiments')
            out_dir = out_parent_dir / csv_path.name[:-4]
            out_dir.mkdir(parents=True, exist_ok=True)
            results = []
            for th_mr in [-1.5,-1,-0.5,0.5,1,1.5]:
                print(f'cross-validation with th={th_mr}')
                results.append(net.experiment(out_dir, th_mr=th_mr))
                print()

            print(f'saving results')
            results_sum = pd.concat([result for result in results],axis=1)
            results_sum = results_sum.loc[:,~results_sum.columns.duplicated()]
            results_sum.to_csv(out_dir/'results.csv')
            print()

if __name__ == '__main__':

    RSI_ver01()