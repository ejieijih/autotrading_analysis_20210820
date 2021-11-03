

is_2017 = ent_years == 2008

is_train = is_2017

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

    y_pred = MR_model.predict(Xd_train)

    print(y_train[y_pred>0].sum())

