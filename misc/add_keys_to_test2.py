from pathlib import Path
import pandas as pd
import numpy as np

if __name__ == '__main__':
    # set paths
    dir_path = Path('./data')
    csv_path1 = dir_path / 'パラメーターテスト.txt'
    csv_path2 = dir_path / 'パラメーターテスト2.txt'

    # load csv1
    df1 = pd.read_csv(str(csv_path1))
    keys = df1.keys()

    # load csv2
    df2 = pd.read_csv(str(csv_path2),header=None)
    df3 = pd.DataFrame(df2.values,columns=keys)

    fname = str(dir_path / 'パラメーターテスト2_keys.txt')
    print('saving: {}'.format(fname))
    df3.to_csv(fname, index=False)
    



