# autotrading_analysis_20210820

## How to use

1. make `./data` directory
   ```
   $ mkdir ./data
   ```
2. put `パラメーターテスト.txt` and `パラメーターテスト2.txt` in the `data` directory
3. run `add_keys_to_test2.py` to produce `./data/パラメーターテスト2_keys.txt`
   ```
   $ python add_keys_to_test2.py
   ```
4. run other python scripts to get analysis results
   ```
   $ python pc_vs_gl_w.py
   ```
5. see `./out/pc_vs_gl_w....png` to define thresholding values for each component
7. use `./principal_components_100dim_to_6val.csv` to calculate each component value as follow;
   ```py
   import numpy as np

   # X: matrics (sample_num x indicator_num)
   # pc01_vec: 1st principal components vector (such as `buy_pc01` in the csv file)

   # normalize indicator vector
   X_norm = X.copy()
   for i in indicator_num:
      X_norm[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])

   # calc projection
   pc01_projected_val = np.zeros(sample_num)
   pc01_projected_val[j] = np.dot(X[j,:], pc01_vec)
   ```