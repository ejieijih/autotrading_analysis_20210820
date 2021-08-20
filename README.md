# autotrading_analysis_20210820

analysis of autotrading for hikikoumori

## How to use

1. make `./data` directory
   ```
   $ mkdir ./data
   ```
2. put `パラメーターテスト4.txt` in the `data` directory
3. run `./pc_vs_gl_w.py` to get analysis results
   ```
   $ python pc_vs_gl_w.py
   ```
4. see `./out/pc_vs_gl_w_*.png` to define thresholding values for each component
5. use principal component vectors in `./principal_components_100dim_to_6val.csv` to calculate each component value as follow;
   ```py
   import numpy as np

   # X: matrics (sample_num x indicator_num)
   # pc01_vec: 1st principal components vector (such as `buy_pc01` in the csv file)

   # normalize indicator vector
   X_norm = X.copy()
   for i in indicator_num:
      X_norm[:,i] = (X[:,i]-np.mean(X[:,i]))/np.std(X[:,i])

   # project indicator values to 1st-pc-vector
   pc01_projected_val = np.dot(X, pc01_vec)
   ```
normalizeの参考リンク [[1]]

[1]: https://qiita.com/maskot1977/items/082557fcda78c4cdb41f#%E8%A1%8C%E5%88%97%E3%81%AE%E6%A8%99%E6%BA%96%E5%8C%96