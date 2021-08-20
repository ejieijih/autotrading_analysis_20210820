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
4. run other python codes to get analysis results
   ```
   $ python osc_vs_gl.py
   $ python pca_100dim_to_6pc.py
   ```