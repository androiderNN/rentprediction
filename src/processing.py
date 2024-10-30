import os, pickle
import numpy as np
import pandas as pd

import config
from features import columns

def process():
    train_df = pd.read_csv(os.path.join(config.raw_dir, 'train.csv'))
    test_df = pd.read_csv(os.path.join(config.raw_dir, 'test.csv'))

    # 列毎の処理
    train_df, test_df = columns.columnprocessor(train_df, test_df)

    pickle.dump(train_df, open(os.path.join(config.df_dir, 'colprocessed_train.pkl'), 'wb'))
    pickle.dump(test_df, open(os.path.join(config.df_dir, 'colprocessed_test.pkl'), 'wb'))

if __name__ == '__main__':
    process()
