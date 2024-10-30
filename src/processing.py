import os, pickle
import numpy as np
import pandas as pd

import config
from features import columns

def process():
    train_df = pd.read_csv(config.raw_train)
    test_df = pd.read_csv(config.raw_test)

    # 列毎の処理
    train_df, test_df = columns.columnprocessor(train_df, test_df)

    pickle.dump(train_df, open(config.col_train_df, 'wb'))
    pickle.dump(test_df, open(config.col_test_df, 'wb'))

if __name__ == '__main__':
    process()
