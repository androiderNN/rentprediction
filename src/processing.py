import os, pickle
import numpy as np
import pandas as pd

import config
from features import columns

def train_clearance(df):
    df = df.loc[df['money_room']<1e+7]
    return df

def process():
    # 読み込み
    train_df = pd.read_csv(config.raw_train, usecols=lambda x: x not in config.ignore_cols)
    test_df = pd.read_csv(config.raw_test, usecols=lambda x: x not in config.ignore_cols)

    # 列毎の処理
    train_df, test_df = columns.columnprocessor(train_df, test_df)

    # 外れ値の削除
    train_df = train_clearance(train_df)

    pickle.dump(train_df, open(config.col_train_df, 'wb'))
    pickle.dump(test_df, open(config.col_test_df, 'wb'))
    print('succeed')

if __name__ == '__main__':
    process()
