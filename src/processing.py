import os, pickle
import numpy as np
import pandas as pd

import config

def train_clearance(df):
    df = df.loc[df['money_room']<1e+7]
    return df

def make_tmpdf():
    '''
    一つの関数ですべて処理すると落ちるので、一時ファイルの作成のみに絞った処理を行う
    外れ値の削除と必要な列の抽出を行いtmp_dfとして保存'''
    # 読み込み
    train_df = pd.read_csv(config.raw_train, usecols=lambda x: x not in config.ignore_cols)
    test_df = pd.read_csv(config.raw_test, usecols=lambda x: x not in config.ignore_cols)

    # 外れ値の削除
    train_df = train_clearance(train_df)

    # とりあえずdtypeで抽出
    train_df = train_df.select_dtypes(include='number')
    test_df = test_df.select_dtypes(include='number')

    pickle.dump(train_df, open(config.tmp_train_df, 'wb'))
    pickle.dump(test_df, open(config.tmp_test_df, 'wb'))
    pickle.dump([c for c in train_df.columns.tolist() if c!=config.target_name], open(config.df_cols, 'wb'))
    del train_df, test_df

    print('succeed')

# def target_encoding(train, test, columns):

def process():
    train_df = pickle.load(open(config.tmp_train_df, 'rb'))
    test_df = pickle.load(open(config.tmp_test_df, 'rb'))

if __name__ == '__main__':
    make_tmpdf()    # 一次ファイルの作成
