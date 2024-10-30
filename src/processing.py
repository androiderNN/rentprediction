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

    # 異常値の削除
    train_df = train_clearance(train_df)

    pickle.dump(train_df, open(config.tmp_train_df, 'wb'))
    pickle.dump(test_df, open(config.tmp_test_df, 'wb'))
    del train_df, test_df

    print('tmpfile created')

def target_encoding(train:pd.DataFrame, test, columns):
    train = train.reset_index(drop=True)
    test = test.reset_index(drop=True)

    dic = {col: train.groupby(col)[config.target_name].mean() for col in columns}   # 平均値のマップ
    
    tmp = {col+'_TE': train[col].map(dic[col]) for col in columns}
    train = train.drop(columns=columns)
    train = train.join(pd.DataFrame(tmp))

    tmp = {col+'_TE': test[col].map(dic[col]) for col in columns}
    test = test.drop(columns=columns)
    test = test.join(pd.DataFrame(tmp))

    return train, test

def process():
    train_df = pickle.load(open(config.tmp_train_df, 'rb'))
    test_df = pickle.load(open(config.tmp_test_df, 'rb'))

    te_cols = ['madori_kind_all']
    train_df, test_df = target_encoding(train_df, test_df, te_cols)

    # dtypeで抽出
    train_df = train_df.select_dtypes(include='number')
    test_df = test_df.select_dtypes(include='number')

    pickle.dump(train_df, open(config.train_df, 'wb'))
    pickle.dump(test_df, open(config.test_df, 'wb'))
    pickle.dump([c for c in train_df.columns.tolist() if c!=config.target_name], open(config.df_cols, 'wb'))
    print('process succeed')

if __name__ == '__main__':
    make_tmpdf()    # 一次ファイルの作成
    process()   # 特徴量作成
