import os
import numpy as np
import pandas as pd

def drop_cols(df):
    drop_cols = ['lat', 'lon']
    df = df.drop(columns=drop_cols)
    return df

def columnprocessor(train, test):
    '''
    列処理の統合関数'''
    train: pd.DataFrame
    test: pd.DataFrame

    # 列の削除
    train = drop_cols(train)
    test = drop_cols(test)

    # とりあえずdtypeで抽出
    train = train.select_dtypes(include='number')
    test = test.select_dtypes(include='number')

    return train, test
