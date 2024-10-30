import os
import numpy as np
import pandas as pd

def col_replace(df):
    df = df.drop(columns=['building_name_ruby'])
    return df

def columnprocessor(train, test):
    '''
    列処理の統合関数'''
    train: pd.DataFrame
    test: pd.DataFrame

    # 列の単純な置き換え
    train = col_replace(train)
    test = col_replace(test)

    # とりあえずdtypeで抽出
    train = train.select_dtypes(include='number')
    test = test.select_dtypes(include='number')

    return train, test
