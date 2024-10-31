import os, pickle, datetime, math
import numpy as np
import pandas as pd

import config

def train_clearance(df):
    df = df.loc[df['money_room']<1e+7]
    # df = df.loc[df['money_room']<1.5e+6]
    return df

def make_tmpdf():
    '''
    一つの関数ですべて処理すると落ちるので、一時ファイルの作成のみに絞った処理を行う
    外れ値の削除と必要な列の抽出を行いtmp_dfとして保存'''
    # 読み込み
    train_df = pd.read_csv(config.raw_train, usecols=lambda x: x not in config.ignore_cols, dtype=config.df_dtype)
    test_df = pd.read_csv(config.raw_test, usecols=lambda x: x not in config.ignore_cols, dtype=config.df_dtype)

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

def replace_values(df:pd.DataFrame) -> None:
    # floor_count
    col = 'floor_count'
    df.loc[(df[col]>65)|(df[col]==0), col] = None  # 異常値の削除
    dic = {i+j: i for i in range(16, 65, 5) for j in range(5)}
    df[col] = df[col].replace(dic)

    # building_structure
    col = 'building_structure'
    dic = {2:None, 7:None, 12:None, 13:None}    # データ数の少ないクラスをnanに
    df[col] = df[col].replace(dic).astype(float)

    # year_built
    col = 'year_built'
    df.loc[df[col]<1900, col] = None
    df.loc[(df[col]<1950)&(df['floor_count']>=3), col] = None
    # yb = np.array([s[:4]+'-'+s[4:6] if len(s)==8 else '' for s in df['year_built'].astype(str)], dtype=np.datetime64)   # length=3ならNaN、8なら年月
    # bcd = np.array(df['building_create_date'], dtype=np.datetime64)
    # diff = bcd - yb # 築年月とデータ登録日時の差
    # df.loc[diff<0, 'year_built'] = None # 築年月がデータ登録以前の欄はnanに

def create_new_cols(train, test) -> None:
    for df in [train, test]:
        # old_months
        yb = np.array([s[:4]+'-'+s[4:6] if len(s)==8 else '' for s in df['year_built'].astype(str)], dtype=np.datetime64)   # length=3ならNaN、8なら年月
        bmd = np.array(df['building_modify_date'], dtype=np.datetime64)
        old_year = (bmd-yb).astype('timedelta64[Y]')
        df['old_year'] = pd.Series(old_year, dtype=float)

        # post
        df['post'] = (df['post1']*10000 + df['post2'])

        # floor_plan_code
        df['num_rooms'] = (df['floor_plan_code']//100).astype(np.float16)   # 部屋数
        df['room_type'] = (df['floor_plan_code']%100).astype(np.float16)    # 部屋種別

    # tag_ids
    tag_dic = {
        '223101': 'tag_ind_washbasin',
        '310101': 'tag_autolock',
        '230401': 'tag_kitchen_system',
        '210201': 'tag_gas_pipe',
        '210202': 'tag_gas_propane',
        '320101': 'tag_elevator',
        '220301': 'tag_bath_separate',
        '220401': 'tag_bath_reheat',    #以上feature importance上位

        '310501': 'tag_interphone',
        '321101': 'tag_deliverybox',    # 中程度
        
        '293101': 'tag_furnished',
        '260501': 'tag_internet',
        '340401': 'tag_room_corner',
        '321001': 'tag_park_bike',
    }

    tag_ids = pickle.load(open(config.tag_ids_train, 'rb'))
    tmp = tag_ids[tag_dic.keys()]
    train[list(tag_dic.values())] = tmp
    
    tag_ids = pickle.load(open(config.tag_ids_test, 'rb'))
    tmp = tag_ids[tag_dic.keys()]
    test[list(tag_dic.values())] = tmp
    
    return train, test

def drop_cols(df:pd.DataFrame):
    cols = [
        'year_built',   # old_yearで築年数を表したため不要と思われる
        'post1', 'post2',   # 結合してpostとした
        'floor_plan_code',  # num_rooms, room_typeに分割
    ]
    return df.drop(columns=cols)

def process():
    train_df = pickle.load(open(config.tmp_train_df, 'rb'))
    test_df = pickle.load(open(config.tmp_test_df, 'rb'))

    te_cols = ['madori_kind_all']
    train_df, test_df = target_encoding(train_df, test_df, te_cols)

    # 置換
    replace_values(train_df)
    replace_values(test_df)

    # 特徴量作成
    train_df, test_df = create_new_cols(train_df, test_df)

    # 列削除
    train_df = drop_cols(train_df)
    test_df = drop_cols(test_df)

    # dtypeで抽出
    train_df = train_df.select_dtypes(include='number')
    test_df = test_df.select_dtypes(include='number')

    pickle.dump(train_df, open(config.train_df, 'wb'))
    pickle.dump(test_df, open(config.test_df, 'wb'))
    pickle.dump([c for c in train_df.columns.tolist() if c!=config.target_name], open(config.df_cols, 'wb'))
    print('process succeed')

if __name__ == '__main__':
    # make_tmpdf()    # 一次ファイルの作成
    process()   # 特徴量作成
