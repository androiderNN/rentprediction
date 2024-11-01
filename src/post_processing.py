import os, pickle, time
import numpy as np
import pandas as pd

import config
from models import base_model

def dump_unittype_money():
    tmp_train = pickle.load(open(config.tmp_train_df, 'rb'))

    tmp = tmp_train[['money_room', 'building_id', 'building_name', 'building_type', 'room_floor', 'unit_area']]
    tmp['unittype_id'] = tmp['building_id'] + '_' + tmp['unit_area'].astype(str).str[0] # building_idと面積の最上桁でid作成

    unit_money = tmp.groupby('unittype_id')['money_room'].mean()    # 平均算出
    unit_money = unit_money.reset_index()
    unit_money.columns = ['unittype_id', 'unit_money']

    pickle.dump(unit_money, open(config.unittype_money, 'wb'))

def fill_by_unittype(pred: pd.DataFrame):
    '''
    予測済みのdfを投げると、一致するtrainデータを検索しそのmoney_roomで埋める
    pred: pd.DataFrame
        'pred', 'building_id', 'room_floor', 'unit_area'列が必要'''
    # dump_unittype_money()   # unittype_money再生成
    unittype_money = pickle.load(open(config.unittype_money, 'rb'))

    pred['unittype_id'] = pred['building_id'] + '_' + pred['unit_area'].astype(str).str[0]
    pred = pd.merge(pred, unittype_money, on='unittype_id', how='left') # 結合

    mask = pred['unit_money'].isna()
    pred.loc[mask, 'unit_money'] = pred.loc[mask, 'pred']

    pred = pred.rename(columns={'pred': 'model_pred', 'unit_money': 'pred'})
    return pred
