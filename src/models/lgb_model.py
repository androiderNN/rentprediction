import os, pickle, sys
import numpy as np
import pandas as pd
import lightgbm as lgb

import base_model
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

class modeler_lgb():
    def __init__(self, params):
        '''
        params: dict
            lgbのパラメータを'model_params'として持つ'''
        self.num_boost_round = params['num_boost_round']
        self.model_params = params['model_params']
    
    def train(self, tr_x, tr_y, es_x, es_y):
        tr_lgb = lgb.Dataset(tr_x, tr_y)
        es_lgb = lgb.Dataset(es_x, es_y)
        
        self.model = lgb.train(
            params=self.model_params,
            train_set=tr_lgb,
            num_boost_round=self.num_boost_round,
            valid_sets=es_lgb,
            valid_names='estop',
            callbacks=[lgb.early_stopping(stopping_rounds=2, verbose=True)]
        )

    def predict(self, x):
        return self.model.predict(x)

def lgb_supplements(modeler):
    '''
    feature importanceのdf作成'''
    cols = pickle.load(open(config.df_cols, 'rb')) # 列名
    fi = modeler.model.feature_importance(importance_type='gain')   # 重要度

    fi = pd.DataFrame({'column': cols, 'importance': fi})
    return {'feature_importance': fi}

if __name__ == '__main__':
    rand = 0

    modeler_params = {
        'num_boost_round': 1000,
        'model_params':{
            'object': 'regression',
            'metric': 'rmse',
            'verbose': -1,
            'learning_rate': 0.2,
            'feature_fraction': 0.5,
            'bagging_fraction': 0.5,
            'bagging_freq': 1,
            # 'max_depth': 5,
            # 'min_child_leaves': 10000,
        }
    }

    trainer_params = {
        'rand': rand,
        'modeler_class': modeler_lgb,
        'modeler_params': modeler_params,
        'score_fn': None,
        'supplements_fn': lgb_supplements
    }

    params = {
        'trainer_params': trainer_params,
        'use_cv': True,
        'use_log': True,
        'verbose': True,
        'model_type': 'lgb',
    }

    print(params)

    rr = base_model.rentregressor(params)
    rr.main()
