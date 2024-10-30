import os, pickle, sys, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
import config

now = datetime.datetime.now()
time = now.strftime('%m%d-%H:%M:%S')

def root_mean_squared_error(truth, pred):
    mse = mean_squared_error(truth, pred)
    return mse**0.5

class trainer_holdout():
    def __init__(self, params):
        '''
        hold-outでの学習と予測を行うクラス
        params: dict
            'rand', 'modeler_class', 'modeler_params', 'score_fn'をkeyにもつ
        '''
        self.rand = params['rand']

        self.modeler_class = params['modeler_class']
        self.modeler_params = params['modeler_params']
        self.score_fn = params['score_fn']

        self.modeler = None

    def train(self, tr_x, tr_y):
        # データ分割
        tr_x, es_x, tr_y, es_y = train_test_split(tr_x, tr_y, test_size=0.1, random_state=self.rand)

        # 学習
        self.modeler = self.modeler_class(self.modeler_params)
        self.modeler.train(tr_x, tr_y, es_x, es_y)

        # 結果表示
        tr_pred = self.modeler.predict(tr_x)
        es_pred = self.modeler.predict(es_x)

        tr_score = self.score_fn(tr_y, tr_pred)
        es_score = self.score_fn(es_y, es_pred)

        print(f'train score: {tr_score}')
        print(f'estop score: {es_score}')

    def predict(self, x):
        return self.modeler.predict(x)

class rentregressor():
    def __init__(self, params):
        params['trainer_params']['score_fn'] = root_mean_squared_error

        if params['use_cv']:
            pass
        else:
            self.trainer = trainer_holdout(params['trainer_params'])

    def export(self, test_pred):
        pass

    def main(self):
        # データのロードと分割
        train_df = pickle.load(open(config.col_train_df, 'rb'))
        test_df = pickle.load(open(config.col_test_df, 'rb'))

        tr_x = train_df.drop(columns=config.target_name)
        tr_y = train_df[config.target_name]

        # 学習
        self.trainer.train(tr_x, tr_y)

        test_pred = self.trainer.predict(test_df.drop(columns='index'))
        self.export(test_pred)
