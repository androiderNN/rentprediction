import os, pickle, sys, datetime
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, KFold
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
        tr_x, es_x, tr_y, es_y = train_test_split(tr_x, tr_y, test_size=0.2, random_state=self.rand)

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

class trainer_crossvalidation():
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

        self.modeler_array = list()

    def train(self, x, y):
        # データ分割
        train_x, va_x, train_y, va_y = train_test_split(x, y, test_size=0.2, random_state=self.rand)

        kf = KFold(n_splits=4, random_state=self.rand, shuffle=True)

        for i, (tr_idx, es_idx) in enumerate(kf.split(train_x)):
            print('\nFold', i)

            # 分割
            tr_x, es_x = train_x.iloc[tr_idx], train_x.iloc[es_idx]
            tr_y, es_y = train_y.iloc[tr_idx], train_y.iloc[es_idx]

            # 学習
            modeler = self.modeler_class(self.modeler_params)
            modeler.train(tr_x, tr_y, es_x, es_y)

            self.modeler_array.append(modeler)

            # 結果表示
            tr_pred = modeler.predict(tr_x)
            es_pred = modeler.predict(es_x)

            tr_score = self.score_fn(tr_y, tr_pred)
            es_score = self.score_fn(es_y, es_pred)

            print(f'train score: {tr_score}')
            print(f'estop score: {es_score}')
        
        # 最終結果
        train_pred = self.predict(train_x)
        va_pred = self.predict(va_x)

        train_score = self.score_fn(train_y, train_pred)
        va_score = self.score_fn(va_y, va_pred)

        print('\nmean score')
        print(f'train score: {train_score}')
        print(f'valid score: {va_score}')

    def predict(self, x):
        pred = [modeler.predict(x) for modeler in self.modeler_array]
        pred = np.array(pred).mean(axis=0)
        return pred

class rentregressor():
    def __init__(self, params):
        '''
        params = {
            'trainer_params': dict,
            'use_cv': bool,
            'verbose': bool,
            'model_type': str,
        }'''
        params['trainer_params']['score_fn'] = root_mean_squared_error

        if params['use_cv']:
            self.trainer = trainer_crossvalidation(params['trainer_params'])
        else:
            self.trainer = trainer_holdout(params['trainer_params'])
        
        self.verbose = params['verbose']
        self.model_type = params['model_type']
        self.exdir = os.path.join(config.ex_dir, time+'_'+self.model_type)

    def export(self, index, test_pred):
        df = pd.DataFrame({
            'index': index,
            'money_room': test_pred
        })
        df.to_csv(os.path.join(self.exdir, 'submission.csv'), index=False, header=False)

    def main(self):
        # データのロードと分割
        train_df = pickle.load(open(config.col_train_df, 'rb'))
        test_df = pickle.load(open(config.col_test_df, 'rb'))

        tr_x = train_df.drop(columns=config.target_name)
        tr_y = train_df[config.target_name]

        # 学習
        self.trainer.train(tr_x, tr_y)

        # 予測
        test_pred = self.trainer.predict(test_df.drop(columns='index'))

        # 出力
        if self.verbose:
            if input('出力しますか？(y/n)') == 'y':
                os.mkdir(self.exdir)
                self.export(test_df['index'], test_pred)
                print('export succeed')
