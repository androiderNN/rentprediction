import os

home_dir = os.path.dirname(os.getcwd())

ex_dir = os.path.join(home_dir, 'export')
src_dir = os.path.join(home_dir, 'src')

data_dir = os.path.join(home_dir, 'data')
raw_dir = os.path.join(data_dir, 'raws')
df_dir = os.path.join(data_dir, 'df')

raw_train = os.path.join(raw_dir, 'train.csv')
raw_test = os.path.join(raw_dir, 'test.csv')

col_train_df = os.path.join(df_dir, 'colprocessed_train.pkl')
col_test_df = os.path.join(df_dir, 'colprocessed_test.pkl')

target_name = 'money_room'
