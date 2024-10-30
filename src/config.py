import os

home_dir = os.path.dirname(os.getcwd())

ex_dir = os.path.join(home_dir, 'export')
src_dir = os.path.join(home_dir, 'src')

data_dir = os.path.join(home_dir, 'data')
raw_dir = os.path.join(data_dir, 'raws')
df_dir = os.path.join(data_dir, 'df')

raw_train = os.path.join(raw_dir, 'train.csv')
raw_test = os.path.join(raw_dir, 'test.csv')

tmp_train_df = os.path.join(df_dir, 'tmp_train.pkl')
tmp_test_df = os.path.join(df_dir, 'tmp_test.pkl')
df_cols = os.path.join(df_dir, 'df_cols.pkl')

target_name = 'money_room'
# ignore_cols = [
#     'building_name_ruby',   # nanのみ
#     'land_shidou_a', # 同上
#     'land_shidou_b',    # データ数1
#     'reform_exterior',  # データ数200
#     'reform_exterior_other',    # 同上
#     'reform_common_area',   # データ数600
#     'reform_place', # データ数1000
#     'reform_wet_area_other',    # データ数300
# ]
ignore_cols = \
    ['reform_exterior', 'reform_exterior_other', 'reform_common_area', 'reform_place', 'reform_wet_area_other'] + \
    ['name_ruby', 'land_shidou_a', 'building_name_ruby', 'land_shidou_b',
    'school_jun_code', 'school_ele_code', 'reform_exterior_date',
    'free_rent_gen_timing', 'traffic_car', 'reform_common_area_date',
    'money_shuuzenkikin', 'money_rimawari_now', 'reform_wet_area_date',
    'reform_date', 'money_shuuzen', 'land_mochibun_a', 'land_mochibun_b',
    'reform_interior_date', 'total_floor_area', 'snapshot_land_shidou',
    'snapshot_land_area', 'building_area'] + \
    ['lat', 'lon']  # データ数の少ない列、他に類似の特徴量が存在する列