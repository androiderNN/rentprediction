import os
import numpy as np

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
train_df = os.path.join(df_dir, 'train_df.pkl')
test_df = os.path.join(df_dir, 'test_df.pkl')
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
    ['money_hoshou_company'] + \
    ['nl', 'el']  # データ数の少ない列、他に類似の特徴量が存在する列

df_dtype = {
    'target_ym': np.int32,
    'money_room': np.int32,
    'building_id': object,
    'building_status': np.int32,
    'building_create_date': object,
    'building_modify_date': object,
    'building_type': np.int32,
    'building_name': object,
    'homes_building_name': object,
    'homes_building_name_ruby': object,
    'unit_count': np.float32,
    'full_address': object,
    'lon': np.float32,
    'lat': np.float32,
    'building_structure': np.float32,
    'floor_count': np.float32,
    'basement_floor_count': np.float32,
    'year_built': np.float32,
    'building_land_area': np.float32,
    'land_area_all': np.float32,
    'unit_area_min': np.float32,
    'unit_area_max': np.float32,
    'building_land_chimoku': np.float32,
    'land_youto': np.float32,
    'land_toshi': np.float32,
    'land_chisei': np.float32,
    'land_area_kind': np.float32,
    'land_setback_flg': np.float32,
    'land_setback': np.float32,
    'land_kenpei': np.float32,
    'land_youseki': np.float32,
    'land_road_cond': np.float32,
    'land_seigen': object,
    'building_area_kind': np.float32,
    'management_form': np.float32,
    'management_association_flg': np.float32,
    'building_tag_id': object,
    'unit_id': np.int32,
    'unit_name': object,
    'room_floor': np.float32,
    'balcony_area': np.float32,
    'dwelling_unit_window_angle': np.float32,
    'room_count': np.float32,
    'unit_area': np.float32,
    'floor_plan_code': np.float32,
    'reform_place_other': object,
    'reform_wet_area': object,
    'reform_interior': object,
    'reform_interior_other': object,
    'reform_etc': object,
    'renovation_date': object,
    'renovation_etc': object,
    'unit_tag_id': object,
    'bukken_id': np.int32,
    'snapshot_create_date': object,
    'new_date': object,
    'snapshot_modify_date': object,
    'timelimit_date': object,
    'flg_open': np.int32,
    'flg_own': np.int32,
    'bukken_type': np.int32,
    'flg_investment': np.float32,
    'empty_number': np.float32,
    'empty_contents': object,
    'post1': np.float32,
    'post2': np.float32,
    'addr1_1': np.int32,
    'addr1_2': np.int32,
    'addr2_name': object,
    'addr3_name': object,
    'addr4_name': object,
    'rosen_name1': object,
    'eki_name1': object,
    'bus_stop1': object,
    'bus_time1': np.float32,
    'walk_distance1': np.float32,
    'rosen_name2': object,
    'eki_name2': object,
    'bus_stop2': object,
    'bus_time2': np.float32,
    'walk_distance2': np.float32,
    'traffic_other': object,
    'house_area': np.float32,
    'flg_new': np.int32,
    'house_kanrinin': np.float32,
    'room_kaisuu': np.float32,
    'snapshot_window_angle': np.float32,
    'madori_number_all': np.int32,
    'madori_kind_all': np.int32,
    'money_kyoueki': np.float32,
    'money_kyoueki_tax': np.float32,
    'money_sonota_str1': object,
    'money_sonota1': np.float32,
    'money_sonota_str2': object,
    'money_sonota2': np.float32,
    'money_sonota_str3': object,
    'money_sonota3': np.float32,
    'parking_money': np.float32,
    'parking_money_tax': np.float32,
    'parking_kubun': np.float32,
    'parking_distance': np.float32,
    'parking_number': np.float32,
    'parking_memo': object,
    'genkyo_code': np.float32,
    'usable_status': np.int32,
    'usable_date': np.float32,
    'school_ele_name': object,
    'school_ele_distance': np.float32,
    'school_jun_name': object,
    'school_jun_distance': np.float32,
    'convenience_distance': np.float32,
    'super_distance': np.float32,
    'hospital_distance': np.float32,
    'park_distance': np.float32,
    'drugstore_distance': np.float32,
    'bank_distance': np.float32,
    'shopping_street_distance': np.float32,
    'est_other_name': object,
    'est_other_distance': np.float32,
    'statuses': object,
    'parking_keiyaku': np.float32,
    'money_hoshou_company': object,
    'free_rent_duration': np.float32
}