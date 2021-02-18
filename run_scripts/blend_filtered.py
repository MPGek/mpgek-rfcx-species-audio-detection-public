import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def get_gf_name_lb_value(data_file_name: str, collect_gf_type=False):
    gf_types = ['gf_best_val_Lwlrap_', 'gf_best_val_f2_', 'gf_best_val_loss_',
                'gf_best_e_id_0_', 'gf_best_e_id_1_', 'gf_best_e_id_2_', 'gf_best_e_id_3_']

    found_gf_type = ''

    data_file_short_name = data_file_name
    for gf_type in gf_types:
        if gf_type in data_file_short_name:
            found_gf_type = gf_type.replace('gf_best_', '')[:-1]
        data_file_short_name = data_file_short_name.replace(gf_type, 'gf_')

    if 'gf_' in data_file_short_name:
        gf_name = data_file_short_name.split('gf_')[1].split('_')[0]
        if collect_gf_type:
            gf_name = gf_name + '_' + found_gf_type
    else:
        gf_name = 'blend'

    lb_value_split = data_file_short_name.split('_LB')
    lb_value = lb_value_split[1].replace('.csv', '') if len(lb_value_split) > 1 else 'xxx'

    return gf_name, lb_value


def merge_files(run_path: Path, data_file):
    new_name = ''
    lb_values = '_LB'
    for idx, name in enumerate(data_file):
        gf_name, lb_value = get_gf_name_lb_value(name, idx > 0)
        new_name += '_' + gf_name
        lb_values += '_' + lb_value

    base_data = pd.read_csv(run_path / data_file[0])
    extra_data = [pd.read_csv(run_path / file_name) for file_name in data_file[1:]]

    for extra_data_item in extra_data:  # type:pd.DataFrame
        for column in extra_data_item.columns:  # type: str
            if column.startswith('s'):
                base_data[column] = extra_data_item[column]

    return base_data, new_name, lb_values


def main():
    is_debug_mode = sys.gettrace() is not None

    # set env variable for data
    os.environ['DATA_FOLDER'] = '../data'

    # ===== Init folders and models list

    # LB: 0.907
    data_files = [
        # ['20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv'],
        ['20210128_best/sub_gf_best_val_loss_14299_210128_0223_MEAN_loss_0.1359_Lwlrap_0.8768_f2_0.7481_LB0.894.csv',
         '20210208_best/dS/sub_gf_best_val_loss_18664_210208_1715_MEAN_loss_0.0988_Lwlrap_0.9530_f2_0.8384.csv'],
    ]

    # LB: 0.901
    data_files = [
        ['20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv'],
    ]

    # LB: 0.904
    data_files = [
        ['20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
         '20210208_best/dS/sub_gf_best_val_loss_18664_210208_1715_MEAN_loss_0.0988_Lwlrap_0.9530_f2_0.8384.csv'],
    ]

    # LB: 0.886
    data_files = [
        ['20210208_best/dS/sub_gf_best_val_loss_187bc_210208_2215_MEAN_loss_0.1259_Lwlrap_0.9464_f2_0.8393.csv',
         '20210208_best/dS/sub_gf_best_val_loss_18664_210208_1715_MEAN_loss_0.0988_Lwlrap_0.9530_f2_0.8384.csv'],
    ]

    is_notebook = platform.node() == 'nb-162'
    prefix = ''
    if is_notebook:
        prefix = '../runs_server/'

    run_path = Path(os.environ['DATA_FOLDER']) / 'runs' / prefix

    new_name = ''
    lb_values = '_LB'
    data = []  # type:List[pd.DataFrame]
    for data_file in data_files:
        if isinstance(data_file, list):
            file_data, gf_name, lb_value = merge_files(run_path, data_file)
            data.append(file_data)
        else:
            data.append(pd.read_csv(run_path / data_file))
            gf_name, lb_value = get_gf_name_lb_value(data_file[0])

        new_name += '_' + gf_name
        lb_values += '_' + lb_value

    time_prefix = datetime.now().strftime('blend_%y%m%d_%H%M%S')
    new_name = time_prefix + new_name + lb_values + '.csv'
    # new_name = '20210205_best_X.csv'

    mean_data = pd.concat(data).groupby('recording_id').mean()  # type:pd.DataFrame
    mean_data.to_csv(run_path / new_name, index_label='recording_id')


if __name__ == '__main__':
    main()
