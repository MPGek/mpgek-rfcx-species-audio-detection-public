import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List

import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))


def main():
    is_debug_mode = sys.gettrace() is not None

    # set env variable for data
    os.environ['DATA_FOLDER'] = '../data'

    # ===== Init folders and models list

    # LB: 0.870
    data_files = [
        # '20201220_best/sub_gf_05d97_201218_1230_mean_score_0.816678_LB0.839.csv',
        # '20201220_best/sub_gf_06a93_201220_2349_mean_score_0.809369_LB0.835.csv',
        # '20201220_best/sub_gf_05d96_201218_1230_mean_score_0.819896_LB0.830.csv',
        '20201221_best/sub_gf_07003_201221_1151_mean_score_0.798729_LB0.848.csv',
        '20201223_best/sub_gf_07b94_201223_1109_mean_score_0.825592_LB0.844.csv',
        '20201224_best/sub_gf_07f3a_201224_1111_mean_score_0.887582_LB0.847.csv',
        '20201224_best/sub_gf_07f3c_201224_1111_mean_score_0.884792_LB0.843.csv',
        '20201225_best/sub_gf_083c5_201225_2259_mean_score_0.899647_LB0.845.csv',
        # '20201225_best/sub_gf_084a9_201225_2326_mean_score_0.893309_LB0.831.csv',
        '20201225_best/sub_gf_086dc_201225_2328_mean_score_0.887892_LB0.854.csv',
        # '20201228_best/sub_gf_091a3_201228_0101_mean_score_0.891370_LB0.838.csv',
        '20201228_best/sub_gf_091a4_201228_0101_mean_score_0.881926_LB0.840.csv',
        '20201228_best/sub_gf_09a2e_201228_2333_mean_score_0.895147_LB0.849.csv',
        '20201229_best/sub_gf_09d36_201229_1003_mean_score_0.888785_LB0.850.csv',
    ]

    # LB: 0.870
    data_files = [
        # '20201221_best/sub_gf_07003_201221_1151_mean_score_0.798729_LB0.848.csv',
        # '20201223_best/sub_gf_07b94_201223_1109_mean_score_0.825592_LB0.844.csv',
        # '20201224_best/sub_gf_07f3a_201224_1111_mean_score_0.887582_LB0.847.csv',
        # '20201224_best/sub_gf_07f3c_201224_1111_mean_score_0.884792_LB0.843.csv',
        '20201225_best/sub_gf_083c5_201225_2259_mean_score_0.899647_LB0.845.csv',
        '20201225_best/sub_gf_086dc_201225_2328_mean_score_0.887892_LB0.854.csv',
        '20201228_best/sub_gf_091a4_201228_0101_mean_score_0.881926_LB0.840.csv',
        '20201228_best/sub_gf_09a2e_201228_2333_mean_score_0.895147_LB0.849.csv',
        '20201229_best/sub_gf_09d36_201229_1003_mean_score_0.888785_LB0.850.csv',
        # '20201229_best/',
    ]

    # LB: 0.869
    data_files = [
        '20201225_best/sub_gf_083c5_201225_2259_mean_score_0.899647_LB0.845.csv',
        '20201225_best/sub_gf_086dc_201225_2328_mean_score_0.887892_LB0.854.csv',
        '20201228_best/sub_gf_091a4_201228_0101_mean_score_0.881926_LB0.840.csv',
        '20201228_best/sub_gf_09a2e_201228_2333_mean_score_0.895147_LB0.849.csv',
        '20201229_best/sub_gf_09d36_201229_1003_mean_score_0.888785_LB0.850.csv',
        '20201229_best/sub_gf_09f68_201229_2357_mean_score_0.895878_LB0.855.csv',
    ]

    # LB: 0.878
    data_files = [
        '20201225_best/sub_gf_086dc_201225_2328_mean_score_0.887892_LB0.854.csv',
        '20201229_best/sub_gf_09f68_201229_2357_mean_score_0.895878_LB0.855.csv',
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
    ]

    # LB: 0.881
    data_files = [
        '20201225_best/sub_gf_086dc_201225_2328_mean_score_0.887892_LB0.854.csv',
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
    ]

    # LB: 0.888
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0ab9c_210101_0524_mean_score_0.888258_LB0.871.csv',
    ]

    # LB: 0.893
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0ab9c_210101_0524_mean_score_0.888258_LB0.871.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
    ]

    # LB: 0.895
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
    ]

    # LB: 0.896
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b3b3_210102_0929_mean_score_0.897286_LB0.879.csv',
        '20210102_best/sub_gf_0b3b4_210102_0958_mean_score_0.900911_LB0.879.csv',
        '20210102_best/sub_gf_0b61a_210102_1952_mean_score_0.894345_LB0.880.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
    ]

    # LB: 0.892
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
    ]

    # LB: 0.894
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
    ]

    # LB: 0.898
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61a_210102_1952_mean_score_0.894345_LB0.880.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
    ]

    # LB: 0.898
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
    ]

    # LB: 0.897-0.893
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
        '20210104_best/sub_gf_0be8e_210104_0536_mean_score_0.901053_LB0.873.csv',  # 0.897
        '20210104_best/sub_gf_0c254_210104_2018_mean_score_0.895165_LB0.858.csv',  # 0.893
    ]

    # LB: 0.898
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
        '20210105_best/sub_gf_0c3f0_210105_0459_mean_score_0.894801_LB0.880.csv',  # 0.898
    ]

    # LB: 0.897
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        # '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
        '20210105_best/sub_gf_0c3f0_210105_0459_mean_score_0.894801_LB0.880.csv',
    ]

    # LB: 0.897
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210101_best/sub_gf_0b006_210101_2324_mean_score_0.878792_LB0.881.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        # '20210103_best/sub_gf_0b8f0_210103_1016_mean_score_0.888087_LB0.872.csv',
        '20210105_best/sub_gf_0c3f0_210105_0459_mean_score_0.894801_LB0.880.csv',  # 0.898
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        '20210106_best/sub_gf_best_val_Lwlrap_0ce23_210106_2339_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.879.csv',
        '20210107_best/sub_gf_best_val_f2_0cfef_210107_1101_MEAN_Lwlrap_0.8931_f2_0.8012_LB0.884.csv'
    ]

    # LB: 0.900
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_f2_0d35f_210108_0419_MEAN_Lwlrap_0.8994_f2_0.8059_LB0.885.csv',
        # '20210107_best/sub_gf_best_val_f2_0d355_210108_0328_MEAN_Lwlrap_0.8883_f2_0.7951_LB0.885.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d355_210108_0228_MEAN_Lwlrap_0.9026_f2_0.7813_LB0.889.csv',
        '20210108_best/sub_gf_best_val_Lwlrap_0d89e_210109_0036_MEAN_Lwlrap_0.9024_f2_0.8197_LB0.885.csv',  # 0.900
    ]

    # LB: 0.901
    data_files = [
        '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_f2_0d35f_210108_0419_MEAN_Lwlrap_0.8994_f2_0.8059_LB0.885.csv',
        # '20210107_best/sub_gf_best_val_f2_0d355_210108_0328_MEAN_Lwlrap_0.8883_f2_0.7951_LB0.885.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d355_210108_0228_MEAN_Lwlrap_0.9026_f2_0.7813_LB0.889.csv',
    ]

    # LB: 0.901
    data_files = [
        # '20201230_best/sub_gf_0a666_201230_2252_mean_score_0.896035_LB0.886.csv',
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_f2_0d35f_210108_0419_MEAN_Lwlrap_0.8994_f2_0.8059_LB0.885.csv',
        # '20210107_best/sub_gf_best_val_f2_0d355_210108_0328_MEAN_Lwlrap_0.8883_f2_0.7951_LB0.885.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d355_210108_0228_MEAN_Lwlrap_0.9026_f2_0.7813_LB0.889.csv',
    ]

    # LB: 0.904
    data_files = [
        '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d355_210108_0228_MEAN_Lwlrap_0.9026_f2_0.7813_LB0.889.csv',
        '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210111_best/sub_gf_best_val_Lwlrap_0e60f_210111_1458_MEAN_Lwlrap_0.8898_f2_0.8025_LB0.887.csv',
        '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
    ]

    # LB: 0.910
    data_files = [
        # '20210102_best/sub_gf_0b61b_210102_1925_mean_score_0.899791_LB0.889.csv',
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d355_210108_0228_MEAN_Lwlrap_0.9026_f2_0.7813_LB0.889.csv',
        '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
    ]

    # LB: 0.912
    data_files = [
        '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv',  # 0.912
        '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',  # 0.912
    ]

    # LB: 0.925
    data_files = [
        # '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        # '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv',  # 0.912
        # '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',  # 0.912
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]

    # LB: 0.925
    data_files = [
        # '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        # '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv',  # 0.912
        # '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',  # 0.912
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]

    # LB: 0.926
    data_files = [
        # '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        # '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        # '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv',  # 0.912
        # '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',  # 0.912
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]

    # LB: 0.921
    data_files = [
        'blend_210118_231315_best_0ef12_best_L-B_0.904_0.901_0.892_LB0.925.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]

    # LB: 0.926
    data_files = [
        # '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        # '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv', # 0.925
        # '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv', # 0.924
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        # '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv', # 0.920
        # '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]

    # LB: 0.902
    data_files = [
        # '20210106_best/sub_gf_best_val_f2_0cde7_210107_0022_MEAN_loss_0.0520_Lwlrap_0.9013_f2_0.7923_LB0.890.csv',
        # '20210106_best/sub_gf_best_val_f2_0ce23_210107_0026_MEAN_loss_0.0535_Lwlrap_0.8973_f2_0.7916_LB0.890.csv',
        # '20210107_best/sub_gf_best_val_Lwlrap_0d35f_210108_0316_MEAN_Lwlrap_0.9071_f2_0.7962_LB0.890.csv',
        # '20210111_best/sub_gf_best_val_f2_0e60f_210111_1620_MEAN_Lwlrap_0.8792_f2_0.8147_LB0.897.csv',
        # '20210112_best/sub_gf_best_val_f2_0ea18_210112_1011_MEAN_Lwlrap_0.8873_f2_0.8279_LB0.894.csv',
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv', # 0.925
        # '20210113_best/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv', # 0.924
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        # '20210113_best/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv', # 0.920
        # '20210114_best/sub_gf_best_e_id_0_0f392_210113_1629_MEAN_Lwlrap_0.8754_f2_0.8100_LB0.888.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]

    # LB: 0.922
    data_files = [
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv', # 0.925
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]

    # LB: 0.0.922
    data_files = [
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv', # 0.925
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        # '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]

    # LB: 0.928
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv', # 0.925
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]

    # LB: 0.922
    data_files = [
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210128_best/sub_gf_best_val_loss_14299_210128_0223_MEAN_loss_0.1359_Lwlrap_0.8768_f2_0.7481_LB0.894.csv',
    ]

    # LB: 0.907
    data_files = [
        '20210128_best/sub_gf_best_val_loss_14299_210128_0223_MEAN_loss_0.1359_Lwlrap_0.8768_f2_0.7481_LB0.894.csv',
        '20210131_best/sub_gf_best_val_loss_156ca_210201_0034_MEAN_loss_0.1697_Lwlrap_0.8951_f2_0.2456_LB0.887.csv',
    ]

    # LB: 0.883
    data_files = [
        '20210203_best/sub_gf_best_val_loss_1650b_210203_0819_MEAN_loss_0.1792_Lwlrap_0.8754_f2_0.2234_LB0.888.csv',
        '20210203_best/sub_gf_best_val_loss_16777_210203_1617_MEAN_loss_0.1763_Lwlrap_0.8884_f2_0.2266_LB0.869.csv',
    ]

    # LB: 0.911
    data_files = [
        '20210203_best/sub_gf_best_val_loss_1650b_210203_0819_MEAN_loss_0.1792_Lwlrap_0.8754_f2_0.2234_LB0.888.csv',
        '20210203_best/sub_gf_best_val_loss_164b9_210202_2336_MEAN_loss_0.1426_Lwlrap_0.8753_f2_0.7487_LB0.892.csv',
    ]

    # LB: 0.911
    data_files = [
        '20210205_best_X/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',
        '20210205_best_X/sub_gf_best_e_id_0_15c6a_210201_1630_MEAN_loss_0.2074_Lwlrap_0.9155_f2_0.2291_LB0.885.csv',
        '20210205_best_X/sub_gf_best_val_f2_0ef12_210113_0602_MEAN_Lwlrap_0.8694_f2_0.8134_LB0.895.csv',
        '20210205_best_X/sub_gf_best_val_f2_0ef13_210113_0131_MEAN_Lwlrap_0.8868_f2_0.8225_LB0.898.csv',
        '20210205_best_X/sub_gf_best_val_loss_15c6a_210201_1531_MEAN_loss_0.1723_Lwlrap_0.9000_f2_0.2526_LB0.881.csv',
        '20210205_best_X/sub_gf_best_val_loss_16a32_210203_2104_MEAN_loss_0.1432_Lwlrap_0.8767_f2_0.7344_LB0.883.csv',
        '20210205_best_X/sub_gf_best_val_loss_16b7f_210204_0249_MEAN_loss_0.1484_Lwlrap_0.8645_f2_0.7270_LB0.886.csv',
        '20210205_best_X/sub_gf_best_val_loss_16d12_210204_1334_MEAN_loss_0.1484_Lwlrap_0.8690_f2_0.7193_LB0.885.csv',
        '20210205_best_X/sub_gf_best_val_loss_16fe1_210204_2204_MEAN_loss_0.1456_Lwlrap_0.8723_f2_0.7146_LB0.882.csv',
        '20210205_best_X/sub_gf_best_val_loss_150a6_210130_1049_MEAN_loss_0.1627_Lwlrap_0.8767_f2_0.7152_LB0.882.csv',
        '20210205_best_X/sub_gf_best_val_loss_156ca_210201_0034_MEAN_loss_0.1697_Lwlrap_0.8951_f2_0.2456_LB0.887.csv',
        '20210205_best_X/sub_gf_best_val_loss_161cb_210202_1116_MEAN_loss_0.1960_Lwlrap_0.9095_f2_0.4187_LB0.892.csv',
        '20210205_best_X/sub_gf_best_val_loss_164b9_210202_2336_MEAN_loss_0.1426_Lwlrap_0.8753_f2_0.7487_LB0.892.csv',
        '20210205_best_X/sub_gf_best_val_loss_1650b_210203_0819_MEAN_loss_0.1792_Lwlrap_0.8754_f2_0.2234_LB0.888.csv',
        '20210205_best_X/sub_gf_best_val_loss_14298_210128_0401_MEAN_loss_0.1250_Lwlrap_0.8742_f2_0.7118_LB0.887.csv',
        '20210205_best_X/sub_gf_best_val_loss_14299_210128_0223_MEAN_loss_0.1359_Lwlrap_0.8768_f2_0.7481_LB0.894.csv',
        '20210205_best_X/sub_gf_best_val_loss_14883_210129_1953_MEAN_loss_0.1405_Lwlrap_0.8807_f2_0.7339_LB0.894.csv',
        '20210205_best_X/sub_gf_best_val_loss_17253_210205_1241_MEAN_loss_0.1539_Lwlrap_0.8821_f2_0.7192_LB0.882.csv',
        '20210205_best_X/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        '20210205_best_X/sub_gf_best_val_Lwlrap_16b7f_210204_0324_MEAN_loss_0.1725_Lwlrap_0.8938_f2_0.7556_LB0.881.csv',
        '20210205_best_X/sub_gf_best_val_Lwlrap_16fe1_210204_2233_MEAN_loss_0.1803_Lwlrap_0.8959_f2_0.7599_LB0.880.csv',
    ]

    # LB: 0.926
    data_files = [
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # 0.925
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        # '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.929
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # 0.925
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.925
    data_files = [
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # w/wo - 0.929
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        # '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.925
    data_files = [
        # '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # w/wo - 0.929
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.917
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # w/wo - 0.929
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        # '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        # '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.926
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # 0.925
        # '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
        '20210211_best/sub_gf_best_val_bceTP_194b7_210211_0923_MEAN_loss_0.1093_Lwlrap_0.9028_f2_0.3640_LB0.887.csv'
    ]

    # LB: 0.882
    data_files = [
        '20210215_best/sub_gf_best_val_bceTP_1a5a7_210215_1802_MEAN_loss_0.3939_Lwlrap_0.8853_bceTP_0.1719_LB0.876.csv',
        '20210215_best/sub2_gf_best_val_bceTP_1a5a7_210215_1802_MEAN_loss_0.3484_Lwlrap_0.8871_bceTP_0.1677_LB0.881.csv',
    ]




    # LB: 0.902
    data_files = [
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]
    # LB: 0.921
    data_files = [
        'blend_210118_231315_best_0ef12_best_L-B_0.904_0.901_0.892_LB0.925.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]
    # LB: 0.922
    data_files = [
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]
    # LB: 0.922
    data_files = [
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]
    # LB: 0.926
    data_files = [
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]
    # LB: 0.926
    data_files = [
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
    ]
    # LB: 0.928
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # 0.925
        '20210113_best/sub_gf_best_val_Lwlrap_0ef12_210113_0442_MEAN_Lwlrap_0.8900_f2_0.8040_LB0.901.csv',  # 0.926+Last
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
    ]
    # LB: 0.929
    data_files = [
        '20210113_best/sub_gf_best_e_id_0_0ef12_210113_0201_MEAN_Lwlrap_0.8731_f2_0.8051_LB0.904.csv',  # 0.925
        '20210118_best/sub_gf_best_val_loss_10d0c_210118_2033_MEAN_loss_0.1938_Lwlrap_0.8780_f2_0.4856_LB0.892.csv',
        '20210121_best/sub_gf_best_val_loss_11e06_210121_0533_MEAN_loss_0.1691_Lwlrap_0.8916_f2_0.5099_LB0.893.csv',
        '20210206_best/sub_gf_best_val_loss_17691_210206_1355_MEAN_loss_0.1399_Lwlrap_0.8614_f2_0.7221_LB0.912.csv',
    ]

    # LB: 0.921
    data_files = [
        '20210216_best/sub_gf_best_val_bceTP_1b099_210216_1731_MEAN_loss_0.0266_Lwlrap_0.9831_bceTP_0.1163.csv',
        '20210216_best/sub_gf_best_val_loss_1b099_210216_1710_MEAN_loss_0.0224_Lwlrap_0.9861_bceTP_0.1419_LB0.919.csv',
        '20210216_best/sub2_gf_best_val_bceTP_1b099_210216_1812_MEAN_loss_0.0232_Lwlrap_0.9877_bceTP_0.1123.csv',
        '20210216_best/sub2_gf_best_val_loss_1b099_210216_1751_MEAN_loss_0.0210_Lwlrap_0.9855_bceTP_0.1164.csv',
    ]

    # LB: 0.915
    data_files = [
        '20210216_best/sub_gf_best_val_bceTP_1b0a1_210216_2302_MEAN_loss_0.1789_Lwlrap_0.8488_f2_0.6614_bceTP_0.0756.csv',
        '20210216_best/sub_gf_best_val_loss_1b0a1_210216_2026_MEAN_loss_0.1476_Lwlrap_0.8783_f2_0.7571_bceTP_0.1253.csv',
    ]

    # LB: 0.915
    data_files = [
        'blend_210216_223844_1b099_1b099_1b099_1b099_LB_xxx_0.919_xxx_xxx___LB0.921.csv',
        'blend_210217_002610_1b0a1_1b0a1_LB_xxx_xxx___LB0.915.csv',
    ]

    # # LB: 0.938
    # data_files = [
    #     '20210217_best/sub_gf_best_val_bceTP_1b0a1_210216_2302_MEAN_loss_0.1789_Lwlrap_0.8488_f2_0.6614_bceTP_0.0756___LB0.915.csv',
    #     '20210217_best/sub2_gf_best_val_bceTP_1b0a1_210216_2302_MEAN_loss_0.1789_Lwlrap_0.8488_f2_0.6614_bceTP_0.0756___LB0.915.csv',
    #     '20210217_best/sub_gf_best_val_bceTP_1b5e9_210217_1722_MEAN_loss_0.0287_Lwlrap_0.8030_bceTP_0.1325.csv',
    #     '20210217_best/sub2_gf_best_val_bceTP_1b5e9_210217_1722_MEAN_loss_0.0258_Lwlrap_0.8374_bceTP_0.1130.csv',
    # ]

    # LB: 0.941
    data_files = [
        '20210217_best/blend_210207_235431_0ef12_10d0c_11e06_17691_LB_0.901_0.892_0.893_0.912___LB0.929.csv',
        'blend_210217_174408_1b0a1_1b0a1_1b5e9_1b5e9_LB_0.915_0.915_xxx_xxx___LB0.938.csv',
    ]

    # LB: 0.XXX
    data_files = [
        '20210217_best/sub_gf_best_val_bceTP_1b0a1_210216_2302_MEAN_loss_0.1789_Lwlrap_0.8488_f2_0.6614_bceTP_0.0756___LB0.915.csv',
        '20210217_best/sub2_gf_best_val_bceTP_1b5e9_210217_1722_MEAN_loss_0.0258_Lwlrap_0.8374_bceTP_0.1130.csv',
    ]

    # LB: 0.937
    data_files = [
        '20210217_best/blend_210207_235431_0ef12_10d0c_11e06_17691_LB_0.901_0.892_0.893_0.912___LB0.929.csv',
        'blend_210217_174848_1b0a1_1b5e9_LB_0.915_xxx.csv',
    ]

    # LB: 0.923
    data_files = [
        '20210217_best/sub_gf_best_val_bceTP_1b5e9_210217_1722_MEAN_loss_0.0287_Lwlrap_0.8030_bceTP_0.1325.csv',
        '20210217_best/sub2_gf_best_val_bceTP_1b5e9_210217_1722_MEAN_loss_0.0258_Lwlrap_0.8374_bceTP_0.1130.csv',
    ]



#
    is_notebook = platform.node() == 'nb-162'
    prefix = ''
    if is_notebook:
        prefix = '../runs_server/'

    run_path = Path(os.environ['DATA_FOLDER']) / 'runs' / prefix

    new_name = ''
    lb_values = '_LB'
    data = []  # type:List[pd.DataFrame]
    for data_file in data_files:
        data.append(pd.read_csv(run_path / data_file))

        data_file_short_name = data_file.replace('gf_best_val_Lwlrap_', 'gf_').replace('gf_best_val_f2_', 'gf_')
        data_file_short_name = data_file_short_name.replace('gf_best_val_loss_', 'gf_')
        data_file_short_name = data_file_short_name.replace('gf_best_val_bceTP_', 'gf_')
        if 'gf_' in data_file_short_name:
            gf_name = data_file_short_name.split('gf_')[1].split('_')[0]
        else:
            gf_name = 'blend'

        lb_value_split = data_file_short_name.split('_LB')
        lb_value = lb_value_split[-1].replace('.csv', '') if len(lb_value_split) > 1 else 'xxx'
        new_name += '_' + gf_name
        lb_values += '_' + lb_value

    time_prefix = datetime.now().strftime('blend_%y%m%d_%H%M%S')
    new_name = time_prefix + new_name + lb_values + '.csv'
    # new_name = '20210205_best_X.csv'

    mean_data = pd.concat(data).groupby('recording_id').mean()  # type:pd.DataFrame
    mean_data.to_csv(run_path / new_name, index_label='recording_id')


if __name__ == '__main__':
    main()
