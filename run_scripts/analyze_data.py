import csv

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from model.datasets import SampleDataset


def collect_data(data: dict, csv_path, is_tp):
    # recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max
    reader = csv.DictReader(open(csv_path, "r"))
    fieldnames = reader.fieldnames

    for row in reader:
        species_id = row['species_id''']
        songtype_id = row['songtype_id']
        t_min = float(row['t_min'])
        t_max = float(row['t_max'])
        f_min = float(row['f_min'])
        f_max = float(row['f_max'])
        duration = t_max - t_min

        key = f'{species_id}|{songtype_id}|{is_tp}'
        if key not in data:
            data[key] = {'count': 0, 'duration': [], 'f_min': [], 'f_max': []}

        data[key]['count'] = data[key]['count'] + 1
        data[key]['duration'].append(duration)
        data[key]['f_min'].append(f_min)
        data[key]['f_max'].append(f_max)


def collect_mean_values(data):
    def _get_stat(data_item):
        return np.min(data_item), np.max(data_item), np.mean(data_item)

    for key, value in data.items():
        value['duration'] = _get_stat(value['duration'])
        value['f_min'] = _get_stat(value['f_min'])
        value['f_max'] = _get_stat(value['f_max'])


def analyze_data(path, is_tp):
    data = pd.read_csv(path)
    data[SampleDataset.k_key] = data['species_id'].astype(str) + '|' + data['songtype_id'].astype(str)
    data[SampleDataset.k_duration] = data['t_max'] - data['t_min']
    # data = data.sort_values([SampleDataset.k_species_id, SampleDataset.k_songtype_id], ascending=[True, True])

    gr_min = data.groupby(SampleDataset.k_key).min(numeric_only=True)
    gr_mean = data.groupby(SampleDataset.k_key).mean(numeric_only=True)
    gr_max = data.groupby(SampleDataset.k_key).max(numeric_only=True)

    gr_min = gr_min.sort_values([SampleDataset.k_species_id, SampleDataset.k_songtype_id])  # type:pd.DataFrame
    gr_mean = gr_mean.sort_values([SampleDataset.k_species_id, SampleDataset.k_songtype_id])  # type:pd.DataFrame
    gr_max = gr_max.sort_values([SampleDataset.k_species_id, SampleDataset.k_songtype_id])  # type:pd.DataFrame

    gr_orig = gr_min[[SampleDataset.k_species_id, SampleDataset.k_songtype_id]]
    gr_orig['tp'] = is_tp

    columns = [SampleDataset.k_f_min, SampleDataset.k_f_max, SampleDataset.k_duration]
    gr_min = gr_min[columns].add_suffix('_min')
    gr_mean = gr_mean[columns].add_suffix('_mean')
    gr_max = gr_max[columns].add_suffix('_max')

    new_data = pd.concat([gr_orig, gr_min, gr_mean, gr_max], axis=1)
    columns = [
        SampleDataset.k_species_id, SampleDataset.k_songtype_id, 'tp',
        SampleDataset.k_f_min + '_min', SampleDataset.k_f_min + '_mean', SampleDataset.k_f_min + '_max',
        SampleDataset.k_f_max + '_min', SampleDataset.k_f_max + '_mean', SampleDataset.k_f_max + '_max',
        SampleDataset.k_duration + '_min', SampleDataset.k_duration + '_mean', SampleDataset.k_duration + '_max',
    ]
    new_data = new_data[columns]

    return new_data


def main():
    statistics_data = {}
    statistics_data_tp = {}
    statistics_data_fp = {}

    # tips = sns.load_dataset("tips")
    # sns.displot(tips, x="size")
    # plt.show()

    # collect_data(statistics_data, r'd:\Projects\Kaggle\rfcx-species-audio-detection_data\train_tp.csv', 1)
    # collect_data(statistics_data, r'd:\Projects\Kaggle\rfcx-species-audio-detection_data\train_tp.csv', 1)
    # collect_data(statistics_data_tp, r'd:\Projects\Kaggle\rfcx-species-audio-detection_data\train_tp.csv', 1)
    # collect_data(statistics_data_fp, r'd:\Projects\Kaggle\rfcx-species-audio-detection_data\train_fp.csv', 0)

    data_tp = analyze_data(r'../data/train_tp.csv', 1)
    data_fp = analyze_data(r'../data/train_fp.csv', 0)
    data_all = pd.concat([data_tp, data_fp])

    data_all.to_csv(r'../data/stat.csv', index=False)
    exit(0)

    # recording_id,species_id,songtype_id,t_min,f_min,t_max,f_max
    csv_tp = pd.read_csv(r'../data/train_tp.csv')
    csv_tp['key'] = csv_tp['species_id'].astype(str) + '|' + csv_tp['songtype_id'].astype(str)
    csv_tp['duration'] = csv_tp['t_max'] - csv_tp['t_min']
    csv_tp.sort_values(['f_min', 'f_max'], ascending=[True, True])

    csv_fp = pd.read_csv(r'../data/train_fp.csv')
    csv_fp['key'] = csv_fp['species_id'].astype(str) + '|' + csv_fp['songtype_id'].astype(str)
    csv_fp['duration'] = csv_fp['t_max'] - csv_fp['t_min']
    csv_fp.sort_values(['f_min', 'f_max'], ascending=[True, True])

    species_id_tp = sorted(set(csv_tp['species_id']))
    species_id_fp = sorted(set(csv_fp['species_id']))
    keys_tp = sorted(set(csv_tp['key']))
    keys_fp = sorted(set(csv_fp['key']))

    min_tp = csv_tp['f_min'].min()
    mean_tp = csv_tp['f_min'].mean()
    max_tp = csv_tp['f_max'].max()
    min_fp = csv_fp['f_min'].min()
    mean_fp = csv_fp['f_min'].mean()
    max_fp = csv_fp['f_max'].max()

    d_min_tp = csv_tp['duration'].min()
    d_mean_tp = csv_tp['duration'].mean()
    d_max_tp = csv_tp['duration'].max()
    d_min_fp = csv_fp['duration'].min()
    d_mean_fp = csv_fp['duration'].mean()
    d_max_fp = csv_fp['duration'].max()

    df_1 = csv_tp[['key', 'f_min']]
    df_1['f_type'] = 'min'
    df_1.rename(columns={'f_min': 'f'}, inplace=True)
    df_2 = csv_tp[['key', 'f_max']]
    df_2['f_type'] = 'max'
    df_2.rename(columns={'f_max': 'f'}, inplace=True)

    df = pd.concat([df_1, df_2])
    df = df.sort_values(['f'], ascending=[False])
    # grouped = csv_tp.groupby("key")
    # users_sorted_average = pd.DataFrame({col: vals['key'] for col, vals in grouped}).mean().sort_values(
    #     ascending=True)

    # sns.displot(csv_tp, x='key', y='duration', height=5, aspect=3)
    sns.displot(csv_tp, x='key', y='duration', height=5, aspect=3)
    plt.show()

    # sns.catplot(x="key", y="f_min",data=csv_tp)
    sns.catplot(x="key", y="f", hue='f_type', data=df)
    plt.show()
    exit(1)

    collect_mean_values(statistics_data_tp)

    keys = sorted(statistics_data_fp.keys())
    counts = [statistics_data_fp[k]['count'] for k in keys]

    sns.displot(x=keys, y=counts, height=5, aspect=3)
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.show()


if __name__ == '__main__':
    main()
