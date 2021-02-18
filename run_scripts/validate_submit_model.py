import argparse
import json
import os
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import List, Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from model.datasets import SampleDataset, make_dataloader
from model.metrics import Lwlrap
from run_scripts.train_model_simple import build_model, build_optimizer
from model.train_utils import print_fold_train_results_and_save, store_samples_dataset_info
from model.train_validate_loops import validate_epoch
from utils.utils import StreamWrapper

_allowed_scores = ['loss', 'Lwlrap', 'f2', 'bceTP']


def collect_fold_models(run_path: Path, model_prefix, model_suffix):
    fold_folders = list(run_path.glob('models_fold_*'))  # type:List[Path]
    fold_folders = [fold for fold in fold_folders if fold.is_dir()]
    fold_folders = sorted(fold_folders)

    files = []
    scores = {}  # type:Dict[str,List]

    for fold in fold_folders:
        prefix = f'model{model_suffix}{model_prefix}_'
        fold_files = list(fold.glob(prefix + '*.pt'))  #
        fold_files = [fold_file for fold_file in fold_files if not fold_file.stem.endswith('_optimizer')]
        fold_files = sorted(fold_files)

        file = fold_files[-1]
        file_parts = str(file.stem)[len(prefix):].split('_')
        epoch = file_parts[0]
        file_scores = {metric: float(value) for metric, value in zip(*([iter(file_parts[1:])] * 2))}

        if not scores:
            scores = {metric: [] for metric in file_scores}

        for metric, value in file_scores.items():
            scores[metric].append(value)

        files.append(file)

    for metric, value in scores.items():
        scores[metric].append(np.mean(value))

    return files, scores


def save_submit(record_ids, outputs, run_path, fold_group_id, model_prefix, time_prefix, submit_suffix, class_map,
                model_suffix):
    results = {SampleDataset.k_recording_id: record_ids}

    for class_index in range(outputs.shape[1]):
        class_key = f's{class_map[class_index]}'
        results[class_key] = outputs[:, class_index]

    df = pd.DataFrame(results, )
    submission_path = run_path / f'sub{model_suffix}_gf{model_prefix}_{fold_group_id}_{time_prefix}_{submit_suffix}.csv'
    df.to_csv(submission_path, index=False)


def get_score_suffix(scores, idx):
    suffix = ''

    for name in _allowed_scores:
        if name not in scores:
            continue

        if suffix:
            suffix = suffix + '_'
        suffix = suffix + f'{name}_{scores[name][idx]:0.4f}'

    return suffix


def do_validate_submit(is_submit_mode, run_name, model_prefix='', model_suffix=''):
    is_debug_mode = sys.gettrace() is not None

    # set env variable for data
    os.environ['DATA_FOLDER'] = '../data'

    # ===== Init folders and models list

    if model_prefix:
        model_prefix = '_' + model_prefix

    if run_name is None:
        # run_name = '_run_20-12-14_16-41-10_gf_04c55'  # submission_20-12-15_16-29-21_mean LB: 0.816
        # run_name = '_run_20-12-16_11-23-36_gf_05657'  # sub_gf_05657_201217_0105_mean_score_0.823873 LB:0.789
        # run_name = '_run_20-12-15_23-44-00_gf_0539c'  # sub_gf_0539c_201217_0113_mean_score_0.822257 LB:0.815 T:0.885
        # run_name = '_run_20-12-16_15-38-12_gf_05756'
        # run_name = '_run_20-12-16_15-41-18_gf_05759'
        # run_name = '_run_20-12-13_01-18-02_gf_0431a'
        # run_name = '_run_20-12-18_13-18-10_gf_0620a'  # sub_gf_0620a_201218_1516_mean_score_0.821364 LB:0.824 T:0.
        # run_name = '_run_20-12-18_18-10-24_gf_0632e'  # sub_gf_0632e_201220_0014_mean_score_0.821638 LB:0.824 T:0.
        # run_name = '_run_20-12-20_01-45-26_gf_06a95'  # sub_gf_06a95_201220_2356_mean_score_0.823640 LB:0.822 T:0. | en0 Ag10 RC10+v10-5 Cut bce-sum
        # run_name = '_run_20-12-21_00-55-10_gf_07003'  # sub_gf_07003_201221_1144_mean_score_0.800503 LB:0.000 T:0. | en0 Ag10 RC10+v10-5 Cut bce-smooth-0.24 g0
        # run_name = '_run_20-12-21_18-27-48_gf_0741f'  # sub LB:0.000 T:0. | en0 Ag10 RC10+v10-5 Cut bce-smooth-0.24 RemDup
        # run_name = '_run_20-12-23_02-16-19_gf_07b94'  # sub LB:0.000 T:0. | en0 Ag12 384x768 RC6+v6-2 bce-smooth-0.24

        # run_name = '_run_20-12-17_18-18-49_gf_05d96'  # sub_gf_05d96_201218_1230_mean_score_0.819896 LB:0.830 T:0.868604
        # run_name = '_run_20-12-17_18-19-15_gf_05d97'  # sub_gf_05d97_201218_1230_mean_score_0.816678 LB:0.839 T:0.867563
        # run_name = '_run_20-12-20_01-43-38_gf_06a93'  # sub_gf_06a93_201220_2349_mean_score_0.809369 LB:0.835 T:0.862578 | en0 Ag10 RC10+v10-5 Cut bce-smooth-0.24
        # run_name = '_run_20-12-21_00-55-49_gf_07003'  # sub_gf_07003_201221_1151_mean_score_0.798729 LB:0.848 T:0.866835 | en0 Ag10 RC10+v10-5 Cut bce-smooth-0.24 g1
        # run_name = '_run_20-12-23_02-16-58_gf_07b94'  # sub_gf_07b94_201223_1109_mean_score_0.825592 LB:0.844 T:0.878845 | en0 Ag12 384x768 RC6+v6-2 bce-mean
        # run_name = '_run_20-12-23_17-50-47_gf_07f3a'  # sub_gf_07f3a_201224_1111_mean_score_0.887582 LB:0.847 T:0.887440 | en0 Ag12 384x768 RC6+v6-2-wo_empty_cut bce-mean e50
        # run_name = '_run_20-12-23_17-52-57_gf_07f3c'  # sub_gf_07f3c_201224_1111_mean_score_0.884792 LB:0.843 T:0.884647 | en0 Ag12 384x768 RC6+v6-2-wo_EC reduceLR-6-0.6 e100
        # run_name = '_run_20-12-24_13-13-18_gf_083c5'  # sub_gf_083c5_201225_2259_mean_score_0.899647 LB:0.845 | en4 Ag12 384x768 RC6+v6-2-wo_EC reduceLR-6-0.6 e100
        # run_name = '_run_20-12-24_17-01-22_gf_084a9'  # sub_gf_084a9_201225_2326_mean_score_0.893309 LB:0.831 | rns50 Ag12 384x768 RC6+v6-2-wo_EC reduceLR-6-0.6 e100 G2
        # run_name = '_run_20-12-25_02-24-27_gf_086dc'  # sub_gf_086dc_201225_2328_mean_score_0.887892 LB:0.854 | en4 Ag12 384x768 RC6+v6-2-wo_EC reduceLR-6-0.6 e100 kf-SSS-10
        # run_name = '_run_20-12-26_01-56-07_gf_08c60'  # en4 Ag12 Mix1 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10 LB:0.828 | en4 Ag12 Mix1 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = '_run_20-12-27_00-23-10_gf_091a3'  # sub_gf_091a3_201228_0101_mean_score_0.891370.csv LB:0.838 | en0 Ag12 Mix2 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = '_run_20-12-27_00-24-00_gf_091a4'  # sub_gf_091a4_201228_0101_mean_score_0.881926 LB:0.840 | en0 Ag12 Mix0 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = '_run_20-12-28_12-50-48_gf_09a2e'  # sub_gf_09a2e_201228_2333_mean_score_0.895147 LB:0.849 | en4 Ag12 Mix2 384x768 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = '_run_20-12-29_01-46-28_gf_09d36'  # sub_gf_09d36_201229_1003_mean_score_0.888785 LB:0.850 | en4 Ag12 Mix2 384x384 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = '_run_20-12-29_11-08-54_gf_09f68'  # sub_gf_09f68_201229_2357_mean_score_0.895878 LB:0.855 | en4 Ag12 Mix2 384x786 fixMetric RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        run_name = '_run_20-12-30_16-58-19_gf_0a666'  # sub_gf_0a666_201230_2252_mean_score_0.896035 LB:0.886 | en4 Ag13 Mix3 384x786 RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # run_name = ''  # sub LB:0.000 |

        # run_name = '_run_20-12-29_11-08-54_gf_09f68'  # sub_gf_09f68_201229_2357_mean_score_0.895878 LB:0.855 | en4 Ag12 Mix2 384x786 fixMetric RC6+v6-2woE rLR-6-0.6 kf-SSS-10
        # experiments:
        #      RC     -   Test   -  LB   - File
        # RC6+v6-1woE - 0.891630 - 0.000 - sub_gf_09f68
        # RC6+v6-2woE - 0.895878 - 0.855 - sub_gf_09f68_201229_2357_mean_score_0.895878
        # RC6+v6-3woE - 0.892270 - 0.855 - sub_gf_09f68_201230_1619_mean_score_BY_3_0.892270
        # RC6+v6-4woE - 0.894788 - 0.853 - sub_gf_09f68_201230_1553_mean_score_BY_4_0.894788
        # RC6+v6-5woE - 0.889870 - 0.000 - sub_gf_09f68
        # RC6+v6-6woE - 0.886759 - 0.000 - sub_gf_09f68

    update_validate_params = {}
    # update_validate_params = {'by_crops': 6, 'crops_offset': 3, 'by_labeled_crops': True}

    is_notebook = platform.node() == 'nb-162'
    # if is_notebook:
    #     run_name = '../runs_server/' + run_name

    run_path = Path(os.environ['DATA_FOLDER']) / 'runs' / run_name
    model_files, scores = collect_fold_models(run_path, model_prefix, model_suffix)
    fold_group_id = run_name.split('_gf_')[1]

    # ===== Loggers
    time_prefix = datetime.now().strftime('%y%m%d_%H%M')
    log_prefix = f'submit{model_suffix}_log{model_prefix}_' if is_submit_mode else f'validate{model_suffix}_log{model_prefix}_'
    log_prefix = log_prefix + time_prefix

    orig_stdout, orig_stderr = sys.stdout, sys.stderr
    sys.stdout = StreamWrapper(run_path / f'{log_prefix}_output.log', orig_stdout)
    sys.stderr = StreamWrapper(run_path / f'{log_prefix}_error.log', orig_stderr)

    # ==== INIT MODEL
    print('is submit mode:', is_submit_mode)

    with open(run_path / 'cfg.json', 'r', encoding='utf-8') as f:
        cfg = json.load(f)
    with open(run_path / 'labels.json', 'r', encoding='utf-8') as f:
        class_map = json.load(f)
    cfg['train_params']['pretrained'] = False
    cfg['train_params']['optimizer'] = None
    cfg['submit_data_loader'] = dict(cfg['test_data_loader'])
    cfg['submit_data_loader']['key'] = 'test'

    # scores = pd.read_csv(run_path / 'folds_result.csv')

    limit_samples = None
    # limit_samples = 20  # DEBUG
    if is_notebook:
        if is_debug_mode:
            limit_samples = 10
        cfg['train_data_loader']['batch_size'] = 4
        cfg['test_data_loader']['batch_size'] = 4
        cfg['submit_data_loader']['batch_size'] = 4

    cfg['test_data_loader']['batch_size'] = 5  # 'by_labeled_crops' will be faster with batch_size 4 - skips more cuts
    # cfg['submit_data_loader']['batch_size'] = 32
    cfg['submit_data_loader']['batch_size'] = 16
    if 'validate_params' in cfg and cfg['validate_params'].get('by_crops', 0) > 0:
        cfg['validate_params']['by_labeled_crops'] = True

    cfg['validate_params'].update(update_validate_params)

    # # DEBUG
    # cfg['validate_params'] = {
    # #     'target_crop': True, 'crops_padding': 0.5
    # }
    # # cfg['test_data_loader']['cut_samples'] = True
    # # cfg['test_data_loader']['batch_size'] = 1
    # # cfg['test_data_loader']['batch_size'] = 4
    # cfg['validate_params'] = {
    #     # 'by_crops': 10, 'crops_offset': 5,
    #     'by_crops': 10, 'crops_offset': 2,
    #     # 'by_crops': 4, 'crops_offset': 2,
    # }

    num_workers = 2
    # limit_samples, num_workers = 10, 0  # DEBUG
    cfg['train_data_loader']['num_workers'] = num_workers
    cfg['test_data_loader']['num_workers'] = num_workers
    cfg['submit_data_loader']['num_workers'] = num_workers

    print('=' * 36, 'Config', '=' * 36)
    print(json.dumps(cfg, indent=2, ensure_ascii=False))
    print('=' * 80)

    # ===== INIT DATASET
    print("Opening dataset...")

    if cfg['train_params']['loss'] in ('bce_fp6.4', 'bce_fp'):
        cfg['train_params']['loss'] = 'bce'
    if is_submit_mode:
        cfg['train_data_loader']['use_fp'] = False

    folds_count = cfg['train_params'].get('folds', 1)
    kfold_statified_shuffle_splits = cfg['train_params'].get('kfold_sss', False)
    if not is_submit_mode:
        samples_dataset = SampleDataset(cfg['audio_params'], cfg['train_data_loader'], folds_count,
                                        disable_cache=is_notebook,
                                        kfold_statified_shuffle_splits=kfold_statified_shuffle_splits)
    else:
        samples_dataset = SampleDataset(cfg['audio_params'], cfg['submit_data_loader'], -1, disable_cache=is_notebook,
                                        is_test_mode=True, classes_num=len(class_map))

    # ==== INIT MODEL
    print("Init model...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model_list = []
    for model_path in model_files:
        print('Loading model:', model_path)
        model = build_model(cfg, samples_dataset.classes_num)

        state_dict = torch.load(model_path, map_location=device)
        if cfg['train_params'].get('multi_gpu', False):
            prefix = 'module.'
            state_dict = {k if not str(k).startswith(prefix) else k[len(prefix):]: v for k, v in state_dict.items()}
        model.load_state_dict(state_dict, strict=True)

        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        model.to(device)

        model.to(device).eval()
        model_list.append(model)

    _, criterion_list = build_optimizer(cfg, None, class_map, samples_dataset.classes_num)

    # ==== Validate LOOP
    fold_results = {'fold': []}
    lwlrap_metric = Lwlrap(class_map)

    folds_outputs_list, folds_record_ids = [], []
    for fold_index in range(folds_count):
        if not is_submit_mode:
            dataloader, dataset = make_dataloader(samples_dataset, samples_dataset.test_split[fold_index], cfg,
                                                  'test', sort_t_min=True)
            store_samples_dataset_info(run_path, samples_dataset.classes_labels, None, dataset.dataset_split,
                                       f'_validate_f{fold_index}', None)
        else:
            dataloader, dataset = make_dataloader(samples_dataset, samples_dataset.train_split[0], cfg, 'submit')

        metrics_train = {'val_loss': []}
        metrics_train.update({'val_' + metric: [] for metric in cfg['metrics']})

        progress_bar = tqdm(total=len(dataloader.dataset), file=orig_stderr)
        res = validate_epoch(1, 1, model_list[fold_index], device, criterion_list, dataloader, cfg, metrics_train,
                             progress_bar, limit_samples, fold_index, folds_count)
        outputs, targets, record_ids = res
        progress_bar.close()

        if is_submit_mode:
            outputs = torch.sigmoid(outputs)
            folds_outputs_list.append(outputs.cpu().numpy())
            if not folds_record_ids:
                folds_record_ids = record_ids
            assert folds_record_ids == record_ids, "Record ids should be equal for every fold!"

            save_submit(record_ids, folds_outputs_list[-1], run_path, fold_group_id, model_prefix, time_prefix,
                        f'fold_{fold_index}_' + get_score_suffix(scores, fold_index), class_map, model_suffix)

        # collect and print results
        if not is_submit_mode:
            print('=' * 28, f'Results for Fold {fold_index + 1} / {folds_count}', '=' * 28)

            lwlrap_metric(outputs, targets)
            print('Lwlrap concatenate:')
            print(lwlrap_metric)

            print('-' * 80)
            for key, value in metrics_train.items():
                if key not in fold_results:
                    fold_results[key] = []
                fold_results[key].append(value[-1])
            fold_results['fold'].append(fold_index)
            print_fold_train_results_and_save(fold_results, None)

            print('=' * 80)

    # ===== Store submission files
    if not is_submit_mode:
        exit(0)

    folds_outputs = np.array(folds_outputs_list)  # type: np.ndarray
    # sum_outputs = np.sum(folds_outputs, axis=0)
    mean_outputs = np.mean(folds_outputs, axis=0)
    # max_outputs = np.max(folds_outputs, axis=0)

    # save_submit(record_ids, sum_outputs, run_path, 'sum', time_prefix, fold_group_id, scores[-1])
    save_submit(record_ids, mean_outputs, run_path, fold_group_id, model_prefix, time_prefix,
                'MEAN_' + get_score_suffix(scores, -1), class_map, model_suffix)
    # save_submit(record_ids, max_outputs, run_path, 'max', time_prefix, fold_group_id, scores[-1])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--submit', '-s', help="Submit results", action='store_true')
    parser.add_argument('--run_name', '-r', help="Name of the folder with run", default=None)
    parser.add_argument('--model_prefix', '-pref', default='')
    parser.add_argument('--model_suffix', '-m', default='')
    args = parser.parse_args()

    do_validate_submit(args.submit, args.run_name, args.model_prefix, args.model_suffix)
