import csv
import functools
import os
import sys
import time
from typing import Dict


class StreamWrapper(object):
    def __init__(self, file_path, original_stream):
        self.file = open(file_path, 'a', encoding='utf-8', buffering=1)
        self.original_stream = original_stream

    def write(self, buf):
        self.original_stream.write(buf)
        self.file.write(buf)

    def flush(self):
        self.original_stream.flush()
        self.file.flush()

    def close(self):
        self.file.close()


def perf_timing(file=None):
    def perf_timing_wrap(f):
        @functools.wraps(f)
        def wrap(*args, **kwargs):
            start = time.perf_counter()
            ret = f(*args, **kwargs)
            end = time.perf_counter()

            out_file = file if file is not None else sys.stdout
            print("{} function took {:0.3f} ms".format(f.__qualname__, (end - start) * 1000.0), file=out_file)
            return ret

        return wrap

    return perf_timing_wrap


def fix_windows_path(file_path: str):
    if sys.platform == 'win32':
        file_path = os.path.abspath(file_path)
        if file_path.startswith('\\\\?\\') is False:
            file_path = '\\\\?\\' + file_path

    return file_path


def write_csv_train_metrics(csv_path, metrics_train: Dict):
    with open(csv_path, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + list(metrics_train.keys()))

        for epoch in range(len(next(iter(metrics_train.values())))):
            row = [epoch + 1]
            row.extend([f"{value[epoch]:.8f}" for value in metrics_train.values()])
            writer.writerow(row)
