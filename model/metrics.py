from typing import List

import numpy as np
import torch
from torch.nn import functional as F

SMOOTH = 1e-5


def get_binary_metrics_compnents(input: torch.Tensor, target: torch.Tensor, threshold=None):
    input = torch.sigmoid(input)

    if threshold is not None:
        input = F.threshold(input, threshold, 0, True)
        input = torch.clip(input / threshold, 0, 1)

    tp = torch.sum(torch.mul(input, target), dim=1)
    fp = torch.sum(input, dim=1) - tp
    fn = torch.sum(target, dim=1) - tp

    return tp, fp, fn


class CombinedMetrics(torch.nn.Module):
    def __init__(self, metrics: List[torch.nn.Module], metrics_scale: List[float]):
        super().__init__()
        self.metrics = metrics
        self.metrics_scale = metrics_scale

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        score = [self.metrics_scale[idx] * self.metrics[idx](input, target) for idx in range(len(self.metrics))]
        score = sum(score)
        return score


class BinaryRecallMetrics(torch.nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, fp, fn = get_binary_metrics_compnents(input, target, self.threshold)

        score = (tp + SMOOTH) / (tp + fn + SMOOTH)
        score = torch.mean(score)

        return score


class BinaryPrecisionMetrics(torch.nn.Module):
    def __init__(self, threshold=None):
        super().__init__()
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, fp, fn = get_binary_metrics_compnents(input, target, self.threshold)

        score = (tp + SMOOTH) / (tp + fp + SMOOTH)
        score = torch.mean(score)

        return score


class BinaryFScoreMetrics(torch.nn.Module):
    def __init__(self, beta, threshold=None):
        super().__init__()
        self.beta = beta
        self.threshold = threshold

    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        tp, fp, fn = get_binary_metrics_compnents(input, target, self.threshold)

        score = ((1 + self.beta ** 2) * tp + SMOOTH) \
                / ((1 + self.beta ** 2) * tp + self.beta ** 2 * fn + fp + SMOOTH)
        score = torch.mean(score)

        return score


class CustomMetrics:
    def __call__(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        input = torch.sigmoid(input)
        self.update({'prediction': input, 'target': target})
        res = self.compute()
        return torch.tensor(res)

    def reset(self):
        pass

    def update(self, step_output: dict):
        pass

    def compute(self):
        return 0.0


# Source: https://github.com/DCASE-REPO/dcase2019_task2_baseline/blob/master/evaluation.py
class LwlrapBase:
    """Computes label-weighted label-ranked average precision (lwlrap)."""

    def __init__(self, class_map):
        self.num_classes = 0
        self.total_num_samples = 0
        self._class_map = class_map

    def accumulate(self, batch_truth, batch_scores):
        """Accumulate a new batch of samples into the metric.
        Args:
          truth: np.array of (num_samples, num_classes) giving boolean
            ground-truth of presence of that class in that sample for this batch.
          scores: np.array of (num_samples, num_classes) giving the
            classifier-under-test's real-valued score for each class for each
            sample.
        """
        assert batch_scores.shape == batch_truth.shape
        num_samples, num_classes = batch_truth.shape
        if not self.num_classes:
            self.num_classes = num_classes
            self._per_class_cumulative_precision = np.zeros(self.num_classes)
            self._per_class_cumulative_count = np.zeros(self.num_classes, dtype=np.int)
        assert num_classes == self.num_classes
        for truth, scores in zip(batch_truth, batch_scores):
            pos_class_indices, precision_at_hits = (self._one_sample_positive_class_precisions(scores, truth))
            self._per_class_cumulative_precision[pos_class_indices] += (precision_at_hits)
            self._per_class_cumulative_count[pos_class_indices] += 1
        self.total_num_samples += num_samples

    def _one_sample_positive_class_precisions(self, scores, truth):
        """Calculate precisions for each true class for a single sample.
        Args:
          scores: np.array of (num_classes,) giving the individual classifier scores.
          truth: np.array of (num_classes,) bools indicating which classes are true.
        Returns:
          pos_class_indices: np.array of indices of the true classes for this sample.
          pos_class_precisions: np.array of precisions corresponding to each of those
            classes.
        """
        num_classes = scores.shape[0]
        pos_class_indices = np.flatnonzero(truth > 0)
        # Only calculate precisions if there are some true classes.
        if not len(pos_class_indices):
            return pos_class_indices, np.zeros(0)
        # Retrieval list of classes for this sample.
        retrieved_classes = np.argsort(scores)[::-1]
        # class_rankings[top_scoring_class_index] == 0 etc.
        class_rankings = np.zeros(num_classes, dtype=np.int)
        class_rankings[retrieved_classes] = range(num_classes)
        # Which of these is a true label?
        retrieved_class_true = np.zeros(num_classes, dtype=np.bool)
        retrieved_class_true[class_rankings[pos_class_indices]] = True
        # Num hits for every truncated retrieval list.
        retrieved_cumulative_hits = np.cumsum(retrieved_class_true)
        # Precision of retrieval list truncated at each hit, in order of pos_labels.
        precision_at_hits = (retrieved_cumulative_hits[class_rankings[pos_class_indices]] / (
                1 + class_rankings[pos_class_indices].astype(np.float)))
        return pos_class_indices, precision_at_hits

    def per_class_lwlrap(self):
        """Return a vector of the per-class lwlraps for the accumulated samples."""
        return (self._per_class_cumulative_precision / np.maximum(1, self._per_class_cumulative_count))

    def per_class_weight(self):
        """Return a normalized weight vector for the contributions of each class."""
        return (self._per_class_cumulative_count / float(np.sum(self._per_class_cumulative_count)))

    def overall_lwlrap(self):
        """Return the scalar overall lwlrap for cumulated samples."""
        return np.sum(self.per_class_lwlrap() * self.per_class_weight())

    def __str__(self):
        per_class_lwlrap = self.per_class_lwlrap()
        per_class_weight = self.per_class_weight()
        # List classes in descending order of lwlrap.
        s = ([f'Lwlrap({name}) = {lwlrap:.06f} x w{weight:.06f}' for (lwlrap, weight, name) in
              sorted([(per_class_lwlrap[i], per_class_weight[i], self._class_map[i]) for i in range(self.num_classes)],
                     reverse=True)])
        s.append('Overall lwlrap = %.6f' % (self.overall_lwlrap()))
        return '\n'.join(s)


class Lwlrap(CustomMetrics):
    name = 'lwlrap'
    better = 'max'

    def __init__(self, classes, filter_classes_num=0):
        self.classes = classes
        self.lwlrap = LwlrapBase(self.classes)
        self.filter_classes_num = filter_classes_num

    def reset(self):
        self.lwlrap.num_classes = 0
        self.lwlrap.total_num_samples = 0

    def update(self, step_output: dict):
        pred = step_output['prediction'].cpu().numpy()
        trg = step_output['target'].cpu().numpy()

        if self.filter_classes_num > 0:
            pred = pred[:, :self.filter_classes_num]
            trg = trg[:, :self.filter_classes_num]

            allowed_rows = []
            for idx, trg_row in enumerate(trg):
                if np.count_nonzero(trg_row) > 0:
                    allowed_rows.append(idx)

            pred = pred[allowed_rows, :]
            trg = trg[allowed_rows, :]

        self.lwlrap.accumulate(trg, pred)

    def compute(self):
        return self.lwlrap.overall_lwlrap()

    def __str__(self):
        return str(self.lwlrap)
