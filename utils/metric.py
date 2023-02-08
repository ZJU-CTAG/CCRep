from typing import Optional

import torch
from allennlp.training.metrics import Metric

use_predicted_idxes_metrics = ['BooleanAccuracy']
use_probabilities_metrics = {
    'Auc': {'squeeze': True, 'expand': False},
    'CategoricalAccuracy': {'squeeze': False, 'expand': False},
    'F1Measure': {'squeeze': False, 'expand': True},
}


def update_metric(metric: Metric,
                  pred_idxes: torch.Tensor,
                  probs: torch.Tensor,
                  labels: torch.Tensor,
                  mask: Optional[torch.BoolTensor] = None):
    if metric is None:
        return
    metric_class_name = metric.__class__.__name__
    labels = labels.view(labels.size(0),)
    if metric_class_name in use_predicted_idxes_metrics:
        metric(pred_idxes, labels, mask=mask)
    elif metric_class_name in use_probabilities_metrics:
        if use_probabilities_metrics[metric_class_name]['squeeze'] and len(probs.shape) > 1:
            probs = probs[:, 1]
        if use_probabilities_metrics[metric_class_name]['expand'] and len(probs.shape) == 1:
            probs = expand_1d_prob_tensor(probs)
        metric(probs, labels, mask=mask)
    else:
        raise ValueError(f'Unregistered metric: {metric_class_name}')

def expand_1d_prob_tensor(prob_tensor):
    residual_prob_tensor = torch.ones((prob_tensor.size(0),), device=prob_tensor.device)
    residual_prob_tensor = residual_prob_tensor - prob_tensor
    return torch.stack((residual_prob_tensor, prob_tensor), dim=1)