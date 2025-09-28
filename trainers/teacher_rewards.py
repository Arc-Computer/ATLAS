import os
import abc
import gc
from collections import defaultdict
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from typing import Any, Callable, Optional, Sequence
from .teacher_base import (
    find_sublist_start_end, extract_and_left_align_from_mask, TeacherReward,
    find_valid_subsequence, find_first_last_one_idxs, log_tensor_info,
    is_tensor, TeacherTrainer,
)
from .extraction_utils import ATLASExtractionUtils
import random


def combine_items(items):

    if isinstance(items[0], torch.Tensor):
        return torch.cat(items, dim=0)

    elif isinstance(items[0], float):
        return items

    elif isinstance(items[0], list):
        return items

    elif isinstance(items[0], dict):
        combined = {}
        for key in items[0]:

            values = [item[key] for item in items]
            combined[key] = combine_items(values)
        return combined
    else:
        return items


def combine_list_elements(list_of_lists):

    n = len(list_of_lists[0])
    result = []
    for i in range(n):
        items = [lst[i] for lst in list_of_lists]
        result.append(combine_items(items))
    return result


def to_torch_tensor(data, device='cpu', dtype=None):

    if isinstance(data, torch.Tensor):
        return data.to(device, dtype=dtype) if dtype else data.to(device)

    if isinstance(data, np.ndarray):
        tensor = torch.from_numpy(data)
        return tensor.to(device, dtype=dtype) if dtype else tensor.to(device)

    if isinstance(data, (list, tuple)):
        tensor = torch.tensor(
            data, dtype=dtype) if dtype else torch.tensor(data)
        return tensor.to(device)

    raise TypeError


