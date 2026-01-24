import os
import abc
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Callable, Sequence
from transformers import PreTrainedTokenizer
import re
import random
from itertools import islice


def is_subsequence(sub, seq):
    sub_len = len(sub)
    return any(sub == list(islice(seq, i, i + sub_len))
               for i in range(len(seq) - sub_len + 1))


def find_valid_subsequence(sub, seq):
    if is_subsequence(sub, seq):
        return sub
    if len(sub) > 1:
        if is_subsequence(sub[1:], seq):
            return sub[1:]
        if is_subsequence(sub[:-1], seq):
            return sub[:-1]
    if len(sub) > 2 and is_subsequence(sub[1:-1], seq):
        return sub[1:-1]
    return None


def is_tensor(t):
    if isinstance(t, torch.Tensor):
        return True
    return False


def log_tensor_info(tensor):
    print(f"Shape: {tensor.shape}")
    print(f"Max: {tensor.max().item()}")
    print(f"Min: {tensor.min().item()}")
    print(f"Mean: {tensor.float().mean().item()}")
    print(f"Nan: {torch.isnan(tensor).any().item()}")


def find_first_last_one_idxs(tensor, negative_indices=False):
    b, m = tensor.shape
    tensor = tensor.to(torch.long)
    first_indices = torch.where(tensor.any(dim=1), tensor.argmax(dim=1), -1)
    last_indices = torch.where(
        tensor.any(dim=1), m - 1 - tensor.flip(dims=[1]).argmax(dim=1), -1)

    if negative_indices:
        first_indices = torch.where(
            first_indices != -1, first_indices - m, -1)
        last_indices = torch.where(
            last_indices != -1, last_indices - m, -1)

    return first_indices, last_indices


def extract_and_left_align_from_mask(matrix, mask):
    bsz = matrix.size(0)
    rows = []
    max_len = 0
    for i in range(bsz):
        row = matrix[i][mask[i].bool()]
        rows.append(row)
        max_len = max(max_len, row.size(0))

    padded_rows, padded_masks = [], []
    for row in rows:
        pad_len = max_len - row.size(0)

        padded_rows.append(F.pad(row, (0, pad_len)))
        row_mask = torch.ones_like(row, dtype=torch.bool)
        padded_masks.append(F.pad(row_mask, (0, pad_len)))

    data_out = torch.stack(padded_rows, dim=0)
    mask_out = torch.stack(padded_masks, dim=0)
    return data_out, mask_out


def find_sublist_start_end(lst, sublst, from_end=False, reverse_search=False):
    if sublst is not None:
        sub_len = len(sublst)
        indices_range = (
            range(len(lst) - sub_len, -1, -1) if reverse_search else
            range(len(lst) - sub_len + 1)
        )

        for i in indices_range:
            if lst[i:i + sub_len] == sublst:
                if from_end:
                    return i - len(lst), i + sub_len - len(lst)
                else:
                    return i, i + sub_len
    return None


def find_indices_between_tags(content, start_tag, end_tag):

    failures = 1
    start_content = content.find(start_tag)
    if start_content == -1:
        failures += 1
        start_content = 0
    end_content = content.find(end_tag, start_content + len(start_tag))
    if end_content == -1:
        failures += 1
        end_content = -1
    return start_content, end_content, failures


def replace_text_between_tags(content, content2, start_tag, end_tag):

    start_content = content.find(start_tag)
    if start_content == -1:
        raise NotImplementedError
    end_content = content.find(end_tag, start_content + len(start_tag))
    if end_content == -1:
        raise NotImplementedError

    start2 = content2.find(start_tag)
    if start2 == -1:
        raise NotImplementedError
    end2 = content2.find(end_tag, start2 + len(start_tag))
    if end2 == -1:
        raise NotImplementedError

    sub2 = content2[start2 + len(start_tag):end2]

    new_content = (content[:start_content + len(start_tag)]
                   + sub2
                   + content[end_content:])

    replaced_start_new = start_content + len(start_tag)
    replaced_end_new = replaced_start_new + len(sub2)
    replaced_start_c2 = start2 + len(start_tag)
    replaced_end_c2 = replaced_start_c2 + len(sub2)

    return (new_content,
            (replaced_start_c2, replaced_end_c2),
            (replaced_start_new, replaced_end_new))


class TeacherTrainer(abc.ABC):
    def __init__(
        self,
        student_model,
        teacher_model,
        tokenizer,
        reward_functions=None,
        output_dir=None,
        logging_prob=0.0
    ):
        if reward_functions is None:

            reward_functions = self.reward_funcs
        for rw in reward_functions:
            if isinstance(rw, TeacherReward):
                rw.link_with_trainer(
                    trainer=self,
                    student_model=student_model,
                    teacher_model=teacher_model,
                    tokenizer=tokenizer,
                )
        self.logging_dir = output_dir
        self.logging_prob = logging_prob
        self.last_logged_iter = -1
        self.do_log_to_file = False
        if logging_prob > 0.0 and output_dir is not None:
            self.do_log_to_file = True
            self.logging_dir = output_dir + '/teacher_chats'
            os.makedirs(self.logging_dir, exist_ok=True)
            self.logging_file = self.logging_dir + '/log.txt'

    def log_to_file(self, *args, **kwargs):
        if self.do_log_to_file:
            if not (self.state.global_step == self.last_logged_iter):
                with open(f"{self.logging_file}", "a") as f:
                    f.write("\n\n============================\n" +
                            f"Global step: {self.state.global_step}")
                self.last_logged_iter = self.state.global_step
            if random.random() < self.logging_prob:
                for log_value in args:
                    assert isinstance(log_value, str)
                    with open(f"{self.logging_file}", "a") as f:
                        f.write("\n\n==============\n" + log_value)
                for log_name, log_value in kwargs.items():
                    assert isinstance(log_value, str)
                    with open(f"{self.logging_dir}/{log_name}.txt", "a") as f:
                        f.write("\n\n==============\n" + log_value)

    def log_metric(self, **kwargs):
        logged_dict = {}
        for log_name, log_value in kwargs.items():
            if is_tensor(log_value):
                log_value = log_value.mean().item()

            elif isinstance(log_value, (list, tuple)):
                log_value = np.mean(log_value)
            else:
                log_value = float(log_value)
            logged_dict[log_name] = log_value
            if self.accelerator.is_main_process:
                self._metrics[log_name].append(log_value)
        return logged_dict

    def obtain_vllm_completions(
        self,
        inputs,
    ):

        raise NotImplementedError


class TeacherReward(abc.ABC):
    def link_with_trainer(
            self, trainer, student_model, teacher_model, tokenizer,):
        self.__name__ = self.__class__.__name__
        self.trainer: TeacherTrainer = trainer
        self.student_model: torch.Module = student_model
        self.teacher_model: torch.Module = teacher_model
        self.tokenizer: PreTrainedTokenizer = tokenizer

    def _make_normalize_fn(
            self, normalize_fn, temp=1, clip_min=None, clip_max=None):
        if isinstance(normalize_fn, str):
            normalize_fn = normalize_fn.lower()
        elif isinstance(normalize_fn, Callable):
            return normalize_fn

        def apply_clipping(x):
            if clip_min is not None:
                x = torch.clamp(x, min=clip_min)
            if clip_max is not None:
                x = torch.clamp(x, max=clip_max)
            return x

        if normalize_fn is None or normalize_fn == 'none':
            def f(x):
                return apply_clipping(x / temp)
        elif normalize_fn == 'exp':
            def f(x):
                return apply_clipping(torch.exp(x / temp))
        elif normalize_fn == 'exp_norm':
            def f(x):
                return apply_clipping(1 - torch.exp(-x / temp))
        else:
            raise NotImplementedError
        return f

    def _make_reduction_fn(self, reduction_fn, function_log_name=None):
        if isinstance(reduction_fn, Callable):
            if function_log_name is not None:
                log_names_to_store = [function_log_name + '_custom']
                return reduction_fn, log_names_to_store
            return reduction_fn

        def _flatten(seq):
            for i in seq:
                if isinstance(i, Sequence) and not isinstance(i, str):
                    yield from _flatten(i)
                else:
                    yield i

        if isinstance(reduction_fn, str):
            ops = [reduction_fn.lower()]
        elif isinstance(reduction_fn, Sequence):
            ops = [op.lower() for op in _flatten(reduction_fn)]
        else:
            raise NotImplementedError

        def f(x, mask):
            out = []
            for op in ops:
                try:
                    if op == 'mean':
                        o = torch.sum(x * mask, dim=-1) / \
                            torch.sum(mask, dim=-1)
                    elif op == 'sum':
                        o = torch.sum(x * mask, dim=-1)
                    elif op == 'min':
                        tmp = x.masked_fill(mask == 0,
                                            torch.finfo(x.dtype).max)
                        o = torch.min(tmp, dim=-1).values
                    elif op == 'max':
                        tmp = x.masked_fill(mask == 0,
                                            torch.finfo(x.dtype).min)
                        o = torch.max(tmp, dim=-1).values
                    elif op == 'median':
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanmedian(tmp, dim=-1).values
                    elif op == 'first_quartile':
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanquantile(tmp.float(),
                                              0.25, dim=-1)
                    elif op == 'last_quartile':
                        tmp = x.masked_fill(mask == 0, float('nan'))
                        o = torch.nanquantile(tmp.float(),
                                              0.75, dim=-1)
                    else:
                        raise NotImplementedError
                except Exception:

                    print(f'Invalid completion when reducing {x}...')
                    o = torch.full(
                        x.shape[:-1], float('nan'),
                        dtype=x.dtype, device=x.device)
                out.append(o)
            return torch.stack(out, dim=-1)

        if function_log_name is not None:
            log_names_to_store = [
                function_log_name + '_' + op for op in ops]
            return f, log_names_to_store
        return f

    def _make_elementwise_normalize_fn(self, normalize_fn, temp=1):

        if normalize_fn is None:
            return lambda x: x / temp
        if callable(normalize_fn):
            return normalize_fn

        import torch

        def build_transform(fn, t):
            if fn is None or fn == 'none':
                return lambda x: x / t
            elif fn == 'exp':
                return lambda x: torch.exp(x / t)
            elif fn == 'exp_norm':
                return lambda x: 1 - torch.exp(-x / t)
            elif fn == 'sym_log':
                return lambda x: torch.where(
                    x > 0,
                    torch.log(x / t),
                    torch.where(x < 0, -torch.log(-x / t),
                                torch.zeros_like(x))
                )
            else:
                raise NotImplementedError

        if isinstance(normalize_fn, list):
            trans_list = []
            for fn in normalize_fn:
                if isinstance(fn, str):
                    fn = fn.lower()
                trans_list.append(build_transform(fn, temp))

            def f(x):
                if x.shape[-1] != len(trans_list):
                    raise ValueError()
                channels = [trans_list[i](x[..., i])
                            for i in range(x.shape[-1])]
                return torch.stack(channels, dim=-1)
            return f
        elif isinstance(normalize_fn, str):
            return build_transform(normalize_fn.lower(), temp)
        else:
            raise TypeError()

    @abc.abstractmethod
    def __call__(
        self,
        prompts,
        completions,
        **kwargs,
    ):
        raise NotImplementedError
