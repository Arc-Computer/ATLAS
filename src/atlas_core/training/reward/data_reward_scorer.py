import abc
import os
import pickle
from tqdm import tqdm
import random
import numpy as np
import copy
from omegaconf import DictConfig, OmegaConf
import hydra
import wandb
import torch
import torch.nn.functional as F
from collections import defaultdict
from datetime import timedelta
from dataclasses import dataclass, field
from unittest.mock import patch
from typing import Any, Callable, Dict, List, Optional, Union
from collections import defaultdict, Counter

from accelerate.utils import broadcast_object_list, gather, gather_object


from torch import distributed as dist, nn
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedTokenizer,
    Trainer,
    TrainingArguments,
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
import accelerate
from transformers.trainer_utils import TrainOutput
from torch.utils.data import Subset


from datasets import Dataset, IterableDataset
from ..utils.trl_15 import prepare_deepspeed
from ..algorithms.teacher_base import TeacherReward, TeacherTrainer



def get_top_hf_dataset(dataset, score_column, number_to_keep, reverse=False):

    scores = dataset[score_column]

    sorted_indices = sorted(range(len(dataset)),
                            key=lambda i: np.max([s.item() for s in scores[i]])
                            if hasattr(scores[i], "item") else scores[i],
                            reverse=True)
    if reverse:
        top_indices = sorted_indices[-number_to_keep:]
    else:
        top_indices = sorted_indices[:number_to_keep]

    top_indices.sort()

    return dataset.select(top_indices)


def get_top_dataset(dataset, score_column, number_to_keep):

    scores = []
    for i in range(len(dataset)):
        item = dataset[i]
        score = item[score_column]
        if torch.is_tensor(score):
            score = score.item()
        scores.append(score)

    sorted_indices = sorted(range(len(scores)),
                            key=lambda i: scores[i],
                            reverse=True)
    top_indices = sorted_indices[:number_to_keep]

    top_indices = sorted(top_indices)

    return Subset(dataset, top_indices)


def instantiate_from_target(cfg, **kwargs):

    if not isinstance(cfg, DictConfig):
        cfg = OmegaConf.create(cfg)

    if 'target' in cfg and '_target_' not in cfg:
        cfg['_target_'] = cfg.pop('target')

    if '_target_' not in cfg:
        raise KeyError

    return hydra.utils.instantiate(cfg, **kwargs)


def gather_lists(local_list, accelerator, log_name=None):

    gathered_lists = accelerator.gather_object(local_list)

    combined = []
    for lst in gathered_lists:
        combined.extend(lst)
    return combined


def value_frequencies(values):
    total = len(values)
    counts = Counter(values)
    return {f'freq_{v}': (c / total) * 100 for v, c in counts.items()}


def get_mean_std_max_min_dict(array, prefix):
    res = {}
    res[prefix + '/mean'] = np.mean(array)
    res[prefix + '/std'] = np.std(array)
    res[prefix + '/min'] = np.amin(array)
    res[prefix + '/max'] = np.amax(array)
    return res


def is_unique_sequential(lst, orig_idx=None):
    idxs_for_seq = range(len(lst))
    if orig_idx is not None:
        idxs_for_seq = [orig_idx[i] for i in idxs_for_seq]
    return set(lst) == set(idxs_for_seq)


def list_of_dicts_to_dict_of_lists(lst):
    result = defaultdict(list)
    for d in lst:
        for key, value in d.items():
            result[key].append(value)
    return dict(result)


def merge_samples(old_dataset: Dataset, new_samples: List[dict]):

    dataset_dict = {sample["__index"]: sample for sample in old_dataset}
    for new_sample in new_samples:
        new_index = new_sample["__index"]
        if new_index in dataset_dict:
            dataset_dict[new_index]["completions"] += new_sample["completions"]
        else:
            dataset_dict[new_index] = new_sample
    return list(dataset_dict.values())


def save_pickle(fname, directory, **kwargs):
    if not os.path.exists(directory):
        os.makedirs(directory)
    file_path = os.path.join(directory, f"{fname}.pickle")
    with open(file_path, "wb") as f:
        pickle.dump(kwargs, f)


def load_pickle(fname, directory, *args):
    file_path = os.path.join(directory, f"{fname}.pickle")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return tuple(data[arg] for arg in args if arg in data)


@dataclass
class DataScorerArgs(TrainingArguments):

    ranked_data_dir: Optional[str] = None
    generated_scores_fname: str = 'reward_scores'
    retrieve_scores_from: Optional[str] = None
    peft_config: Optional[Dict[str, Any]] = field(default=None)
    score_from_split: str = 'train'
    target_prompt_column: str = field(default="prompt")
    target_completions_column: str = field(default="completions")
    update_column: Optional[str] = field(default="completion")
    rewards_column: Optional[str] = field(default="completions_rewards")

    store_raw_scores: bool = True
    seed: int = 42
    temperature: float = field(default=0.9,)
    unbias_log_probabilities: bool = field(default=True,)
    activate_debugging_logs: bool = field(default=False,)

    limit_scoring_samples: Optional[int] = None
    formatted_entry_idx: Optional[int] = None


class DataTeacherRewardScorer(Trainer, TeacherTrainer):
    def __init__(
            self,

            model: Union[str, PreTrainedModel],
            args: DataScorerArgs,
            reward_funcs: List[TeacherReward],
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset,
                                         dict[str, Union[Dataset, IterableDataset]]]] = None,
            processing_class: Optional[PreTrainedTokenizerBase] = None,
            reward_funcs_names: Optional[List[str]] = None,
            student_model: Union[str, PreTrainedModel] = None,
            teacher_model_init_kwargs=None,
            student_model_init_kwargs=None,
            logging_prob=0.0,
            offload_models=False,
            **kwargs):

        if teacher_model_init_kwargs is None:
            teacher_model_init_kwargs = {}
        if student_model_init_kwargs is None:
            student_model_init_kwargs = {}

        self._metrics = defaultdict(list)
        self.args = args
        self.seed = self.args.seed
        self.reward_funcs = reward_funcs
        self.formatted_entry_idx = args.formatted_entry_idx

        if self.formatted_entry_idx is not None:
            assert int(self.formatted_entry_idx) == self.formatted_entry_idx
            self.formatted_entry_idx = int(self.formatted_entry_idx)
            self.has_reserved_completion_index = True
        else:
            self.has_reserved_completion_index = False

        self.unbias_log_probabilities = args.unbias_log_probabilities
        self.gen_temperature = args.temperature

        if self.unbias_log_probabilities:
            assert self.gen_temperature > 0.0

        assert len(self.reward_funcs) > 0
        assert len(self.reward_funcs) == 1

        self.reward_funcs_names = (
            reward_funcs_names if reward_funcs_names
            else [reward_func.__name__ for reward_func in self.reward_funcs]
        )
        assert len(self.reward_funcs) == len(self.reward_funcs_names)

        self.peft_config = args.peft_config
        self.score_from_split = args.score_from_split
        if self.score_from_split == 'train':
            self.dataset = train_dataset
        elif self.score_from_split in ['val', 'eval', 'test']:
            self.dataset = eval_dataset
        else:
            raise NotImplementedError

        self.target_prompt_column = args.target_prompt_column
        self.target_completions_column = args.target_completions_column
        self.update_column = args.update_column
        self.rewards_column = args.rewards_column
        self.store_raw_scores = args.store_raw_scores
        self.set_seed(seed=self.seed)
        self.create_accelerator_and_postprocess()

        self.processing_class = processing_class

        teacher_model = model
        if isinstance(teacher_model, str):
            self.teacher_model = AutoModelForCausalLM.from_pretrained(
                teacher_model, **teacher_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.teacher_model = prepare_deepspeed(
                    self.teacher_model,
                    self.accelerator,
                    offload_to_cpu=offload_models)
            else:
                self.teacher_model = self.accelerator.prepare_model(
                    self.teacher_model, evaluation_mode=True)

                if offload_models:
                    self.teacher_model = accelerate.cpu_offload(
                        model=self.teacher_model)
            if processing_class is None:
                self.processing_class = AutoTokenizer.from_pretrained(
                    model, padding_side="left")
        else:

            raise NotImplementedError
            self.teacher_model = teacher_model

        assert self.processing_class is not None
        self.processing_class.padding_side = 'left'

        if student_model is None:

            self.student_model = self.ref_model
        if isinstance(student_model, str):
            self.student_model = AutoModelForCausalLM.from_pretrained(
                student_model, **student_model_init_kwargs)
            if self.is_deepspeed_enabled:
                self.student_model = prepare_deepspeed(
                    self.student_model,
                    self.accelerator,
                    offload_to_cpu=offload_models)
            else:
                self.student_model = self.accelerator.prepare_model(
                    self.student_model, evaluation_mode=True)

                if offload_models:
                    self.student_model = accelerate.cpu_offload(
                        model=self.student_model)
        else:

            raise NotImplementedError
            self.student_model = student_model

        TeacherTrainer.__init__(
            self,
            student_model=self.student_model,
            teacher_model=self.teacher_model,
            tokenizer=self.processing_class,
            reward_functions=self.reward_funcs,
            output_dir=self.args.output_dir,
            logging_prob=logging_prob,
        )

        self.ranked_data_dir = args.ranked_data_dir
        os.makedirs(self.ranked_data_dir, exist_ok=True)
        self.generated_scores_fname = args.generated_scores_fname
        self.full_generated_file_path = os.path.join(
            self.ranked_data_dir, f"{self.generated_scores_fname}.pickle")

        self.reward_fn = self.reward_funcs[0]

        self.cached_reward_raw_tensors = None

        self.retrieve_scores_from = self.args.retrieve_scores_from
        if self.retrieve_scores_from is None:
            self.retrieve_scores_from = self.ranked_data_dir

        self.out_file_path = os.path.join(self.ranked_data_dir, "ranked.json")

        check_picke_file = os.path.join(
            self.retrieve_scores_from, f"{self.generated_scores_fname}.pickle")
        if os.path.exists(check_picke_file):
            self.cached_reward_raw_tensors = load_pickle(
                self.generated_scores_fname,
                self.retrieve_scores_from,
                'cached_reward_raw_tensors',
            )
            if isinstance(self.cached_reward_raw_tensors, tuple) and len(
                    self.cached_reward_raw_tensors) == 1:
                self.cached_reward_raw_tensors = (
                    self.cached_reward_raw_tensors[0])
            print('Loaded cached reward tensor!')

        self.limit_scoring_samples = self.args.limit_scoring_samples
        self.limited_scoring_run = self.limit_scoring_samples is not None

    def _print_for_process(self, to_print: str, only_main=False):
        if only_main:
            if self.accelerator.is_main_process:
                print(f'Main process: {to_print}')
        else:
            print(f'Process {self.accelerator.process_index}: {to_print}')

    def _print_debugging_logs(self, to_print: str):
        if self.args.activate_debugging_logs:
            print(f'Process {self.accelerator.process_index}: {to_print}')

    def set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def rank_prompt(self, sample_idx, log_completion=False):
        sample = self.dataset[sample_idx]
        prompt = sample[self.args.target_prompt_column]
        completions = sample[self.args.target_completions_column]
        num_completions = len(completions)

        sample_true_idx = sample.get('__index', sample_idx)
        reward_kwargs = {}

        for k in sample.keys():
            if k not in [self.args.target_prompt_column, 'prompts',
                         self.args.target_completions_column, 'completions',
                         'completion', 'prompt']:
                reward_kwargs[k] = [sample[k]]
        if self.cached_reward_raw_tensors is not None:
            cached_tensors_for_completions = self.cached_reward_raw_tensors[
                sample_idx]
            assert cached_tensors_for_completions[0][
                '__index'] == sample_true_idx
            assert len(cached_tensors_for_completions) == len(completions)
        else:
            cached_tensors_for_completions = [
                {} for _ in range(num_completions)]
            assert True
        rws = []
        rw_infos = []
        if isinstance(completions, str):
            completions = [completions]
        for i, completion in enumerate(completions):

            cached_tensors = {
                f'cached_{k}': [v] for k, v in
                cached_tensors_for_completions[i].items()}
            reward_kwargs.update(cached_tensors)

            rw, rw_info, raw_tensors = self.reward_fn(
                prompts=[prompt],
                completions=[completion],
                **reward_kwargs,
                return_info_dict=True,
                return_raw_tensors=self.store_raw_scores,
            )

            cached_tensors_for_completions[i].update(raw_tensors[0])
            cached_tensors_for_completions[i]['rw'] = rw
            cached_tensors_for_completions[i]['__index'] = sample_true_idx
            rws.append(rw[0])
            rw_infos.extend(rw_info)
        return rws, rw_infos, cached_tensors_for_completions

    def train(self, resume_from_checkpoint=None):
        all_data_prompts = self.dataset[self.args.target_prompt_column]
        data_completions = self.dataset[self.args.target_completions_column]

        data_prompts = all_data_prompts

        all_num_samples = len(data_prompts)
        if self.limited_scoring_run:
            all_num_samples = min(all_num_samples, self.limit_scoring_samples)

        all_sample_indices = list(range(all_num_samples))

        num_processes = self.accelerator.num_processes

        process_batch_size = (
            all_num_samples + num_processes - 1)//num_processes

        perm = np.random.permutation(len(all_sample_indices))
        all_sample_indices_shuffled = [all_sample_indices[i] for i in perm]
        all_sample_idxs_split = [
            all_sample_indices_shuffled[
                process_idx*process_batch_size:
                (process_idx + 1)*process_batch_size]
            for process_idx in range(num_processes)
        ]
        inv_perm = np.argsort(perm)

        all_sample_idxs_split = broadcast_object_list(
            object_list=all_sample_idxs_split, from_process=0)
        inv_perm = broadcast_object_list(
            object_list=inv_perm, from_process=0)

        first_sample_idxs = all_sample_idxs_split[0]

        sample_idxs = all_sample_idxs_split[self.accelerator.process_index]

        if len(first_sample_idxs) > len(sample_idxs):
            self.padded_idxs = len(first_sample_idxs) - len(sample_idxs)
            sample_idxs.extend(first_sample_idxs[-self.padded_idxs:])
        else:
            self.padded_idxs = 0

        if len(sample_idxs) > 0:
            self._print_for_process(
                f'Ranking prompts {sample_idxs[0]}-{sample_idxs[-1]}',
                only_main=False)

        num_samples = len(sample_idxs)
        all_completion_rewards = []
        best_completions = []
        line_contents = []
        all_best_completion_indices = []
        cached_raw_tensors_to_store = [{} for _ in range(num_samples)]
        if self.cached_reward_raw_tensors is not None:
            cached_raw_tensors_to_store = [
                self.cached_reward_raw_tensors[ix] for ix in sample_idxs]
            assert len(cached_raw_tensors_to_store) == num_samples, (
                f'Cached length {len(cached_raw_tensors_to_store)} does not '
                f'match data length {num_samples}')

        for process_prompt_idx, sample_idx in enumerate(sample_idxs):
            self._print_for_process(
                f'Sample {process_prompt_idx+1}/{num_samples}', only_main=True)
            rws, rw_infos, cached_tensors_for_completions = self.rank_prompt(
                sample_idx=sample_idx, log_completion=True)
            completions = data_completions[sample_idx]
            sample = self.dataset[sample_idx]
            if '__index' not in sample:
                sample['__index'] = sample_idx
            cached_raw_tensors_to_store[
                process_prompt_idx] = cached_tensors_for_completions
            all_completion_rewards.append(rws)
            if self.has_reserved_completion_index:
                formatted_entry_idx = self.formatted_entry_idx
                if formatted_entry_idx < 0:
                    formatted_entry_idx = formatted_entry_idx + len(rws)
                rws_wo_reserved_idx = [rew_t for rew_t_i, rew_t in enumerate(
                    rws) if rew_t_i != formatted_entry_idx]
                best_completion_idx = np.nanargmax(rws_wo_reserved_idx)

                format_value_best = rw_infos[
                    best_completion_idx]["match_reward"]
                if format_value_best < 0:
                    best_completion_idx = formatted_entry_idx
            else:
                best_completion_idx = np.nanargmax(rws)
            all_best_completion_indices.append(best_completion_idx)
            best_completion = completions[best_completion_idx]
            best_completions.append(best_completion)

            line_content = {**sample}
            new_line_contents = {
                self.update_column: best_completion,
                self.rewards_column: rws,
            }
            line_content.update(new_line_contents)
            line_contents.append(line_content)

        if self.padded_idxs > 0:
            all_completion_rewards = all_completion_rewards[:-self.padded_idxs]
            best_completions = best_completions[:-self.padded_idxs]
            all_best_completion_indices = all_best_completion_indices[
                :-self.padded_idxs]
            line_contents = line_contents[:-self.padded_idxs]
            cached_raw_tensors_to_store = cached_raw_tensors_to_store[
                :-self.padded_idxs]

        self.accelerator.wait_for_everyone()
        self._print_for_process(
            'Gathering all_completion_rewards...', only_main=True)
        all_completion_rewards = gather_object(all_completion_rewards)

        all_completion_rewards = [all_completion_rewards[i] for i in inv_perm]
        self._print_for_process(f"{all_completion_rewards}", only_main=True)

        self._print_for_process(
            f'Number of gathered rws: {len(all_completion_rewards)}',
            only_main=True)

        self._print_for_process(
            'Gathering best_completions...', only_main=True)
        best_completions = gather_object(best_completions)
        best_completions = [best_completions[i] for i in inv_perm]

        self._print_for_process(
            'Gathering line_contents...', only_main=True)
        line_contents = gather_object(line_contents)
        line_contents = [line_contents[i] for i in inv_perm]
        test_idx = 27
        if test_idx is not None and test_idx < all_num_samples:
            self._print_for_process(
                'Comparing idxs sanity check: ', only_main=True)
            self._print_for_process(
                line_contents[test_idx]['__index'], only_main=True)
            self._print_for_process(
                self.dataset[test_idx].get('__index', test_idx), only_main=True)

        self._print_for_process(
            'Gathering all_best_completion_indices...', only_main=True)
        all_best_completion_indices = gather_object(
            all_best_completion_indices)
        all_best_completion_indices = [
            all_best_completion_indices[i] for i in inv_perm]

        self._print_for_process(
            'Gathering cached_raw_tensors_to_store...', only_main=True)
        cached_raw_tensors_to_store = gather_object(
            cached_raw_tensors_to_store)
        cached_raw_tensors_to_store = [
            cached_raw_tensors_to_store[i] for i in inv_perm]
        completion_wins = value_frequencies(values=all_best_completion_indices)

        if self.accelerator.is_main_process:
            print('Win index percentages: ')
            print(completion_wins)
            print('Storing data to disk...')
            ds = Dataset.from_list(line_contents)

            ds.to_json(self.out_file_path)

            if self.store_raw_scores:
                print('Storing pickle of raw tensors...')
                save_pickle(
                    fname=self.generated_scores_fname,
                    directory=self.ranked_data_dir,
                    cached_reward_raw_tensors=cached_raw_tensors_to_store,
                )

        self.accelerator.wait_for_everyone()
        if wandb.run is not None:
            res_to_log = completion_wins
            res_to_log['step'] = 1
            wandb.log(res_to_log)
        self._print_for_process(
            f'Ranking complete, results stored at {self.out_file_path}',
            only_main=False)

        return TrainOutput(
            global_step=1, training_loss=0.0, metrics=completion_wins)

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val)
                   for key, val in self._metrics.items()}

        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        super().log(logs)

        self._metrics.clear()

    def log_metrics(self, split, metrics):
        pass

    def save_metrics(self, split, metrics):
        pass

    def save_state(self):
        pass

    def save_model(self, output_dir):
        pass

    def create_model_card(self, metadata):
        pass

    def push_to_hub(self):
        pass


@dataclass
class DataConcatenatorArgs(TrainingArguments):

    concatenated_data_dir: Optional[str] = None
    target_completions_column: str | List[str] = field(default="completions")
    concatenated_completions_column: str = field(default="completions")


class DataCompletionConcatenator(Trainer, TeacherTrainer):
    def __init__(
            self,
            args: DataConcatenatorArgs = None,
            train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
            eval_dataset: Optional[Union[Dataset, IterableDataset,
                                         dict[str, Union[Dataset, IterableDataset]]]] = None,
            tokenizer: Optional[PreTrainedTokenizerBase] = None,
            datasets_to_concatenate: List[
                Dataset | IterableDataset | DictConfig] = None,
            **kwargs):

        self.args = args
        self.concatenated_data_dir = self.args.concatenated_data_dir
        self.out_file_path = os.path.join(
            self.concatenated_data_dir, "merged.json")
        self.target_completions_column = []
        self.concatenated_completions_column = (
            self.args.concatenated_completions_column)
        if isinstance(self.args.target_completions_column, str):
            self.target_completions_column = [
                self.args.target_completions_column
                for _ in datasets_to_concatenate]
        else:
            self.target_completions_column = self.args.target_completions_column
        assert len(datasets_to_concatenate) == len(
            self.target_completions_column)
        self.datasets_to_concatenate = []
        if isinstance(tokenizer, (dict, DictConfig)):
            tokenizer = instantiate_from_target(tokenizer)
        for data, col in zip(
                datasets_to_concatenate, self.target_completions_column):
            if isinstance(data, (Dataset, IterableDataset)):
                inst_data = data
            else:
                assert isinstance(data, (DictConfig, dict)), f'{type(data)}'
                inst_data = instantiate_from_target(data, tokenizer=tokenizer)
            inst_data = inst_data['train_dataset']
            print(inst_data)
            print(col)
            assert col in inst_data.column_names
            self.datasets_to_concatenate.append(inst_data)

        self.dataset = self.datasets_to_concatenate[0]
        self.dataset_completions = [data[compl_col] for data, compl_col in zip(
            self.datasets_to_concatenate, self.target_completions_column)]
        self.data_len = len(self.dataset)
        print(f'target data len {self.data_len}')
        for data in self.datasets_to_concatenate:
            print(f'other data len {len(data)}')
            assert len(data) == self.data_len, (
                "Trying to concatenate different data sizes")
        self.all_data_indices = [
            data['__index'] for data in self.datasets_to_concatenate if
            '__index' in data]

    def get_all_completions_for_idx(self, idx):
        all_completions = []
        for dataset_comp in self.dataset_completions:
            dataset_comp_idx = dataset_comp[idx]
            if isinstance(dataset_comp_idx, str):
                all_completions.append(dataset_comp_idx)
            else:
                assert isinstance(dataset_comp_idx, (tuple, list)), (
                    f'{type(dataset_comp_idx)}')
                all_completions.extend(list(dataset_comp_idx))
        return all_completions

    def train(self, resume_from_checkpoint=None):

        line_contents = []

        all_num_original_completions = []
        all_num_new_completions = []
        for sample_idx in tqdm(range(self.data_len)):
            completions = self.dataset_completions[0][sample_idx]
            num_completions = len(completions)
            all_num_original_completions.append(num_completions)
            sample = self.dataset[sample_idx]
            new_completions = self.get_all_completions_for_idx(idx=sample_idx)
            num_new_completions = len(new_completions)
            all_num_new_completions.append(num_new_completions)
            if '__index' in sample:
                sample_true_idx = sample['__index']
            line_content = {**sample}
            line_content.update(
                {self.concatenated_completions_column: new_completions})
            line_contents.append(line_content)

        mean_num_original_completions = np.mean(all_num_original_completions)
        mean_num_new_completions = np.mean(all_num_new_completions)

        print('Made new dataset with '
              f'{mean_num_original_completions}-->'
              f'{mean_num_new_completions} average completions')

        ds = Dataset.from_list(line_contents)

        ds.to_json(self.out_file_path)

        print(f'Merging complete, results stored at {self.out_file_path}')
        return TrainOutput(global_step=1, training_loss=0.0, metrics={})

    def log_metrics(self, split, metrics):
        pass

    def save_metrics(self, split, metrics):
        pass

    def save_state(self):
        pass

    def save_model(self, output_dir):
        pass

    def create_model_card(self, metadata):
        pass

    def push_to_hub(self):
        pass
