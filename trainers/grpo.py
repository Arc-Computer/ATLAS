import os
import gc
import time
import csv
import textwrap
import warnings
from collections import defaultdict
from typing import Any, Callable, Optional, Sized, Union
from unittest.mock import patch

import torch
import torch.utils.data
import transformers
from accelerate import PartialState

from accelerate.utils import broadcast_object_list, gather, gather_object, is_peft_model, set_seed
from accelerate.utils.other import is_compiled_module
import accelerate
from datasets import Dataset, IterableDataset
from packaging import version
from torch import nn
import torch.nn.functional as F
import numpy as np
import concurrent.futures
import deepspeed
from contextlib import nullcontext
from torch.utils.data import Sampler
from transformers import (
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
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available


from trl.data_utils import apply_chat_template, is_conversational, maybe_apply_chat_template
from trl.import_utils import is_vllm_available
from trl.models import create_reference_model
from .utils_trl_15 import prepare_deepspeed, unwrap_model_for_generation
from trl.trainer.utils import generate_model_card, get_comet_experiment_url, pad
from trl.trainer.callbacks import SyncRefModelCallback
from .grpo_config import GRPOConfig


if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_vllm_available():
    from vllm import LLM, SamplingParams

if is_wandb_available():
    import wandb

_VLLM_IMPORT_ERROR = None
try:
    from .custom_ray import RayGeneratorActor
    from .custom_ray.vllm_engine import get_resource_info, print_ram_utilization
except Exception as exc:  # pragma: no cover - optional dependency
    RayGeneratorActor = None
    _VLLM_IMPORT_ERROR = exc

    def get_resource_info(*args, **kwargs):  # type: ignore
        raise exc

    def print_ram_utilization(*args, **kwargs):  # type: ignore
        raise exc

from .vllm_client import VLLMClient


def selective_log_softmax(logits, index):

    if logits.dtype in [torch.float32, torch.float64]:
        selected_logits = torch.gather(
            logits, dim=-1, index=index.unsqueeze(-1)).squeeze(-1)

        logsumexp_values = torch.stack(
            [torch.logsumexp(lg, dim=-1) for lg in logits])
        per_token_logps = selected_logits - logsumexp_values
    else:

        per_token_logps = []
        for row_logits, row_labels in zip(logits, index):
            row_logps = F.log_softmax(row_logits, dim=-1)
            row_per_token_logps = row_logps.gather(
                dim=-1, index=row_labels.unsqueeze(-1)).squeeze(-1)
            per_token_logps.append(row_per_token_logps)
        per_token_logps = torch.stack(per_token_logps)
    return per_token_logps


RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]


class RepeatRandomSampler(Sampler):
    def __init__(
        self,
        data_source: Sized,

        mini_repeat_count: int,

        batch_size: int = 1,


        repeat_count: int = 1,


        artificial_epochs: int = 1,

        indexes_to_split_before_repeat=1,
        seed: Optional[int] = None,
    ):
        self.data_source = data_source
        self.mini_repeat_count = mini_repeat_count
        self.batch_size = batch_size
        self.repeat_count = repeat_count

        self.num_samples = len(data_source)
        self.seed = seed
        assert int(artificial_epochs) == artificial_epochs
        self.artificial_epochs = int(artificial_epochs)
        assert self.artificial_epochs > 0
        self.generator = torch.Generator()
        if seed is not None:
            self.generator.manual_seed(seed)

    def __iter__(self):

        indexes = torch.randperm(
            self.num_samples, generator=self.generator).tolist()
        if self.artificial_epochs > 1:
            for _ in range(self.artificial_epochs - 1):
                new_indexes = torch.randperm(
                    self.num_samples, generator=self.generator).tolist()
                indexes.extend(new_indexes)

        indexes = [indexes[
            i: i + self.batch_size]
            for i in range(0, len(indexes), self.batch_size)]

        indexes = [chunk for chunk in indexes if len(chunk) == self.batch_size]

        for chunk in indexes:

            for _ in range(self.repeat_count):
                for index in chunk:

                    for _ in range(self.mini_repeat_count):

                        yield index

    def __len__(self) -> int:
        return self.num_samples * self.mini_repeat_count * self.repeat_count * self.artificial_epochs


class GRPOTrainer(Trainer):

    _tag_names = ["trl", "grpo"]

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[Union[Dataset, IterableDataset,
                                     dict[str, Union[Dataset, IterableDataset]]]] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[Union[PreTrainedTokenizerBase,
                                                  list[PreTrainedTokenizerBase]]] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[Optional[torch.optim.Optimizer],
                          Optional[torch.optim.lr_scheduler.LambdaLR]] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
    ):

        if args is None:
            model_name = model if isinstance(
                model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        artificial_epochs = args.artificial_epochs
        if artificial_epochs is None or artificial_epochs <= 0:
            artificial_epochs = 1
        self.artificial_epochs = artificial_epochs

        model_init_kwargs = args.model_init_kwargs or {}
        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if isinstance(torch_dtype, torch.dtype) or torch_dtype == "auto" or torch_dtype is None:
                pass
            elif isinstance(torch_dtype, str):
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )

            model_init_kwargs["use_cache"] = (
                False if args.gradient_checkpointing else model_init_kwargs.get(
                    "use_cache")
            )
            model = AutoModelForCausalLM.from_pretrained(
                model, **model_init_kwargs)
        else:
            model_id = model.config._name_or_path
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        if args.gradient_checkpointing:
            model = self._enable_gradient_checkpointing(model, args)

        if is_deepspeed_zero3_enabled():
            self.ref_model = AutoModelForCausalLM.from_pretrained(
                model_id, **model_init_kwargs)
        elif not is_peft_model(model):

            self.ref_model = create_reference_model(model)
        else:

            self.ref_model = None

        if processing_class is None:
            processing_class = AutoTokenizer.from_pretrained(
                model.config._name_or_path, padding_side="left")

        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )

        self._stored_model_init_kwargs = model_init_kwargs
        self.reward_funcs = reward_funcs

        if args.reward_weights is not None:
            if len(args.reward_weights) != len(reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(
                args.reward_weights, dtype=torch.float32)
        else:
            self.reward_weights = torch.ones(
                len(reward_funcs), dtype=torch.float32)

        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions.")

        for i, (reward_processing_class, reward_func) in enumerate(zip(reward_processing_classes, reward_funcs)):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path)
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = reward_processing_class.eos_token

                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        def data_collator(features):
            return features

        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = args.max_completion_length
        self.num_generations = args.num_generations
        self.use_ray = args.use_ray
        self.ray_no_memory_duplication = args.ray_no_memory_duplication
        self.use_vllm = args.use_vllm
        self.temperature = args.temperature
        self.top_p = args.top_p
        self.top_k = args.top_k
        self.min_p = args.min_p
        self.repetition_penalty = args.repetition_penalty
        self.use_vllm_server = args.use_vllm_server
        self.vllm_host = args.vllm_host
        self.vllm_port = args.vllm_port

        self.beta = args.beta

        model.warnings_issued["estimate_tokens"] = True

        self._metrics = defaultdict(list)
        self.log_completions = args.log_completions
        self.save_completions_probability = args.save_completions_probability
        self.store_completions_to_file = False
        if self.save_completions_probability is not None:
            if self.save_completions_probability > 0:
                self.store_completions_to_file = True

        self.generation_aggregation_steps = args.generation_aggregation_steps
        gradient_accumulation = args.gradient_accumulation_steps

        if self.generation_aggregation_steps is not None:
            assert (
                gradient_accumulation % self.generation_aggregation_steps == 0)
        else:
            self.generation_aggregation_steps = 1

        self.different_accumulation_batches = (
            gradient_accumulation//self.generation_aggregation_steps)

        self.optim_pd_train_batch_size = args.per_device_train_batch_size

        args.per_device_train_batch_size = (
            self.optim_pd_train_batch_size *
            self.generation_aggregation_steps
        )

        self.buffered_completion_ids = None
        self.buffered_advantages = None
        self.buffered_prompts_text = None
        self._buffered_batch_idx = 0
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        if self.accelerator.is_main_process:
            print('Batch size input size for gen: '
                  f'{self.optim_pd_train_batch_size} x '
                  f'{self.generation_aggregation_steps} = '
                  f'{self.args.per_device_train_batch_size}')

        pd_orig_batch_size = args.per_device_train_batch_size
        self.backprop_accumulation_steps = args.backprop_accumulation_steps
        if self.backprop_accumulation_steps is None:
            if args.backprop_accumulation_micro_batch_size is not None:
                assert (pd_orig_batch_size %
                        args.backprop_accumulation_micro_batch_size) == 0
                self.backprop_accumulation_steps = (
                    pd_orig_batch_size//args.backprop_accumulation_micro_batch_size)
            else:
                self.backprop_accumulation_steps = 1
        elif args.backprop_accumulation_micro_batch_size is not None:
            assert (pd_orig_batch_size//args.backprop_accumulation_micro_batch_size
                    ) == self.backprop_accumulation_steps
        self.backprop_step_size = pd_orig_batch_size // self.backprop_accumulation_steps
        assert pd_orig_batch_size % self.backprop_accumulation_steps == 0

        if self.backprop_accumulation_steps > 1:

            pass

        num_processes = self.accelerator.num_processes
        global_batch_size = args.per_device_train_batch_size * num_processes

        if self.num_generations is None:

            self.num_generations = global_batch_size
            if self.accelerator.is_main_process:
                print(f"Warning, num_generations is None, setting to the "
                      f"global_batch_size={global_batch_size}")
        possible_values = [n_gen for n_gen in range(
            2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
        if self.num_generations not in possible_values:
            raise ValueError(
                f"The global train batch size ({num_processes} x {args.per_device_train_batch_size}) must be evenly "
                f"divisible by the number of generations per prompt ({self.num_generations}). Given the current train "
                f"batch size, the valid values for the number of generations are: {possible_values}."
            )
        if self.args.eval_strategy != "no":
            global_batch_size = args.per_device_eval_batch_size * num_processes
            possible_values = [n_gen for n_gen in range(
                2, global_batch_size + 1) if (global_batch_size) % n_gen == 0]
            if self.num_generations not in possible_values:
                raise ValueError(
                    f"The global eval batch size ({num_processes} x {args.per_device_eval_batch_size}) must be evenly "
                    f"divisible by the number of generations per prompt ({self.num_generations}). Given the current "
                    f"eval batch size, the valid values for the number of generations are: {possible_values}."
                )

        set_seed(args.seed, device_specific=True)
        if self.use_vllm or self.use_ray and (not self.use_vllm_server):
            if not is_vllm_available():
                raise ImportError(
                    "vLLM is not available and `use_vllm` is set to True. Please install vLLM with "
                    "`pip install vllm` to use it."
                )

            if self.accelerator.is_main_process:
                vllm_device = self.args.vllm_device
                if self.use_ray:
                    print(f'Used generation device: {vllm_device}')
                    pass
                else:
                    if vllm_device == "auto":
                        if torch.cuda.device_count() == 1:
                            vllm_device = "cuda:0"
                        else:
                            vllm_device = f"cuda:{self.accelerator.num_processes}"

                    if vllm_device.split(":")[0] == "cuda" and int(vllm_device.split(":")[1]) >= torch.cuda.device_count():
                        raise ValueError(
                            f"The requested device for vllm ({vllm_device}) is not available. You are likely using vLLM "
                            "without restricting the number of GPUs for training. Set the `--num_processes` argument to a "
                            "value lower than the number of GPUs available on your machine—typically, reducing it by one "
                            f"is sufficient. In your case: `--num_processes {torch.cuda.device_count() - 1}`."
                        )

                    if vllm_device in {f"cuda:{idx}" for idx in range(self.accelerator.num_processes)}:
                        warnings.warn(
                            f"The requested device {vllm_device} is also being used for training. For higher throughput "
                            "and to avoid out-of-memory errors, it is recommended to use a dedicated device for vLLM. "
                            "If this is intentional, you may ignore this warning but should adjust "
                            "`vllm_gpu_memory_utilization` accordingly."
                        )

                world_size_patch = patch(
                    "torch.distributed.get_world_size", return_value=1)
                profiling_patch = patch(
                    "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling", return_value=None
                )
                if self.use_ray:
                    ray_tensor_parallelism = self.args.ray_tensor_parallelism
                    self.ray_share_training_devices = (
                        self.args.ray_share_training_devices)
                    if self.ray_share_training_devices:
                        reserved_gpus = 0
                    else:
                        reserved_gpus = num_processes
                    world_size = torch.cuda.device_count() - reserved_gpus

                    print(f'Detected world size {world_size}')
                    if self.args.ray_data_parallelism is None:
                        self.ray_data_parallelism = world_size//ray_tensor_parallelism
                    else:
                        self.ray_data_parallelism = self.args.ray_data_parallelism
                    self.total_ray_devices = ray_tensor_parallelism*self.ray_data_parallelism
                    assert self.total_ray_devices <= world_size
                    print(
                        f'Using {self.ray_data_parallelism} x '
                        f'{ray_tensor_parallelism} = {self.total_ray_devices} '
                        'inference GPU devices')

                    self.sampling_params = SamplingParams(
                        max_tokens=self.max_completion_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=-1 if args.top_k is None else args.top_k,
                        min_p=0.0 if args.min_p is None else args.min_p,
                        repetition_penalty=args.repetition_penalty,
                    )
                    revision = None
                    self.ray_actor = RayGeneratorActor(
                        model=model.name_or_path,
                        revision=revision,
                        tokenizer=self.processing_class,
                        seed=self.args.seed,


                        ray_num_nodes=1,
                        ray_tensor_parallelism=ray_tensor_parallelism,
                        ray_data_parallelism=self.ray_data_parallelism,
                        vllm_gpu_memory_utilization=(
                            self.args.vllm_gpu_memory_utilization),
                        vllm_dtype=self.args.vllm_dtype,



                        enable_prefix_caching=self.args.enable_prefix_caching,
                        enforce_eager=self.args.enforce_eager,
                        sleep_level=self.args.vllm_sleep_level,




                        max_tokens=self.args.vllm_max_model_len,
                        max_completion_length=self.max_completion_length,

                        temperature=self.args.temperature,
                        top_k=-1 if args.top_k is None else args.top_k,
                        top_p=args.top_p,
                        repetition_penalty=self.args.repetition_penalty,


                        collective_rpc_mode='nccl',
                        reserved_gpus=reserved_gpus,
                        sampling_params=self.sampling_params,
                    )
                    self.state_dict_copy = None
                else:
                    with world_size_patch, profiling_patch:
                        self.llm = LLM(
                            model=model.name_or_path,
                            device=vllm_device,
                            gpu_memory_utilization=self.args.vllm_gpu_memory_utilization,
                            dtype=self.args.vllm_dtype,



                            enable_prefix_caching=True,
                            max_model_len=self.args.vllm_max_model_len,
                        )

                    self.sampling_params = SamplingParams(
                        max_tokens=self.max_completion_length,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        top_k=-1 if args.top_k is None else args.top_k,
                        min_p=0.0 if args.min_p is None else args.min_p,
                        repetition_penalty=args.repetition_penalty,
                    )

            self._last_loaded_step = 0

            self.accelerator.wait_for_everyone()
        elif self.use_vllm_server:
            if self.accelerator.is_main_process:
                self.vllm_clients = [VLLMClient(
                    host=args.vllm_host,
                    server_port=args.vllm_port + i,
                    group_port=args.vllm_group_port + i,
                    connection_timeout=args.vllm_server_timeout)
                    for i in range(args.num_vllm_clients)
                ]

            self.guided_decoding_regex = args.vllm_guided_decoding_regex

            self._last_loaded_step = 0

            self.accelerator.wait_for_everyone()
        else:
            self.generation_config = GenerationConfig(
                max_new_tokens=self.max_completion_length,
                do_sample=True,
                pad_token_id=processing_class.pad_token_id,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                min_p=args.min_p,
                repetition_penalty=args.repetition_penalty,
            )

        self.gen_temperature = args.temperature
        self.unbias_log_probabilities = args.unbias_log_probabilities
        if self.unbias_log_probabilities:
            assert self.gen_temperature > 0

        self.model_accepts_loss_kwargs = False

        self.model.add_model_tags(self._tag_names)

        self.offload_untrained_models = args.offload_untrained_models

        if self.ray_no_memory_duplication:
            raise NotImplementedError
        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(
                    self.ref_model,
                    self.accelerator,
                    offload_to_cpu=self.offload_untrained_models)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True)

                if self.offload_untrained_models:
                    self.ref_model = accelerate.cpu_offload(
                        model=self.ref_model)

        if args.sync_ref_model:
            self.add_callback(SyncRefModelCallback(
                ref_model=self.ref_model, accelerator=self.accelerator))

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                self.reward_funcs[i] = self.accelerator.prepare_model(
                    reward_func, evaluation_mode=True)
                if self.offload_untrained_models:
                    self.reward_funcs[i] = accelerate.cpu_offload(
                        model=self.reward_funcs[i])

        self._buffered_batch_idx = 0

        model.warnings_issued["estimate_tokens"] = True

    def _print_debugging_logs(self, to_print: str):
        if self.args.activate_debugging_logs:
            print(f'Process {self.accelerator.process_index}: {to_print}')

    def _set_signature_columns_if_needed(self):

        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    def _get_train_sampler(self) -> Sampler:

        effective_batch_size = (

            self.optim_pd_train_batch_size
            * self.accelerator.num_processes
            * self.generation_aggregation_steps
        )

        different_samples_per_batch = (
            effective_batch_size // self.num_generations)

        return RepeatRandomSampler(
            data_source=self.train_dataset,
            mini_repeat_count=self.num_generations,
            batch_size=different_samples_per_batch,

            repeat_count=self.generation_aggregation_steps,
            artificial_epochs=self.artificial_epochs,
            seed=self.args.seed,
        )

    def _get_eval_sampler(self, eval_dataset) -> Sampler:
        return RepeatRandomSampler(
            data_source=eval_dataset,
            mini_repeat_count=self.num_generations,
            seed=self.args.seed,
        )

    def _enable_gradient_checkpointing(self, model: PreTrainedModel, args: GRPOConfig) -> PreTrainedModel:

        model.config.use_cache = False

        if is_peft_model(model):
            model.base_model.gradient_checkpointing_enable()

        else:
            model.gradient_checkpointing_enable()

        gradient_checkpointing_kwargs = args.gradient_checkpointing_kwargs or {}
        use_reentrant = (
            "use_reentrant" not in gradient_checkpointing_kwargs or gradient_checkpointing_kwargs[
                "use_reentrant"]
        )

        if use_reentrant:
            model.enable_input_require_grads()

        return model

    def _get_per_token_logps(self, model, input_ids, attention_mask, logits_to_keep):
        self._print_debugging_logs('obtaining logits...')

        logits = model(input_ids=input_ids, attention_mask=attention_mask,
                       num_logits_to_keep=logits_to_keep + 1).logits
        logits = logits[:, :-1, :]

        input_ids = input_ids[:, -logits_to_keep:]

        logits = logits[:, -logits_to_keep:]
        if self.unbias_log_probabilities:
            logits = logits/self.gen_temperature
        self._print_debugging_logs('computing softmax...')
        return selective_log_softmax(logits, input_ids)

    def _pass_weights(self, param_name, param_data):
        if self.use_ray:

            if self.ray_no_memory_duplication:
                self.state_dict_copy.update(
                    {param_name, param_data.clone().detach().cpu()}
                )
            else:
                self.ray_actor.update_state_dict(
                    state_dict={param_name: param_data}, main_engine_idx=0,

                    clone_weight=True,
                )
        elif self.use_vllm_server:
            for client in self.vllm_clients:
                client.update_named_param(param_name, param_data)
        else:
            llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
            llm_model.load_weights([(param_name, param_data)])

    def _move_model_to_vllm(self):

        deepspeed_plugin = self.accelerator.state.deepspeed_plugin
        zero_stage_3 = deepspeed_plugin is not None and deepspeed_plugin.zero_stage == 3
        gather_if_zero3 = deepspeed.zero.GatheredParameters if zero_stage_3 else nullcontext
        self.state_dict_copy = {}

        if is_peft_model(self.model):

            with gather_if_zero3(list(self.model.parameters())):
                self.model.merge_adapter()

                for name, param in self.model.named_parameters():

                    name = name.removeprefix(
                        "base_model.model.").replace(".base_layer", "")
                    if self.model.prefix in name:
                        continue

                    if "original_module" in name:
                        continue
                    name = name.replace("modules_to_save.default.", "")

                    if self.accelerator.is_main_process:
                        self._pass_weights(name, param.data)

                self.model.unmerge_adapter()

        else:

            for name, param in self.model.named_parameters():
                with gather_if_zero3([param]):
                    if self.accelerator.is_main_process:
                        self._pass_weights(name, param.data)

        self.accelerator.wait_for_everyone()
        torch.cuda.synchronize()

        if self.accelerator.is_main_process and self.use_vllm_server:
            for client in self.vllm_clients:
                client.reset_prefix_cache()

    def _pre_model_moving(self,):
        if self.use_ray and self.accelerator.is_main_process:
            self.ray_actor.wake_if_needed()

    def _post_model_moving(self,):
        if self.use_ray and self.ray_no_memory_duplication:
            gc.collect()
            torch.cuda.empty_cache()
            if self.accelerator.is_main_process:
                if self.args.activate_debugging_logs:
                    print('resources before onloading state dict.')
                    res = get_resource_info()
                    print(res)
                    print_ram_utilization()
                    if ("wandb" in self.args.report_to) and (
                            self.state.global_step % self.args.logging_steps == 0):
                        res_to_log = {}
                        for k, k_dict in res.items():
                            for k_dict_k, k_dict_v in k_dict.items():
                                res_to_log[f'{k}/{k_dict_k}/before_onload'] = k_dict_v
                        res_to_log["step"] = self.state.global_step
                        if wandb.run is not None:
                            wandb.log(res_to_log)
                assert self.state_dict_copy is not None, (
                    'saved unwrapped state dict is None')
                self.ray_actor.update_state_dict(
                    state_dict=self.state_dict_copy, main_engine_idx=0,
                    clone_weight=False)
        if self.use_ray and self.accelerator.is_main_process:
            self.ray_actor.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    def _prepare_inputs(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device

        mode = "eval" if self.control.should_evaluate else "train"

        if mode == "eval":
            raise NotADirectoryError
        elif (self._buffered_batch_idx >= self.args.per_device_train_batch_size
              or self.buffered_completion_ids is None):
            if self.buffered_completion_ids is not None:
                assert (
                    self._buffered_batch_idx ==
                    self.args.per_device_train_batch_size)
            self._buffered_batch_idx = 0

            prompts_text, completion_ids, advantages = (
                self._generate_and_score(inputs=inputs))
            self.buffered_completion_ids = completion_ids
            self.buffered_advantages = advantages
            self.buffered_prompts_text = prompts_text
        else:

            prompts_text_for_check = [maybe_apply_chat_template(
                example, self.processing_class)["prompt"] for example in inputs]
            assert len(prompts_text_for_check) == len(
                self.buffered_prompts_text)
            self._print_debugging_logs('Checking prompt texts...')
            for i, (t, b_t) in enumerate(zip(
                    prompts_text_for_check, self.buffered_prompts_text)):
                assert t == b_t, f'idx {i}, {t} != {b_t}'
                pass
            pass

        end_batch_idx = (
            self._buffered_batch_idx + self.optim_pd_train_batch_size)

        prompts_text = self.buffered_prompts_text[
            self._buffered_batch_idx:end_batch_idx]
        completion_ids = self.buffered_completion_ids[
            self._buffered_batch_idx:end_batch_idx]
        advantages = self.buffered_advantages[
            self._buffered_batch_idx:end_batch_idx]

        self._buffered_batch_idx = end_batch_idx

        prompt_inputs = self.processing_class(
            prompts_text, return_tensors="pt", padding=True,
            padding_side="left", add_special_tokens=False
        )
        self._print_debugging_logs('preparing inputs...')
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        if self.max_prompt_length is not None:
            prompt_ids = prompt_ids[:, -self.max_prompt_length:]
            prompt_mask = prompt_mask[:, -self.max_prompt_length:]

        completion_ids = [
            torch.tensor(ids, device=device) for ids in completion_ids]
        completion_ids = pad(
            completion_ids, padding_value=self.processing_class.pad_token_id)
        prompt_completion_ids = torch.cat([prompt_ids, completion_ids], dim=1)

        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(
            1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[
            is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(
            1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)

        logits_to_keep = completion_ids.size(1)

        self._print_debugging_logs('reference model logprobs...')
        with torch.inference_mode():
            if self.ref_model is not None:
                ref_per_token_logps = self._get_per_token_logps(
                    self.ref_model, prompt_completion_ids, attention_mask, logits_to_keep
                )
            else:
                with self.accelerator.unwrap_model(self.model).disable_adapter():
                    ref_per_token_logps = self._get_per_token_logps(
                        self.model, prompt_completion_ids, attention_mask, logits_to_keep
                    )
        self._print_debugging_logs('decoding text...')

        advantages = advantages.to(device)

        self._print_debugging_logs('returning for loss...')
        return {
            "prompt_ids": prompt_ids,
            "prompt_mask": prompt_mask,
            "completion_ids": completion_ids,
            "completion_mask": completion_mask,
            "ref_per_token_logps": ref_per_token_logps,
            "advantages": advantages,
        }

    def _generate_and_score(self, inputs: dict[str, Union[torch.Tensor, Any]]) -> dict[str, Union[torch.Tensor, Any]]:

        device = self.accelerator.device
        prompts = [x["prompt"] for x in inputs]
        prompts_text = [maybe_apply_chat_template(example, self.processing_class)[
            "prompt"] for example in inputs]

        if self.args.use_vllm or self.args.use_ray or self.args.use_vllm_server:

            gc.collect()
            torch.cuda.empty_cache()

            self.accelerator.wait_for_everyone()
            torch.cuda.synchronize()
            if (self.accelerator.is_main_process and
                    self.args.activate_debugging_logs):
                print('resources before generation')
                res = get_resource_info()
                print(res)
                print_ram_utilization()
                if ("wandb" in self.args.report_to) and (
                        self.state.global_step % self.args.logging_steps == 0):
                    res_to_log = {}
                    for k, k_dict in res.items():
                        for k_dict_k, k_dict_v in k_dict.items():
                            res_to_log[f'{k}/{k_dict_k}/before_gen'] = k_dict_v
                    res_to_log["step"] = self.state.global_step
                    if wandb.run is not None:
                        wandb.log(res_to_log)
            if self.state.global_step != self._last_loaded_step:
                if self.accelerator.is_main_process:
                    start = time.time()
                self._pre_model_moving()
                self._move_model_to_vllm()
                self._post_model_moving()
                if self.accelerator.is_main_process:
                    end = time.time()
                    print(f"Total time to update VLLMs: {end - start:.6f} s")
                self._last_loaded_step = self.state.global_step
            gc.collect()
            torch.cuda.empty_cache()
            self._print_debugging_logs('gathering text for vllm...')

            all_prompts_text = gather_object(prompts_text)
            if self.args.shuffle_generation_inputs:

                indices = np.random.permutation(len(all_prompts_text))
                shuffled_prompts = [all_prompts_text[i] for i in indices]
            else:
                shuffled_prompts = all_prompts_text

            if self.accelerator.is_main_process:
                self._print_debugging_logs('vllm generation...')
                if self.use_ray:
                    self._print_debugging_logs('ray generation...')
                    outputs = self.ray_actor.generate(shuffled_prompts)
                    completion_ids = [
                        out.token_ids
                        for completions in outputs
                        for out in completions.outputs
                    ]
                    self._print_debugging_logs('ray generation done.')
                    self._print_debugging_logs(
                        f'logging {len(completion_ids)} completion ids, from '
                        f'{len(shuffled_prompts)} prompts'
                    )
                elif self.use_vllm_server:
                    def generate_with_client(client, prompts):
                        return client.generate(
                            prompts=prompts,
                            n=1,
                            repetition_penalty=self.repetition_penalty,
                            temperature=self.temperature,
                            top_p=self.top_p,
                            top_k=-1 if self.top_k is None else self.top_k,
                            min_p=0.0 if self.min_p is None else self.min_p,
                            max_tokens=self.max_completion_length,
                        )
                    with concurrent.futures.ThreadPoolExecutor() as executor:
                        futures = []
                        ss = 0
                        number_of_prompts = len(shuffled_prompts)
                        number_of_clients = len(self.vllm_clients)
                        base, rem = divmod(
                            number_of_prompts, number_of_clients)

                        chunk_sizes = [
                            base + (1 if i < rem else 0)
                            for i in range(number_of_clients)
                        ]
                        futures = []
                        ss = 0
                        for i, client in enumerate(self.vllm_clients):
                            size = chunk_sizes[i]

                            chunk = shuffled_prompts[ss:ss + size]
                            futures.append(
                                executor.submit(
                                    generate_with_client,
                                    client,
                                    chunk,
                                )
                            )
                            ss += size

                        completion_ids = [f.result() for f in futures]
                        completion_ids = [
                            x for sub in completion_ids for x in sub]

                else:
                    outputs = self.llm.generate(
                        shuffled_prompts,
                        sampling_params=self.sampling_params,
                        use_tqdm=False
                    )
                    completion_ids = [
                        out.token_ids
                        for completions in outputs
                        for out in completions.outputs
                    ]
            else:
                completion_ids = [None] * len(shuffled_prompts)

            if self.args.shuffle_generation_inputs:

                ordered_completion_ids = [None] * len(completion_ids)
                for shuffled_index, original_index in enumerate(indices):
                    ordered_completion_ids[original_index] = completion_ids[
                        shuffled_index]
                completion_ids = ordered_completion_ids

            gc.collect()
            torch.cuda.empty_cache()
            if (self.accelerator.is_main_process and
                    self.args.activate_debugging_logs):
                print('resources after gen.')
                res = get_resource_info()
                print(res)
                print_ram_utilization()
                if ("wandb" in self.args.report_to) and (
                        self.state.global_step % self.args.logging_steps == 0):
                    res_to_log = {}
                    for k, k_dict in res.items():
                        for k_dict_k, k_dict_v in k_dict.items():
                            res_to_log[f'{k}/{k_dict_k}/after_gen'] = k_dict_v
                    res_to_log["step"] = self.state.global_step
                    if wandb.run is not None:
                        wandb.log(res_to_log)

            self._print_debugging_logs('distributing completions...')

            completion_ids = broadcast_object_list(
                completion_ids, from_process=0)

            self._print_debugging_logs('distributed completions!')

            process_slice = slice(
                self.accelerator.process_index * len(prompts),
                (self.accelerator.process_index + 1) * len(prompts),
            )
            completion_ids = completion_ids[process_slice]
        else:

            prompt_inputs = self.processing_class(
                prompts_text, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False
            )
            self._print_debugging_logs('preparing inputs...')
            prompt_inputs = super()._prepare_inputs(prompt_inputs)
            prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

            if self.max_prompt_length is not None:
                prompt_ids = prompt_ids[:, -self.max_prompt_length:]
                prompt_mask = prompt_mask[:, -self.max_prompt_length:]
            with unwrap_model_for_generation(self.model, self.accelerator) as unwrapped_model:
                prompt_completion_ids = unwrapped_model.generate(
                    prompt_ids, attention_mask=prompt_mask, generation_config=self.generation_config
                )

            prompt_length = prompt_ids.size(1)

            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = torch.unbind(
                prompt_completion_ids[:, prompt_length:].cpu(), dim=0)

        self._print_debugging_logs('decoding text...')

        completions_text = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True)
        if is_conversational(inputs[0]):
            completions = []
            for prompt, completion in zip(prompts, completions_text):
                bootstrap = prompt.pop()[
                    "content"] if prompt[-1]["role"] == "assistant" else ""
                completions.append(
                    [{"role": "assistant", "content": bootstrap + completion}])
        else:
            completions = completions_text
        self._print_debugging_logs('computing rws...')
        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device)
        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, nn.Module):
                if is_conversational(inputs[0]):
                    messages = [{"messages": p + c}
                                for p, c in zip(prompts, completions)]
                    texts = [apply_chat_template(x, reward_processing_class)[
                        "text"] for x in messages]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts, return_tensors="pt", padding=True, padding_side="right", add_special_tokens=False
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(
                        **reward_inputs).logits[:, 0]
            else:

                keys = [key for key in inputs[0]
                        if key not in ["prompt", "completion"]]
                reward_kwargs = {key: [example[key]
                                       for example in inputs] for key in keys}
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs)
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device)

        self._print_debugging_logs('gathering rws...')

        rewards_per_func = gather(rewards_per_func)

        rewards = (rewards_per_func *
                   self.reward_weights.to(device).unsqueeze(0)).sum(dim=1)

        self._print_debugging_logs('normalizing rws...')

        mean_grouped_rewards = rewards.view(-1,
                                            self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)

        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0)
        advantages = (rewards - mean_grouped_rewards) / \
            (std_grouped_rewards + 1e-4)

        process_slice = slice(
            self.accelerator.process_index * len(prompts),
            (self.accelerator.process_index + 1) * len(prompts),
        )
        advantages = advantages[process_slice]

        reward_per_func = rewards_per_func.mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, nn.Module):
                reward_func_name = reward_func.config._name_or_path.split(
                    "/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item())

        self._metrics["reward"].append(rewards.mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards.mean().item())

        self._print_debugging_logs('logging...')

        if ((self.log_completions and "wandb" in self.args.report_to) or
            self.store_completions_to_file) and (
                self.state.global_step % self.args.logging_steps == 0):

            import pandas as pd

            table = {
                "step": [str(self.state.global_step)] * len(rewards),
                "prompt": gather_object(prompts_text),
                "completion": gather_object(completions_text),
                "reward": rewards.tolist(),
            }
            df = pd.DataFrame(table)

            if self.accelerator.is_main_process:
                if self.store_completions_to_file:
                    filtered_df = df[
                        np.random.rand(len(df)) < self.save_completions_probability]
                    if not filtered_df.empty:
                        c_output_path = (
                            f"{self.args.output_dir}/stored_completions")
                        os.makedirs(c_output_path, exist_ok=True)
                        c_file_path = (
                            f"{c_output_path}/{self.state.global_step}.csv")
                        filtered_df.to_csv(
                            c_file_path,
                            mode='a',
                            header=not os.path.exists(c_file_path),
                            index=False,
                            quoting=csv.QUOTE_ALL,
                            escapechar='\\',
                        )

                if self.log_completions and "wandb" in self.args.report_to:
                    if wandb.run is not None:
                        wandb.log({"completions": wandb.Table(dataframe=df)})

        self._print_debugging_logs('returning for loss...')

        return prompts_text, completion_ids, advantages.cpu()

    def _compute_backprop_step_losses(
            self, model, prompt_ids, prompt_mask, completion_ids,
            completion_mask, ref_per_token_logps, advantages,):
        input_ids = torch.cat([prompt_ids, completion_ids], dim=1)
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        logits_to_keep = completion_ids.size(1)
        per_token_logps = self._get_per_token_logps(
            model, input_ids, attention_mask, logits_to_keep)

        per_token_kl = torch.exp(
            ref_per_token_logps - per_token_logps) - (ref_per_token_logps - per_token_logps) - 1

        per_token_loss = torch.exp(
            per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
        per_token_loss = -(per_token_loss - self.beta * per_token_kl)
        with torch.no_grad():
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) /
                       completion_mask.sum(dim=1))
        loss = ((per_token_loss * completion_mask).sum(dim=1) /
                completion_mask.sum(dim=1))
        return loss, mean_kl

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        if return_outputs:
            raise ValueError(
                "The GRPOTrainer does not support returning outputs")

        self._print_debugging_logs('entering loss...')
        prompt_ids, prompt_mask = inputs["prompt_ids"], inputs["prompt_mask"]
        completion_ids, completion_mask = inputs["completion_ids"], inputs["completion_mask"]
        ref_per_token_logps = inputs["ref_per_token_logps"]
        advantages = inputs["advantages"]

        prompt_ids_chunks = torch.split(
            prompt_ids, self.backprop_step_size, dim=0)
        prompt_mask_chunks = torch.split(
            prompt_mask, self.backprop_step_size, dim=0)
        completion_ids_chunks = torch.split(
            completion_ids, self.backprop_step_size, dim=0)
        completion_mask_chunks = torch.split(
            completion_mask, self.backprop_step_size, dim=0)
        ref_chunks = torch.split(
            ref_per_token_logps, self.backprop_step_size, dim=0)
        advantages_chunks = torch.split(
            advantages, self.backprop_step_size, dim=0)

        losses = []
        mean_kls = []
        for (prompt_ids_i, prompt_mask_i, completion_ids_i,
             completion_mask_i, ref_per_token_logps_i, advantages_i) in zip(
                prompt_ids_chunks, prompt_mask_chunks, completion_ids_chunks,
                completion_mask_chunks, ref_chunks, advantages_chunks):
            losses_i, mean_kl_i = self._compute_backprop_step_losses(
                model=model,
                prompt_ids=prompt_ids_i,
                prompt_mask=prompt_mask_i,
                completion_ids=completion_ids_i,
                completion_mask=completion_mask_i,
                ref_per_token_logps=ref_per_token_logps_i,
                advantages=advantages_i,
            )
            losses.append(losses_i)
            mean_kls.append(mean_kl_i)

        loss = torch.concat(losses, dim=0).mean()
        mean_kl = torch.concat(mean_kls, dim=0).mean()

        completion_length = self.accelerator.gather_for_metrics(
            completion_mask.sum(1)).float().mean().item()
        self._metrics["completion_length"].append(completion_length)
        self._metrics["kl"].append(
            self.accelerator.gather_for_metrics(mean_kl).mean().item())

        self._print_debugging_logs('finished loss...')
        gc.collect()
        if self.accelerator.is_main_process:
            torch.cuda.empty_cache()
        self.accelerator.wait_for_everyone()
        if not self.accelerator.is_main_process:
            torch.cuda.empty_cache()
        return loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys: Optional[list[str]] = None):
        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            with self.compute_loss_context_manager():
                loss = self.compute_loss(model, inputs)
            loss = loss.mean().detach()
        return loss, None, None

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {key: sum(val) / len(val)
                   for key, val in self._metrics.items()}

        if next(iter(logs.keys())).startswith("eval_"):
            metrics = {f"eval_{key}": val for key, val in metrics.items()}

        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:
            super().log(logs)
        self._metrics.clear()

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):

        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(self.model.config._name_or_path):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            }
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=wandb.run.get_url() if is_wandb_available(
            ) and wandb.run is not None else None,
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
