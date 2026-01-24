import torch
import ray
import os
import time

from typing import Optional, Dict, Any, Callable, List

from transformers import Trainer, TrainingArguments, AutoConfig, AutoTokenizer
from vllm import LLM, SamplingParams
from ray.util.placement_group import placement_group
from .vllm_engine import (
    create_vllm_engines, batch_vllm_engine_call, get_resource_info)
from .vllm_worker_wrap import stateless_init_process_group
from vllm.utils import get_ip, get_open_port


def get_rank_safe():

    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        return torch.distributed.get_rank()
    return 0


def get_world_size_safe():

    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        return torch.distributed.get_world_size()
    return 1


def barrier_safe():

    if (torch.distributed.is_available() and
            torch.distributed.is_initialized()):
        torch.distributed.barrier()


def _extract_cuda_metadata(tensor: torch.Tensor):
    storage = tensor._typed_storage()
    (
        storage_device,
        storage_handle,
        storage_size_bytes,
        storage_offset_bytes,
        ref_counter_handle,
        ref_counter_offset,
        event_handle,
        event_sync_required,
    ) = storage._share_cuda_()

    return {
        "dtype": tensor.dtype,
        "tensor_size": tensor.size(),
        "tensor_stride": tensor.stride(),
        "tensor_offset": tensor.storage_offset(),
        "storage_cls": type(storage),
        "storage_device": storage_device,
        "storage_handle": storage_handle,
        "storage_size_bytes": storage_size_bytes,
        "storage_offset_bytes": storage_offset_bytes,
        "requires_grad": tensor.requires_grad,
        "ref_counter_handle": ref_counter_handle,
        "ref_counter_offset": ref_counter_offset,
        "event_handle": event_handle,
        "event_sync_required": event_sync_required,
    }


class RayGeneratorActor:
    def __init__(
        self,
        model: str,
        revision: str = None,
        tokenizer: Optional[Any | str] = None,
        seed: int = 42,

        ray_num_nodes: int = 1,
        ray_tensor_parallelism: int = 1,
        ray_data_parallelism: int = 1,
        vllm_gpu_memory_utilization: float = 0.9,
        vllm_dtype: str = "auto",
        enable_prefix_caching: bool = False,
        enforce_eager: bool = True,
        sleep_level: int = 0,

        max_prompt_length: int = 32768,
        max_tokens: int = 32768,

        max_completion_length: Optional[int] = 32768,
        temperature: float = 1.0,
        top_k: int = -1,
        top_p: float = 1.0,
        repetition_penalty: float = 1.0,
        presence_penalty: float = 0.0,
        frequency_penalty: float = 0.0,
        collective_rpc_mode: Optional[str] = 'nccl',

        verbose_generator: bool = True,
        reserved_gpus: int = 0,
        activate_debugging_logs: bool = False,
        sampling_params=None,
        show_progress: bool = False,
    ):
        self.sampling_params = sampling_params
        self.show_progress = show_progress

        self.activate_debugging_logs = activate_debugging_logs
        if reserved_gpus is None:
            reserved_gpus = 0

        if reserved_gpus > 0:
            self._print_debugging_logs('Checking sleep level...')
            self.shared_gpus = False
            assert sleep_level == 0
        else:
            self.shared_gpus = True
        self.model = model
        if tokenizer is None:
            tokenizer = model
        if isinstance(tokenizer, str):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer

        self.seed = seed
        self.ray_num_nodes = ray_num_nodes
        self.ray_tensor_parallelism = ray_tensor_parallelism
        self.ray_data_parallelism = ray_data_parallelism
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.vllm_dtype = vllm_dtype
        self.enable_prefix_caching = enable_prefix_caching
        self.enforce_eager = enforce_eager
        self.sleep_level = int(sleep_level)
        self.enable_sleep = sleep_level > 0
        if self.enable_sleep:
            assert self.sleep_level in [1, 2]
            if self.sleep_level == 2:

                raise NotImplementedError

        self.max_prompt_length = max_prompt_length
        self.max_tokens = max_tokens
        self.max_completion_length = max_completion_length
        if self.max_completion_length is None:
            self.max_completion_length = self.max_tokens

        self.temperature = temperature
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.presence_penalty = presence_penalty
        self.frequency_penalty = frequency_penalty

        self.num_gpus_per_node = (
            self.ray_tensor_parallelism*self.ray_data_parallelism)

        self.total_devices = self.ray_num_nodes*self.num_gpus_per_node

        if not self.shared_gpus:
            vllm_devices = [
                i + reserved_gpus for i in range(self.total_devices)]
            if self.ray_num_nodes > 1:

                raise NotImplementedError
        else:
            vllm_devices = None

        pg = None
        self.model_awoken = False

        runtime_env = {
            'env_vars': {
                "RAY_memory_monitor_refresh_ms": "0",
                "RAY_memory_usage_threshold": "3"
            }
        }

        if vllm_devices is not None:
            print(
                f'Setting visible devices to {vllm_devices} for ray actors init.')
            assert 0 not in vllm_devices
            vllm_devices_str = ",".join(str(d) for d in vllm_devices)
            original_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices_str
        else:
            print(
                f'Vllm device {vllm_devices} is None, defaulting visibility to:')
            print(f'{os.environ.get("CUDA_VISIBLE_DEVICES", None)}.')

        ray.init(runtime_env=runtime_env)
        self._print_debugging_logs(ray.available_resources())
        self.vllm_engines = create_vllm_engines(
            num_engines=ray_data_parallelism,
            tensor_parallel_size=ray_tensor_parallelism,
            pretrain=model,
            revision=revision,
            seed=seed,
            enable_prefix_caching=enable_prefix_caching,
            enforce_eager=enforce_eager,
            max_model_len=max_tokens,



            num_total_actors=1,
            dtype=self.vllm_dtype,
            shared_pg=pg,
            gpu_memory_utilization=vllm_gpu_memory_utilization,
            vllm_enable_sleep=self.enable_sleep,
            sleep_level=self.sleep_level,
            vllm_devices=vllm_devices,
            show_progress=show_progress,
        )
        self.asleep = False
        self.sleep_if_needed()
        print(f'Initialized: {len(self.vllm_engines)} engines')

        if vllm_devices is not None:
            print(f'Setting visible devices back to {original_devices}')
            if original_devices is not None:
                os.environ["CUDA_VISIBLE_DEVICES"] = original_devices
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        self.collective_rpc_mode = collective_rpc_mode
        self.use_collective_rpc_mode = (collective_rpc_mode is not None) and (
            collective_rpc_mode.lower() != 'none')
        if self.use_collective_rpc_mode:
            assert collective_rpc_mode == 'nccl'
        if self.use_collective_rpc_mode:
            self.weight_update_master_addr = get_ip()
            self.weight_update_master_port = get_open_port()
            if reserved_gpus > 0:
                rank_offsets_shift = 1
            else:
                rank_offsets_shift = 0
            rank_offsets = [i*self.ray_tensor_parallelism + rank_offsets_shift
                            for i in range(self.ray_data_parallelism)]

            print('Rank offsets for initializing for weight update process '
                  f'group {rank_offsets}, {type(rank_offsets)}.')
            self.total_update_devices = self.total_devices
            if not self.shared_gpus:
                self.total_update_devices += 1

            engines_init_args = [(
                self.weight_update_master_addr,
                self.weight_update_master_port,
                rank_offset,
                self.total_update_devices)
                for rank_offset in rank_offsets]

            initialization_handles = [
                engine.init_weight_update_group_s.remote(*init_args) for
                engine, init_args in zip(self.vllm_engines, engines_init_args)
            ]
            if not self.shared_gpus:
                self.main_engine_idx = None
                self.model_update_group = stateless_init_process_group(
                    master_address=self.weight_update_master_addr,
                    master_port=self.weight_update_master_port,
                    rank=0,
                    world_size=self.total_update_devices,
                    device=torch.device("cuda:0"),
                )
                print('Stateless process group init')
            else:
                self.main_engine_idx = 0
                self.model_update_group = None

            print('Getting process update group init handles....')
            update_groups_infos = ray.get(initialization_handles)
            if self.shared_gpus:
                self.update_group_info = update_groups_infos[
                    self.main_engine_idx][0]
            else:
                self.update_group_info = None
            print('Successfully initialized process update group!')

    def _print_debugging_logs(self, to_print: str):
        if self.activate_debugging_logs:
            print(f'Ray generator: {to_print}')

    def update_state_dict(
            self, state_dict, clone_weight=True, main_engine_idx=0,):

        if self.main_engine_idx is not None:
            main_engine = self.vllm_engines[self.main_engine_idx]
            other_engines = [self.vllm_engines[i] for i in range(
                len(self.vllm_engines)) if i != self.main_engine_idx]
            device = f'cuda:{self.main_engine_idx}'
        else:
            main_engine = None
            device = 'cuda:0'
        params_names = list(state_dict.keys())
        for k in params_names:
            p = state_dict[k]
            dtype = p.dtype
            shape = p.shape
            p_class = type(p)
            if self.shared_gpus:
                p_metadata = _extract_cuda_metadata(tensor=p)

                update_ref = main_engine.update_self_weight_from_metadata.remote(
                    name=k, p_metadata=p_metadata, clone=True)

                handles = [
                    engine.update_weight_s.remote(
                        name=k, dtype=dtype, shape=shape)
                    for engine in other_engines
                ]
                handles = [update_ref] + handles

                self._print_debugging_logs(f'syncing: {len(handles)} models')
                ray.get(handles)
            else:
                self.model_update_group.broadcast(
                    p,
                    src=0, stream=torch.cuda.current_stream())

                handles = [
                    engine.update_weight_s.remote(
                        name=k, dtype=dtype, shape=shape)
                    for engine in self.vllm_engines
                ]
                ray.get(handles)

    def generate(self, all_prompts: List[str], return_only_completions=False,
                 update_iteration=False, **kwargs):

        self.wake_if_needed()

        if self.sampling_params is None:
            sampling_params = SamplingParams(
                temperature=self.temperature,
                top_p=self.top_p,
                top_k=self.top_k,
                repetition_penalty=self.repetition_penalty,
                presence_penalty=self.presence_penalty,
                frequency_penalty=self.frequency_penalty,
                max_tokens=self.max_completion_length,
            )
        else:
            sampling_params = self.sampling_params

        rank = 0
        world_size = 1

        llms = self.vllm_engines

        refs = []
        batch_size = (len(all_prompts) + len(llms) - 1)//len(llms)
        for i, llm in enumerate(llms):
            prompts = all_prompts[
                i*batch_size:(i + 1)*batch_size]
            refs.append(
                llm.add_requests.remote(
                    rank,
                    prompts=prompts,
                    sampling_params=sampling_params,
                ),
            )
        ray.get(refs)

        all_output_refs = []
        for i, llm in enumerate(llms):

            all_output_refs.append(llm.get_responses.remote(rank))
        all_outputs = sum(ray.get(all_output_refs), [])

        self.sleep_if_needed()

        torch.cuda.synchronize()

        if return_only_completions:
            all_output_completions = []
            for output in all_outputs:
                all_output_completions.extend(output.outputs)

            return all_output_completions

        return all_outputs

    def wake_if_needed(self):
        if self.enable_sleep and self.asleep:
            self._print_debugging_logs('waiting to wake up...')
            batch_vllm_engine_call(self.vllm_engines, "wake_up")
            torch.cuda.synchronize()
            self._print_debugging_logs('awoken!')
            self.asleep = False

    def sleep_if_needed(self):
        if self.enable_sleep and (not self.asleep):
            batch_vllm_engine_call(
                self.vllm_engines, "sleep", level=self.sleep_level)
            torch.cuda.synchronize()
            self.asleep = True

    def reset_prefix_cache(self,):
        batch_vllm_engine_call(
            self.vllm_engines, "reset_prefix_cache")
