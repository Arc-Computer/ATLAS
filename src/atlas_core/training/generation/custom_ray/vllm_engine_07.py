import os
import gc
import queue
from collections import defaultdict
from typing import Any, Dict, List, Optional, TypeVar, Union

import ray
import psutil
import torch
import vllm
from torch.multiprocessing.reductions import rebuild_cuda_tensor
from unittest.mock import patch
import cloudpickle
from ray.util.placement_group import placement_group
from ray.util.scheduling_strategies import PlacementGroupSchedulingStrategy

from .utils import ray_noset_visible_devices

from vllm import LLM
from vllm.config import CompilationConfig
from vllm.engine.arg_utils import (
    EngineArgs,
    HfOverrides,
    PoolerConfig,
    TaskOption,
)
from vllm.logger import init_logger
from vllm.usage.usage_lib import UsageContext
from vllm.worker.worker import Worker
from vllm.utils import Counter


logger = init_logger(__name__)


def print_env_variables():
    for key, value in os.environ.items():
        print(f"{key}={value}")


def print_ram_utilization():
    ram = psutil.virtual_memory()
    print(f"Total RAM: {ram.total / (1024 ** 3):.2f} GB")
    print(f"Used RAM: {ram.used / (1024 ** 3):.2f} GB")
    print(f"Available RAM: {ram.available / (1024 ** 3):.2f} GB")
    print(f"RAM Utilization: {ram.percent:.2f}%")


def get_resource_info() -> Dict[int, Dict[str, Any]]:

    resources = {}
    torch.cuda.empty_cache()
    for i in range(torch.cuda.device_count()):
        try:
            props = torch.cuda.get_device_properties(i)

            mem_allocated = torch.cuda.memory_allocated(i)
            mem_total = props.total_memory
            mem_free = mem_total - mem_allocated

            resources[i] = {
                "name": props.name,

                "total_memory": mem_total / (1024**3),
                "free_memory": mem_free / (1024**3),
                "allocated_memory": mem_allocated / (1024**3),

            }
        except Exception as e:
            resources[i] = {"error": str(e)}

    return resources


@ray.remote
def get_all_env_variables():
    import os
    return os.environ


class CustomLLM(LLM):
    def __init__(
        self,
        model: str,
        tokenizer: Optional[str] = None,
        tokenizer_mode: str = "auto",
        skip_tokenizer_init: bool = False,
        trust_remote_code: bool = False,
        allowed_local_media_path: str = "",
        tensor_parallel_size: int = 1,
        dtype: str = "auto",
        quantization: Optional[str] = None,
        revision: Optional[str] = None,
        tokenizer_revision: Optional[str] = None,
        seed: int = 0,
        gpu_memory_utilization: float = 0.9,
        swap_space: float = 4,
        cpu_offload_gb: float = 0,
        enforce_eager: Optional[bool] = None,
        max_seq_len_to_capture: int = 8192,
        disable_custom_all_reduce: bool = False,
        disable_async_output_proc: bool = False,
        hf_overrides: Optional[HfOverrides] = None,
        mm_processor_kwargs: Optional[Dict[str, Any]] = None,

        task: TaskOption = "auto",
        override_pooler_config: Optional[PoolerConfig] = None,
        compilation_config: Optional[Union[int, Dict[str, Any]]] = None,
        **kwargs,
    ) -> None:

        if "disable_log_stats" not in kwargs:
            kwargs["disable_log_stats"] = True
        if "worker_cls" in kwargs:
            worker_cls = kwargs["worker_cls"]

            if isinstance(worker_cls, type):
                kwargs["worker_cls"] = cloudpickle.dumps(worker_cls)

        if compilation_config is not None:
            if isinstance(compilation_config, (int, dict)):
                compilation_config_instance = CompilationConfig.from_cli(
                    str(compilation_config))
            else:
                compilation_config_instance = compilation_config
        else:
            compilation_config_instance = None

        engine_args = EngineArgs(
            model=model,
            task=task,
            tokenizer=tokenizer,
            tokenizer_mode=tokenizer_mode,
            skip_tokenizer_init=skip_tokenizer_init,
            trust_remote_code=trust_remote_code,
            allowed_local_media_path=allowed_local_media_path,
            tensor_parallel_size=tensor_parallel_size,
            dtype=dtype,
            quantization=quantization,
            revision=revision,
            tokenizer_revision=tokenizer_revision,
            seed=seed,

            gpu_memory_utilization=gpu_memory_utilization,
            swap_space=swap_space,
            cpu_offload_gb=cpu_offload_gb,
            enforce_eager=enforce_eager,
            max_seq_len_to_capture=max_seq_len_to_capture,
            disable_custom_all_reduce=disable_custom_all_reduce,
            disable_async_output_proc=disable_async_output_proc,
            hf_overrides=hf_overrides,
            mm_processor_kwargs=mm_processor_kwargs,
            override_pooler_config=override_pooler_config,
            compilation_config=compilation_config_instance,
            **kwargs,
        )

        self.engine_class = self.get_engine_class()

        self.llm_engine: Worker = self.engine_class.from_engine_args(
            engine_args, usage_context=UsageContext.LLM_CLASS)

        self.request_counter = Counter()

    def _assert_memory_footprint_increased_during_profiling(self, *args, **kwargs):
        return None


@ray.remote
class LLMRayActor:

    def __init__(self, *args,
                 bundle_indices: list = None,
                 override_custom_port=None,
                 override_master_addr=None,
                 **kwargs):
        noset_visible_devices = kwargs.pop("noset_visible_devices")
        if kwargs.get("distributed_executor_backend") == "ray":

            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        elif noset_visible_devices:

            os.environ["CUDA_VISIBLE_DEVICES"] = str(ray.get_gpu_ids()[0])

        num_gpus = kwargs.pop("num_gpus")

        if bundle_indices is not None:
            os.environ["VLLM_RAY_PER_WORKER_GPUS"] = str(num_gpus)
            os.environ["VLLM_RAY_BUNDLE_INDICES"] = ",".join(
                map(str, bundle_indices))
            print(f"creating LLM with bundle_indices={bundle_indices}")

        if override_master_addr is not None:
            os.environ["MASTER_ADDR"] = override_master_addr
        if override_custom_port is not None:
            os.environ["MASTER_PORT"] = str(override_custom_port)

        for var in [





        ]:

            os.environ.pop(var, None)
        for var, value in os.environ.items():
            if var.startswith("ACCELERATE_"):

                os.environ.pop(var, None)

            if var.startswith("TORCHELASTIC_"):

                os.environ.pop(var, None)

        self.num_actors = kwargs.pop("num_actors")
        self.actor_counter = 0
        self.requests = {}
        self.response_queues = defaultdict(queue.Queue)

        self.used_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        print(f'Initializing actor for device {self.used_device}')
        profiling_patch = patch(
            "vllm.worker.worker.Worker._assert_memory_footprint_increased_during_profiling",
            return_value=None
        )

        with profiling_patch:
            self.llm = CustomLLM(*args, **kwargs)
        self.used_device = os.environ.get("CUDA_VISIBLE_DEVICES", None)

        print(f'VLLM actor successfully init. for device {self.used_device}')

    def print_with_device(self, to_print):
        print(f'Actor device {self.used_device}: {to_print}')

    def init_weight_update_group_s(
            self, master_address, master_port, rank_offset, world_size,):
        self.print_with_device(f' init update group with offset {rank_offset} '
                               f'and world size {world_size} -- {master_address}: {master_port}')
        return self.llm.collective_rpc(
            "init_weight_update_group_s",
            args=(master_address, master_port, rank_offset, world_size),
        )

    def get_weight_update_group(
            self, master_address, master_port, rank_offset, world_size,):
        return self.llm.collective_rpc(
            "init_weight_update_group_s",
            args=(master_address, master_port, rank_offset, world_size),
        )

    def update_weight_s(self, name, dtype, shape):
        return self.llm.collective_rpc(
            "update_weight_s", args=(name, dtype, shape))

    def check_weights_changed_s(self):
        return self.llm.collective_rpc("check_weights_changed_s", args=())

    def reset_prefix_cache(self):
        self.llm.llm_engine.reset_prefix_cache()

    def sleep(self, level=1):
        gc.collect()
        torch.cuda.empty_cache()
        self.llm.sleep(level=level)
        gc.collect()
        torch.cuda.empty_cache()

    def wake_up(self):
        gc.collect()
        torch.cuda.empty_cache()
        self.llm.wake_up()
        gc.collect()
        torch.cuda.empty_cache()

    def add_requests(self, actor_rank, prompts, *, sampling_params):

        self.requests[actor_rank] = prompts
        self.actor_counter += 1
        if self.actor_counter == self.num_actors:
            assert len(self.requests) == self.num_actors
            num_requests = []
            requests = []
            for actor_rank, request in self.requests.items():
                num_requests.append((actor_rank, len(request)))
                requests.extend(request)

            if len(requests) > 0:

                responses = self.llm.generate(
                    requests,
                    sampling_params=sampling_params,
                )
            else:
                responses = []

            offset = 0
            self.responses = {}
            for actor_rank, num in num_requests:
                self.response_queues[actor_rank].put(
                    responses[offset:offset + num])
                offset += num

            self.actor_counter = 0
            self.requests = {}

    def get_responses(self, actor_rank):

        return self.response_queues[actor_rank].get()

    def update_state_dict(self, state_dict, broadcast_to_other_devices=False,
                          is_main_process=False):

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights(state_dict.items())

    def update_self_weight(self, name, p):

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights([(name, p)])
        del p

    def update_self_weight_from_metadata(self, name, p_metadata, clone=True):

        if clone:
            p = torch.clone(rebuild_cuda_tensor(torch.Tensor, **p_metadata,))
        else:
            p = rebuild_cuda_tensor(torch.Tensor, **p_metadata,)

        llm_model = self.llm.llm_engine.model_executor.driver_worker.model_runner.model
        llm_model.load_weights([(name, p)])
        del p

    def _get_obj_name_recurse(self, name, obj):
        name = name.split(".", maxsplit=1)
        recurse = len(name) > 1
        next_name = name[1] if recurse else ""
        name = name[0]
        obj = self if obj is None else obj
        return obj, name, next_name, recurse

    def get_remote_attr(self, __name: str, __obj: object | None = None):
        obj, name, next_name, recurse = self._get_obj_name_recurse(
            __name, __obj)
        next_obj = getattr(obj, name)
        if recurse:
            next_obj = self.get_remote_attr(next_name, next_obj)
        return next_obj

    def set_remote_attr(
            self, __name: str, __value: Any, __obj: object | None = None):
        obj, name, next_name, recurse = self._get_obj_name_recurse(
            __name, __obj)
        if recurse:
            self.set_remote_attr(next_name, __value, obj)
        if hasattr(obj, name):
            setattr(obj, name, __value)


def create_vllm_engines(
    num_engines: int,
    tensor_parallel_size: int,
    pretrain: str,
    seed: int,
    enable_prefix_caching: bool,
    enforce_eager: bool,
    max_model_len: int,
    num_total_actors: int,
    dtype: str = "bfloat16",
    shared_pg=None,
    gpu_memory_utilization=None,
    vllm_enable_sleep=False,
    sleep_level=0,
    vllm_devices: list = None,
):
    import vllm

    assert vllm.__version__ >= "0.7.2", "OpenRLHF only supports vllm >= 0.7.2"
    if vllm_devices is not None:
        print(
            f'Setting visible devices to {vllm_devices} for ray actors init.')
        assert 0 not in vllm_devices
        vllm_devices_str = ",".join(str(d) for d in vllm_devices)
        original_devices = os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        os.environ["CUDA_VISIBLE_DEVICES"] = vllm_devices_str

    else:
        print(f'Vllm device {vllm_devices} is None, defaulting visibility to:')
        print(f'{os.environ.get("CUDA_VISIBLE_DEVICES", None)}.')

    vllm_engines = []
    distributed_executor_backend = "uni" if tensor_parallel_size == 1 else "ray"

    use_hybrid_engine = shared_pg is not None
    num_gpus = int(tensor_parallel_size == 1)
    if use_hybrid_engine and tensor_parallel_size == 1:

        num_gpus = 0.2

    if not use_hybrid_engine:

        bundles = [{"GPU": 1, "CPU": 1} for _ in range(
            num_engines * tensor_parallel_size)]
        shared_pg = placement_group(bundles, strategy="PACK")
        ray.get(shared_pg.ready())

    print(f'Initializing {num_engines} engines each with {num_gpus} GPUS')

    for i in range(num_engines):
        bundle_indices = None
        if tensor_parallel_size > 1:
            bundle_indices = list(
                range(i*tensor_parallel_size, (i + 1)*tensor_parallel_size))

        scheduling_strategy = PlacementGroupSchedulingStrategy(
            placement_group=shared_pg,
            placement_group_capture_child_tasks=True,
            placement_group_bundle_index=i*tensor_parallel_size,
        )

        if num_engines >= num_total_actors:
            num_actors = 1
        else:
            num_actors = num_total_actors // num_engines + int(
                i < num_total_actors % num_engines)

        vllm_engines.append(
            LLMRayActor.options(
                num_cpus=num_gpus,
                num_gpus=num_gpus,
                scheduling_strategy=scheduling_strategy,
            ).remote(
                model=pretrain,
                enforce_eager=enforce_eager,
                worker_cls="atlas_core.training.generation.custom_ray.vllm_worker_wrap.WorkerWrap",
                tensor_parallel_size=tensor_parallel_size,
                seed=seed + i,
                distributed_executor_backend=distributed_executor_backend,
                max_model_len=max_model_len,
                enable_prefix_caching=enable_prefix_caching,
                dtype=dtype,
                trust_remote_code=True,
                num_actors=num_actors,
                gpu_memory_utilization=gpu_memory_utilization,
                bundle_indices=bundle_indices,
                num_gpus=0.2 if use_hybrid_engine else 1,
                enable_sleep_mode=vllm_enable_sleep,

                noset_visible_devices=ray_noset_visible_devices(),
            )
        )

    if vllm_enable_sleep:

        batch_vllm_engine_call(
            vllm_engines, "sleep", rank_0_only=False, level=sleep_level)

    if vllm_devices is not None:
        print(f'Setting visible devices back to {original_devices}')
        if original_devices is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = original_devices
        else:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    return vllm_engines


def batch_vllm_engine_refs(
        engines: List[Any], method_name: str, *args,
        rank_0_only: bool = True, **kwargs):
    if rank_0_only and torch.distributed.get_rank() != 0:
        return None

    refs = []
    for engine in engines:
        method = getattr(engine, method_name)
        refs.append(method.remote(*args, **kwargs))

    return refs


def batch_vllm_engine_call(
        engines: List[Any], method_name: str, *args,
        rank_0_only: bool = True, **kwargs):
    if rank_0_only and torch.distributed.get_rank() != 0:
        return None
    refs = batch_vllm_engine_refs(
        engines, method_name, *args, rank_0_only=rank_0_only, **kwargs)

    return ray.get(refs)
