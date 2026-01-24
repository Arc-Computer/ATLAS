import argparse
import logging
import os
from dataclasses import dataclass, field
from typing import Optional, Sequence

import torch

from trl import TrlParser
from trl.import_utils import _is_package_available


def is_fastapi_available() -> bool:
    return _is_package_available("fastapi")


def is_pydantic_available() -> bool:
    return _is_package_available("pydantic")


def is_uvicorn_available() -> bool:
    return _is_package_available("uvicorn")


def is_vllm_available() -> bool:
    return _is_package_available("vllm")


if is_fastapi_available():
    from fastapi import BackgroundTasks, FastAPI


if is_pydantic_available():
    from pydantic import BaseModel


if is_uvicorn_available():
    import uvicorn


if is_vllm_available():
    import vllm
    from vllm import LLM, SamplingParams
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.parallel_state import get_world_group
    from vllm.distributed.utils import StatelessProcessGroup
    from vllm.sampling_params import GuidedDecodingParams
    from vllm.worker.worker import Worker
else:
    Worker = object

logger = logging.getLogger(__name__)


os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"


class WeightSyncWorkerExtension:

    pynccl_comm = None
    client_rank = None

    def init_communicator(self, host: str, port: int, world_size: int) -> None:

        if self.pynccl_comm is not None:
            raise RuntimeError(
                "Weight update group already initialized. "
                "Call close_communicator first.")

        rank = get_world_group().rank

        pg = StatelessProcessGroup.create(
            host=host, port=port, rank=rank, world_size=world_size)

        self.pynccl_comm = PyNcclCommunicator(pg, device=self.device)

        self.client_rank = world_size - 1

    def update_named_param(self, name: str, dtype: torch.dtype, shape: Sequence[int]) -> None:

        if self.pynccl_comm is None:
            raise RuntimeError(
                "Communicator not initialized. Call `init_communicator` first.")

        weight = torch.empty(shape, dtype=dtype, device=self.device)

        self.pynccl_comm.broadcast(
            weight, src=self.client_rank, stream=torch.cuda.current_stream())
        self.pynccl_comm.group.barrier()

        self.model_runner.model.load_weights(weights=[(name, weight)])

    def close_communicator(self) -> None:

        if self.pynccl_comm is not None:
            del self.pynccl_comm
            self.pynccl_comm = None
            self.client_rank = None


class WeightSyncWorker(Worker, WeightSyncWorkerExtension):

    def __init__(self, *args, **kwargs):
        if not is_vllm_available():
            raise ImportError(
                "vLLM is required to use the WeightSyncWorker. Please install it using `pip install vllm`."
            )

        super().__init__(*args, **kwargs)

        self.pynccl_comm = None
        self.client_rank = None


@dataclass
class ScriptArguments:

    model: str = field(
        metadata={"help": "Model name or path to load the model from."}
    )
    revision: Optional[str] = field(
        default=None,
        metadata={
            "help": "Revision to use for the model. If not specified, the default branch will be used."
        },
    )
    tensor_parallel_size: int = field(
        default=1,
        metadata={"help": "Number of tensor parallel workers to use."},
    )
    host: str = field(
        default="0.0.0.0",
        metadata={"help": "Host address to run the server on."},
    )
    port: int = field(
        default=8000,
        metadata={"help": "Port to run the server on."},
    )
    gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    dtype: str = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This can be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )
    enable_prefix_caching: Optional[bool] = field(
        default=None,
        metadata={
            "help": "Whether to enable prefix caching in vLLM. If set to `True`, ensure that the model and the "
            "hardware support this feature."
        },
    )
    seed: Optional[int] = field(
        default=None,
        metadata={
            "help": "Vllm seed."
        },
    )


def main(script_args: ScriptArguments):
    if not is_fastapi_available():
        raise ImportError(
            "FastAPI is required to run the vLLM serve script. Please install it using `pip install fastapi`."
        )

    if not is_pydantic_available():
        raise ImportError(
            "Pydantic is required to run the vLLM serve script. Please install it using `pip install pydantic`."
        )

    if not is_uvicorn_available():
        raise ImportError(
            "Uvicorn is required to run the vLLM serve script. Please install it using `pip install uvicorn`."
        )

    if not is_vllm_available():
        raise ImportError(
            "vLLM is required to run the vLLM serve script. Please install it using `pip install vllm`."
        )

    if vllm.__version__ >= "0.8.3":
        worker_kwargs = dict(
            worker_extension_cls=(
                "__main__.WeightSyncWorkerExtension")
        )
    else:
        worker_kwargs = dict(
            worker_cls=WeightSyncWorker,
        )

    print(f"Initializing server with seed {script_args.seed}")
    llm = LLM(
        model=script_args.model,
        revision=script_args.revision,
        tensor_parallel_size=script_args.tensor_parallel_size,
        gpu_memory_utilization=script_args.gpu_memory_utilization,
        dtype=script_args.dtype,



        enable_prefix_caching=script_args.enable_prefix_caching,
        max_model_len=script_args.max_model_len,
        seed=script_args.seed,
        **worker_kwargs,
    )

    app = FastAPI()

    @app.get("/health/")
    async def health():

        return {"status": "ok"}

    @app.get("/get_tensor_parallel_size/")
    async def get_tensor_parallel_size():

        if vllm.__version__ >= "0.8.3":
            return {
                "tensor_parallel_size": (
                    llm.llm_engine.vllm_config.parallel_config.tensor_parallel_size
                )
            }
        else:
            return {
                "tensor_parallel_size": (
                    llm.llm_engine.parallel_config.tensor_parallel_size
                )
            }

    class GenerateRequest(BaseModel):
        prompts: list[str]
        n: int = 1
        repetition_penalty: float = 1.0
        temperature: float = 1.0
        top_p: float = 1.0
        top_k: int = -1
        min_p: float = 0.0
        max_tokens: int = 16
        guided_decoding_regex: Optional[str] = None

    class GenerateResponse(BaseModel):
        completion_ids: list[list[int]]

    @app.post("/generate/", response_model=GenerateResponse)
    async def generate(request: GenerateRequest):

        if request.guided_decoding_regex is not None:
            guided_decoding = GuidedDecodingParams(
                backend="outlines", regex=request.guided_decoding_regex
            )
        else:
            guided_decoding = None

        sampling_params = SamplingParams(
            n=request.n,
            repetition_penalty=request.repetition_penalty,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            min_p=request.min_p,
            max_tokens=request.max_tokens,
            guided_decoding=guided_decoding,
        )
        all_outputs = llm.generate(
            request.prompts, sampling_params=sampling_params
        )
        completion_ids = [
            list(output.token_ids)
            for outputs in all_outputs
            for output in outputs.outputs
        ]
        return {"completion_ids": completion_ids}

    class InitCommunicatorRequest(BaseModel):
        host: str
        port: int
        world_size: int

    @app.post("/init_communicator/")
    async def init_communicator(
        request: InitCommunicatorRequest, background_tasks: BackgroundTasks
    ):

        background_tasks.add_task(
            llm.collective_rpc,
            "init_communicator",
            args=(
                request.host,
                request.port,
                script_args.tensor_parallel_size + 1,
            ),
        )
        return {"message": "Request received, initializing communicator"}

    class UpdateWeightsRequest(BaseModel):
        name: str
        dtype: str
        shape: list[int]

    @app.post("/update_named_param/")
    async def update_named_param(
        request: UpdateWeightsRequest, background_tasks: BackgroundTasks
    ):

        dtype = torch.__getattribute__(request.dtype.split(".")[-1])
        background_tasks.add_task(
            llm.collective_rpc,
            "update_named_param",
            args=(request.name, dtype, request.shape),
        )

        return {"message": "Request received, updating named parameter"}

    @app.post("/reset_prefix_cache/")
    async def reset_prefix_cache():

        success = llm.llm_engine.reset_prefix_cache()
        return {
            "message": "Request received, resetting prefix cache status: "
            + str(success)
        }

    @app.post("/close_communicator/")
    async def close_communicator():

        llm.collective_rpc("close_communicator")
        return {"message": "Request received, closing communicator"}

    uvicorn.run(app, host=script_args.host, port=script_args.port)


def make_parser(subparsers: argparse._SubParsersAction = None):
    if subparsers is not None:
        parser = subparsers.add_parser(
            "vllm-serve",
            help="Run the vLLM serve script",
            dataclass_types=ScriptArguments,
        )
    else:
        parser = TrlParser(ScriptArguments)
    return parser


if __name__ == "__main__":
    parser = make_parser()
    (script_args,) = parser.parse_args_and_config()
    print('Initializing...')
    main(script_args)
