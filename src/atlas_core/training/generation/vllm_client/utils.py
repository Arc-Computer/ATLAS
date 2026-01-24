import atexit
import logging
import time
from typing import Optional

import torch
from torch import nn

from trl.import_utils import _is_package_available


def is_requests_available() -> bool:
    return _is_package_available("requests")


def is_vllm_available() -> bool:
    return _is_package_available("vllm")


if is_requests_available():
    import requests
    from requests import ConnectionError


if is_vllm_available():
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup


logger = logging.getLogger(__name__)


class VLLMClient:

    def __init__(
        self,
        host: str = "0.0.0.0",
        server_port: int = 8000,
        group_port: int = 51216,
        connection_timeout: float = 0.0,
    ):
        if not is_requests_available():
            raise ImportError(
                "requests is not installed. Please install it with `pip install requests`."
            )
        if not is_vllm_available():
            raise ImportError(
                "vLLM is not installed. Please install it with `pip install vllm`."
            )

        self.session = requests.Session()
        self.host = host
        self.server_port = server_port
        self.group_port = group_port
        self.check_server(
            connection_timeout
        )
        print('Initializing communicator...')
        self.init_communicator()
        print('Communicator initialized!')
        atexit.register(
            self.close_communicator
        )

    def check_server(
        self, total_timeout: float = 0.0, retry_interval: float = 2.0
    ):

        url = f"http://{self.host}:{self.server_port}/health/"
        start_time = time.time()

        while True:
            try:
                response = requests.get(url)
            except requests.exceptions.RequestException as exc:

                elapsed_time = time.time() - start_time
                if elapsed_time >= total_timeout:
                    raise ConnectionError(
                        f"The vLLM server can't be reached at {self.host}:{self.server_port} after {total_timeout} "
                        "seconds. Make sure the server is running by running `trl vllm-serve`."
                    ) from exc
            else:
                if response.status_code == 200:
                    logger.info("Server is up!")
                    return None

            logger.info(
                f"Server is not up yet. Retrying in {retry_interval} seconds..."
            )
            time.sleep(retry_interval)

    def generate(
        self,
        prompts: list[str],
        n: int = 1,
        repetition_penalty: float = 1.0,
        temperature: float = 1.0,
        top_p: float = 1.0,
        top_k: int = -1,
        min_p: float = 0.0,
        max_tokens: int = 16,
        guided_decoding_regex: Optional[str] = None,
    ) -> list[list[str]]:

        url = f"http://{self.host}:{self.server_port}/generate/"
        response = self.session.post(
            url,
            json={
                "prompts": prompts,
                "n": n,
                "repetition_penalty": repetition_penalty,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "min_p": min_p,
                "max_tokens": max_tokens,
                "guided_decoding_regex": guided_decoding_regex,
            },
        )
        if response.status_code == 200:
            return response.json()["completion_ids"]
        else:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

    def init_communicator(self):

        url = f"http://{self.host}:{self.server_port}/get_tensor_parallel_size/"
        response = requests.get(url)
        if response.status_code == 200:
            tensor_parallel_size = response.json()["tensor_parallel_size"]
        else:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        world_size = tensor_parallel_size + 1
        self.rank = (
            tensor_parallel_size
        )

        url = f"http://{self.host}:{self.server_port}/init_communicator/"

        response = self.session.post(
            url,
            json={
                "host": "0.0.0.0",
                "port": self.group_port,
                "world_size": world_size,
            },
        )
        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        pg = StatelessProcessGroup.create(
            host=self.host,
            port=self.group_port,
            rank=self.rank,
            world_size=world_size,
        )
        self.pynccl_comm = PyNcclCommunicator(pg, device="cuda:0")

    def update_named_param(self, name: str, weights: torch.Tensor):

        dtype, shape = str(weights.dtype), tuple(weights.shape)
        url = f"http://{self.host}:{self.server_port}/update_named_param/"
        response = self.session.post(
            url, json={"name": name, "dtype": dtype, "shape": shape}
        )
        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

        self.pynccl_comm.broadcast(
            weights, src=self.rank, stream=torch.cuda.current_stream()
        )
        self.pynccl_comm.group.barrier()

    def update_model_params(self, model: nn.Module):

        for name, param in model.named_parameters():

            self.update_named_param(name, param.data)

    def reset_prefix_cache(self):

        url = f"http://{self.host}:{self.server_port}/reset_prefix_cache/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )

    def close_communicator(self):

        url = f"http://{self.host}:{self.server_port}/close_communicator/"
        response = self.session.post(url)
        if response.status_code != 200:
            raise Exception(
                f"Request failed: {response.status_code}, {response.text}"
            )


if __name__ == "__main__":
    from vllm import SamplingParams

    client = VLLMClient()

    responses = client.generate(
        ["Hello, AI!", "Tell me a joke"],
        n=4,
        max_tokens=32,
        sampling_params=SamplingParams(),
    )
    print("Responses:", responses)

    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B").to("cuda")
    client.update_model_params(model)
