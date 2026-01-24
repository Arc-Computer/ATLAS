import torch
from vllm.worker.worker import Worker

from .utils import get_physical_gpu_id, init_process_group
from torch.multiprocessing.reductions import rebuild_cuda_tensor


import datetime
import pickle
from vllm.distributed.utils import StatelessProcessGroup
from torch.distributed import ProcessGroup, TCPStore


class CustomStatelessProcessGroup(StatelessProcessGroup):

    barrier_counter: int = 0

    def barrier(self):

        barrier_id = self.barrier_counter
        self.barrier_counter += 1

        barrier_prefix = f"barrier/{barrier_id}"
        my_key = f"{barrier_prefix}/{self.rank}"

        self.store.set(my_key, pickle.dumps("ready"))

        for r in range(self.world_size):
            key = f"{barrier_prefix}/{r}"

            pickle.loads(self.store.get(key))

    @staticmethod
    def create(
            host: str,
            port: int,
            rank: int,
            world_size: int,
            data_expiration_seconds: int = 60,
            store_timeout: int = 30,
    ) -> "StatelessProcessGroup":

        store = TCPStore(
            host_name=host,
            port=port,
            world_size=world_size,
            is_master=(rank == 0),
            timeout=datetime.timedelta(seconds=store_timeout),
        )

        return CustomStatelessProcessGroup(
            rank=rank,
            world_size=world_size,
            store=store,
            data_expiration_seconds=data_expiration_seconds
        )


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):

    print('Trying to initialize process group with params: '
          f'MA={master_address}, MP={master_port}, R={rank}, WS={world_size}, DEV={device}')
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator

    pg = CustomStatelessProcessGroup.create(host=master_address,
                                            port=master_port,
                                            rank=rank,
                                            world_size=world_size)

    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerExtension:
    def init_weight_update_group_s(self, master_address, master_port,
                                   rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        model_update_group_info = dict(
            unique_id=self.model_update_group.unique_id,
            world_size=self.model_update_group.world_size,
            rank=rank,
        )
        return model_update_group_info

    def update_weight_s(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream())
        self.model_update_group.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])
        self.model_update_group.group.barrier()
        del weight

    def update_self_weight_from_metadata(self, name, p_metadata):

        p = rebuild_cuda_tensor(torch.Tensor, **p_metadata,)
        self.model_update_group.broadcast(
            p, src=0, stream=torch.cuda.current_stream())
        self.model_update_group.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, p)])
        self.model_update_group.group.barrier()

    def check_weights_changed_s(self):

        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated


class WorkerWrap(Worker):
    def _assert_memory_footprint_increased_during_profiling(self,):
        return None

    def init_weight_update_group_s(self, master_address, master_port,
                                   rank_offset, world_size):
        from vllm.distributed.parallel_state import get_world_group
        rank = get_world_group().rank + rank_offset
        self.model_update_group = stateless_init_process_group(
            master_address,
            master_port,
            rank,
            world_size,
            self.device,
        )
        model_update_group_info = dict(
            unique_id=self.model_update_group.unique_id,
            world_size=self.model_update_group.world_size,
            rank=rank,
        )
        return model_update_group_info

    def update_weight_s(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(
            weight, src=0, stream=torch.cuda.current_stream())
        self.model_update_group.group.barrier()
        self.model_runner.model.load_weights(weights=[(name, weight)])
        self.model_update_group.group.barrier()
        del weight

    def update_self_weight_from_metadata(self, name, p_metadata, clone=True):

        if clone:
            p = torch.clone(rebuild_cuda_tensor(torch.Tensor, **p_metadata,))
        else:
            p = rebuild_cuda_tensor(torch.Tensor, **p_metadata,)

        mean_param = torch.mean(p).item()
        print(f'Dtype {p.dtype}, Online m param: {mean_param}')
        self.model_update_group.broadcast(
            p, src=0, stream=torch.cuda.current_stream())
        self.model_update_group.group.barrier()
        self.model_runner.model.load_weights([(name, p)])
        self.model_update_group.group.barrier()
        del p

    def check_weights_changed_s(self):

        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated
