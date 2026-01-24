import torch
from vllm.worker.worker import Worker

from .utils import get_physical_gpu_id, init_process_group


def stateless_init_process_group(master_address, master_port, rank, world_size,
                                 device):

    print('Trying to initialize process group with params: '
          f'MA={master_address}, MP={master_port}, R={rank}, WS={world_size}, DEV={device}')
    from vllm.distributed.device_communicators.pynccl import PyNcclCommunicator
    from vllm.distributed.utils import StatelessProcessGroup
    pg = StatelessProcessGroup.create(host=master_address,
                                      port=master_port,
                                      rank=rank,
                                      world_size=world_size)
    pynccl = PyNcclCommunicator(pg, device=device)
    return pynccl


class WorkerWrap(Worker):
    def init_process_group(
        self, master_address, master_port, rank_offset, world_size, group_name,
        backend="nccl", use_ray=False
    ):

        assert torch.distributed.is_initialized(
        ), f"default torch process group must be initialized"
        assert group_name != "", f"group name must not be empty"

        rank = torch.distributed.get_rank() + rank_offset

        print(f"init_process_group: master_address={master_address}, "
              f"master_port={master_port}, ",
              f"rank={rank}, world_size={world_size}, group_name={group_name}",)
        if use_ray:
            import ray.util.collective as collective

            collective.init_collective_group(
                world_size=world_size, rank=rank,
                backend=backend, group_name=group_name)

            self._model_update_group = group_name
        else:

            self._model_update_group = init_process_group(
                backend=backend,
                init_method=f"tcp://{master_address}:{master_port}",
                world_size=world_size,
                rank=rank,
                group_name=group_name,
            )
        self._model_update_with_ray = use_ray
        print(f"DONE init_process_group: master_address={master_address}, "
              f"master_port={master_port}, ",
              f"rank={rank}, world_size={world_size}, group_name={group_name}",)

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
        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def receive_weight_s(self, name, dtype, shape):
        weight = torch.empty(shape, dtype=dtype, device="cuda")
        self.model_update_group.broadcast(weight,
                                          src=0,
                                          stream=torch.cuda.current_stream())

        self.model_runner.model.load_weights(weights=[(name, weight)])

        del weight

    def check_weights_changed_s(self):

        weights_updated = True
        for name, p in self.model_runner.model.named_parameters():
            weights_updated = weights_updated and torch.allclose(
                p, torch.zeros_like(p))
        return weights_updated
