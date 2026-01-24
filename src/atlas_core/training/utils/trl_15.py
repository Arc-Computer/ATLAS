

import itertools
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal, Optional, Union

from accelerate.utils import is_deepspeed_available
from packaging import version
from transformers import PreTrainedModel, PreTrainedTokenizer
from transformers.utils.deprecation import deprecate_kwarg

from trl.models.modeling_value_head import (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)


SUPPORTED_ARCHITECTURES = (
    AutoModelForCausalLMWithValueHead,
    AutoModelForSeq2SeqLMWithValueHead,
)

if is_deepspeed_available():
    import deepspeed

if TYPE_CHECKING:
    from accelerate import Accelerator
    from deepspeed.runtime.engine import DeepSpeedEngine
    from torch.nn.parallel.distributed import DistributedDataParallel


@dataclass
class ChatMlSpecialTokens:

    bos_token: str = "<|im_start|>"
    eos_token: str = "<|im_end|>"
    pad_token: str = "<|im_end|>"

    @property
    def system(self):
        return f"{self.bos_token}system"

    @property
    def user(self):
        return f"{self.bos_token}user"

    @property
    def assistant(self):
        return f"{self.bos_token}assistant"

    @property
    def chat_template(self):
        return (
            "{% for message in messages %}"
            f"{{{{'{self.bos_token}' + message['role'] + '\n' + message['content'] + '{self.eos_token}' + '\n'}}}}"
            "{% endfor %}"
            "{% if add_generation_prompt %}"
            f"{{{{ '{self.assistant}\n' }}}}"
            "{% endif %}"
        )


FORMAT_MAPPING = {"chatml": ChatMlSpecialTokens}


def setup_chat_format(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    format: Optional[Literal["chatml"]] = "chatml",
    resize_to_multiple_of: Optional[int] = None,
) -> tuple[PreTrainedModel, PreTrainedTokenizer]:

    if tokenizer.chat_template is not None:
        raise ValueError(
            "Chat template is already added to the tokenizer. If you want to overwrite it, please set it to None"
        )

    if format not in FORMAT_MAPPING:
        raise ValueError(
            f"Format {format} not available. Please use one of {FORMAT_MAPPING.keys()}")

    chat_format = FORMAT_MAPPING[format]()

    tokenizer.eos_token = chat_format.eos_token
    tokenizer.pad_token = chat_format.pad_token
    tokenizer.bos_token = chat_format.bos_token
    tokenizer.add_special_tokens(
        {"additional_special_tokens": [chat_format.bos_token, chat_format.eos_token]})

    tokenizer.chat_template = chat_format.chat_template

    model.resize_token_embeddings(
        len(tokenizer), pad_to_multiple_of=resize_to_multiple_of if resize_to_multiple_of is not None else None
    )

    if getattr(model, "config", None) is not None:
        model.config.pad_token_id = tokenizer.pad_token_id
        model.config.bos_token_id = tokenizer.bos_token_id
        model.config.eos_token_id = tokenizer.eos_token_id

    if getattr(model, "generation_config", None) is not None:
        model.generation_config.bos_token_id = tokenizer.bos_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.pad_token_id = tokenizer.pad_token_id

    return model, tokenizer


def remove_hooks(model: "DeepSpeedEngine") -> None:

    if not hasattr(model, "optimizer"):
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError(
            "The model optimizer is None, which is not yet supported.")

    for param in iter_params(optimizer_offload.module, recurse=True):
        param.ds_active_sub_modules.clear()

    for hook in optimizer_offload.forward_hooks:
        hook.remove()
    for hook in optimizer_offload.backward_hooks:
        hook.remove()

    optimizer_offload.forward_hooks = []
    optimizer_offload.backward_hooks = []


def get_all_parameters(sub_module, recurse=False):
    return itertools.chain(sub_module.named_parameters(recurse=recurse), sub_module.ds_external_parameters())


def iter_params(module, recurse=False):
    return [param for _, param in get_all_parameters(module, recurse)]


def add_hooks(model: "DeepSpeedEngine") -> None:

    if not hasattr(model, "optimizer"):
        return
    if model.optimizer is not None and hasattr(model.optimizer, "parameter_offload"):
        optimizer_offload = model.optimizer.parameter_offload
    elif model.optimizer is not None:
        optimizer_offload = model.optimizer
    else:
        raise RuntimeError(
            "The model optimizer is None, which is not yet supported.")

    if version.parse(deepspeed.__version__) >= version.parse("0.16.4"):

        optimizer_offload._register_deepspeed_module(optimizer_offload.module)
    else:
        optimizer_offload._register_hooks_recursively(optimizer_offload.module)


@contextmanager
@deprecate_kwarg("is_peft_model", "0.16.0", warn_if_greater_or_equal_version=True)
def unwrap_model_for_generation(
    model: Union["DistributedDataParallel", "DeepSpeedEngine"],
    accelerator: "Accelerator",
    gather_deepspeed3_params: bool = True,
):

    if accelerator.state.deepspeed_plugin is not None and accelerator.state.deepspeed_plugin.zero_stage == 3:
        if not gather_deepspeed3_params:
            yield accelerator.unwrap_model(model)
        else:

            with deepspeed.zero.GatheredParameters(model.parameters()):
                remove_hooks(model)
                yield accelerator.unwrap_model(model)
                add_hooks(model)
    else:
        unwrapped_model = accelerator.unwrap_model(model)
        yield unwrapped_model


def prepare_deepspeed(model, accelerator, offload_to_cpu=False):

    deepspeed_plugin = accelerator.state.deepspeed_plugin
    config_kwargs = deepcopy(deepspeed_plugin.deepspeed_config)
    stage = config_kwargs["zero_optimization"]["stage"]

    if model is not None:
        hidden_size = (
            max(model.config.hidden_sizes)
            if getattr(model.config, "hidden_sizes", None)
            else getattr(model.config, "hidden_size", None)
        )
        if hidden_size is not None and stage == 3:

            config_kwargs.update(
                {
                    "zero_optimization.reduce_bucket_size": hidden_size * hidden_size,
                    "zero_optimization.stage3_param_persistence_threshold": 10 * hidden_size,
                    "zero_optimization.stage3_prefetch_bucket_size": 0.9 * hidden_size * hidden_size,
                }
            )

    if offload_to_cpu:

        config_kwargs[
            "zero_optimization"]["offload_optimizer"]["device"] = "cpu"
        config_kwargs["zero_optimization"]["offload_param"]["device"] = "cpu"

    if stage != 3:
        config_kwargs["zero_optimization"]["stage"] = 0
    model, *_ = deepspeed.initialize(model=model, config=config_kwargs)
    model.eval()
    return model
