from dataclasses import dataclass, field
from typing import Optional, Union

from transformers import TrainingArguments


@dataclass
class GRPOConfig(TrainingArguments):

    model_init_kwargs: Optional[dict] = field(
        default=None,
        metadata={
            "help": "Keyword arguments for `transformers.AutoModelForCausalLM.from_pretrained`, used when the `model` "
            "argument of the `GRPOTrainer` is provided as a string."
        },
    )

    remove_unused_columns: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to only keep the column 'prompt' in the dataset. If you use a custom reward function "
            "that requires any column other than 'prompts' and 'completions', you should keep this to `False`."
        },
    )
    max_prompt_length: Optional[int] = field(
        default=512,
        metadata={
            "help": "Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left."
        },
    )
    num_generations: Optional[int] = field(
        default=8,
        metadata={
            "help": "Number of generations to sample. The global batch size (num_processes * per_device_batch_size) "
            "must be divisible by this value."
        },
    )
    max_completion_length: Optional[int] = field(
        default=256,
        metadata={"help": "Maximum length of the generated completion."},
    )

    ds3_gather_for_generation: bool = field(
        default=True,
        metadata={
            "help": "This setting applies to DeepSpeed ZeRO-3. If enabled, the policy model weights are gathered for "
            "generation, improving generation speed. However, disabling this option allows training models that "
            "exceed the VRAM capacity of a single GPU, albeit at the cost of slower generation. Disabling this option "
            "is not compatible with vLLM generation."
        },
    )
    shuffle_generation_inputs: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Randomly shuffle the prompts to distribute uneven loads "
            "at the cost of less prefix caching"
        },
    )

    temperature: float = field(
        default=0.9,
        metadata={
            "help": "Temperature for sampling. The higher the temperature, the more random the completions."},
    )
    top_p: float = field(
        default=1.0,
        metadata={
            "help": "Float that controls the cumulative probability of the top tokens to consider. Must be in (0, 1]. "
            "Set to 1.0 to consider all tokens."
        },
    )
    top_k: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of highest probability vocabulary tokens to keep for top-k-filtering. If `None`, "
            "top-k-filtering is disabled."
        },
    )
    min_p: Optional[float] = field(
        default=None,
        metadata={
            "help": "Minimum token probability, which will be scaled by the probability of the most likely token. It "
            "must be a value between 0.0 and 1.0. Typical values are in the 0.01-0.2 range."
        },
    )
    repetition_penalty: float = field(
        default=1.0,
        metadata={
            "help": "Float that penalizes new tokens based on whether they appear in the prompt and the generated "
            "text so far. Values > 1.0 encourage the model to use new tokens, while values < 1.0 encourage the model "
            "to repeat tokens."
        },
    )
    generation_aggregation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Aggregates the generations to do across accumulation steps"
            "and allows for more efficient generation (due to higher VLLM GPU"
            "utilization) and removing constraints for num_generations."},
    )

    use_vllm: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM for generating completions. If set to `True`, ensure that a GPU is kept "
            "unused for training, as vLLM will require one for generation. vLLM must be installed "
            "(`pip install vllm`)."
        },
    )
    vllm_device: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Device where vLLM generation will run, e.g. 'cuda:1'. If set to 'auto' (default), the system "
            "will automatically select the next available GPU after the last one used for training. This assumes "
            "that training has not already occupied all available GPUs."
        },
    )
    vllm_gpu_memory_utilization: float = field(
        default=0.9,
        metadata={
            "help": "Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV "
            "cache on the device dedicated to generation powered by vLLM. Higher values will increase the KV cache "
            "size and thus improve the model's throughput. However, if the value is too high, it may cause "
            "out-of-memory (OOM) errors during initialization."
        },
    )
    vllm_dtype: Optional[str] = field(
        default="auto",
        metadata={
            "help": "Data type to use for vLLM generation. If set to 'auto', the data type will be automatically "
            "determined based on the model configuration. Find the supported values in the vLLM documentation."
        },
    )
    vllm_max_model_len: Optional[int] = field(
        default=None,
        metadata={
            "help": "If set, the `max_model_len` to use for vLLM. This could be useful when running with reduced "
            "`vllm_gpu_memory_utilization`, leading to a reduced KV cache size. If not set, vLLM will use the model "
            "context size, which might be much larger than the KV cache, leading to inefficiencies."
        },
    )

    use_vllm_server: bool = field(
        default=False,
        metadata={
            "help": "Whether to use a vLLM server for generating completions."
        },
    )
    vllm_host: Optional[str] = field(
        default=None,
        metadata={
            "help": "Host of the vLLM server."
        },
    )
    vllm_port: Optional[int] = field(
        default=None,
        metadata={
            "help": "Port of the vLLM server."
        },
    )
    vllm_group_port: Optional[int] = field(
        default=51216,
        metadata={
            "help": "Port of the vLLM server."
        },
    )
    vllm_server_timeout: float = field(
        default=120.0,
        metadata={
            "help": "Total timeout duration in seconds to wait for the vLLM server to be up. If the server is not up "
            "after the timeout, a `ConnectionError` is raised."
        },
    )
    num_vllm_clients: Optional[int] = field(
        default=1,
        metadata={
            "help": "Number of vLLM clients to use."
        },
    )
    vllm_guided_decoding_regex: Optional[str] = field(
        default=None,
        metadata={
            "help": "Regex for vLLM guided decoding. If `None` (default), guided decoding is disabled."},
    )

    use_ray: Optional[bool] = field(
        default=False,
        metadata={
            "help": "Whether to use vLLM + ray for generating completions."
        },
    )
    ray_share_training_devices: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, assumes we are using the same devices for "
            "training and generation with offloading/sleep. "
        },
    )
    ray_tensor_parallelism: int = field(
        default=1,
        metadata={
            "help": "..."
        },
    )
    ray_data_parallelism: Optional[int] = field(
        default=None,
        metadata={
            "help": "..."
        },
    )
    ray_no_memory_duplication: Optional[bool] = field(
        default=False,
        metadata={
            "help": "If True, avoids gpu memory duplication at the cost of a "
            "slight weight loading overhead"
        },
    )

    enforce_eager: bool = field(
        default=True,
        metadata={
            "help": "..."
        },
    )
    vllm_sleep_level: int = field(
        default=1,
        metadata={
            "help": "0 = no offload, 1 = offload to CPU, 2 = offload to disk"
        },
    )
    enable_prefix_caching: bool = field(
        default=False,
        metadata={
            "help": "..."
        },
    )

    learning_rate: float = field(
        default=1e-6,
        metadata={
            "help": "Initial learning rate for `AdamW` optimizer. The default value replaces that of "
            "`transformers.TrainingArguments`."
        },
    )
    beta: float = field(
        default=0.04,
        metadata={"help": "KL coefficient."},
    )
    reward_weights: Optional[list[float]] = field(
        default=None,
        metadata={
            "help": "Weights for each reward function. Must match the number of reward functions. If `None`, all "
            "rewards are weighted equally with weight `1.0`."
        },
    )
    sync_ref_model: bool = field(
        default=False,
        metadata={
            "help": "Whether to synchronize the reference model with the active model every `ref_model_sync_steps` "
            "steps, using the `ref_model_mixup_alpha` parameter."
        },
    )
    ref_model_mixup_alpha: float = field(
        default=0.9,
        metadata={
            "help": "α parameter from the TR-DPO paper, which controls the mix between the current policy and the "
            "previous reference policy during updates. The reference policy is updated according to the equation: "
            "`π_ref = α * π_θ + (1 - α) * π_ref_prev`. To use this parameter, you must set `sync_ref_model=True`."
        },
    )
    ref_model_sync_steps: int = field(
        default=64,
        metadata={
            "help": "τ parameter from the TR-DPO paper, which determines how frequently the current policy is "
            "synchronized with the reference policy. To use this parameter, you must set `sync_ref_model=True`."
        },
    )

    backprop_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={
            "help": "Accumulates the loss, only during backprop model "
            "computations, to allow disentangling num_generations from the "
            "global micro batch size."},
    )

    backprop_accumulation_micro_batch_size: Optional[int] = field(
        default=None,
        metadata={
            "help": "Alternative way of specifying loss accumulation, in terms "
            "of the max per-device batch that can fit into memory during "
            "backpropation."},
    )

    offload_untrained_models: bool = field(
        default=False,
        metadata={"help": "Allows performing parameter offloading of the "
                  "reference and reward models to minimize memory usage."},
    )

    unbias_log_probabilities: bool = field(
        default=False,
        metadata={
            "help": "Unbias log probabilities by appropriately scaling them by the temperature parameter."},
    )

    activate_debugging_logs: bool = field(
        default=False,
        metadata={"help": "Whether to activate debugging print logs for grpo."},
    )

    log_completions: bool = field(
        default=False,
        metadata={
            "help": "Whether to log the completions during training to wandb."},
    )

    save_completions_probability: Optional[float] = field(
        default=None,
        metadata={
            "help": "Probability to store smaple completions to file at the end of a training step."},
    )

    artificial_epochs: Optional[int] = field(
        default=1,
        metadata={"help": "Number of actual epochs we want to do training for. Specifying here to avoid"
                  " a bug in huggingface's GRPO training with multiple epochs."},
    )

    max_probe_tokens: Optional[int] = field(
        default=500,
        metadata={"help": "Maximum number of tokens for student diagnostic probing."},
    )

    student_diagnostic_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for generating student diagnostic probes."},
    )

    teacher_adaptive_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for generating teacher adaptive teaching."},
    )

    student_with_teaching_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for student solution with teaching."},
    )

    student_baseline_template: Optional[str] = field(
        default=None,
        metadata={"help": "Template for student baseline solution."},
    )
