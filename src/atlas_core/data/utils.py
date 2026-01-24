from transformers import AutoTokenizer, PreTrainedTokenizerBase, AutoConfig
import os
import trl
from datasets import load_dataset


def get_special_token_values(tokenizer, model_name):
    config = AutoConfig.from_pretrained(model_name)
    model_type = getattr(config, 'model_type', None)

    if model_type == 'llama':
        instruction_template = "<|start_header_id|>user<|end_header_id|>"
        response_template = (
            "<|start_header_id|>assistant<|end_header_id|>\n\n")
    elif model_type == 'qwen2':
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
    elif model_type == 'qwen3':
        instruction_template = "<|im_start|>user"
        response_template = "<|im_start|>assistant\n"
    else:
        raise NotImplementedError

    return tokenizer, instruction_template, response_template


def make_masked_sft_collator(
        tokenizer, model_name, custom_start_of_response=None):
    (tokenizer, instruction_template, response_template
     ) = get_special_token_values(tokenizer=tokenizer, model_name=model_name)
    if custom_start_of_response is not None:
        response_template = custom_start_of_response
    print('CHECK RESP. TEMPLATE')
    print(response_template)
    assert tokenizer.pad_token_id is not None
    assert tokenizer.pad_token_id != tokenizer.eos_token_id
    custom_collator = trl.DataCollatorForCompletionOnlyLM(
        instruction_template=instruction_template,
        response_template=response_template,
        tokenizer=tokenizer,
    )

    def data_collator_fn(*args, **kwargs):
        return custom_collator(*args, **kwargs)

    return data_collator_fn


def override_system_prompt(tokenizer_or_tokenizer_name, new_system_prompt):
    if isinstance(tokenizer_or_tokenizer_name, str):
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_or_tokenizer_name)
        tokenizer.apply_chat_template
    elif isinstance(tokenizer_or_tokenizer_name, PreTrainedTokenizerBase):
        tokenizer = tokenizer_or_tokenizer_name
    else:
        raise NotImplementedError

    if hasattr(tokenizer, "default_system_prompt"):
        tokenizer.default_system_prompt = new_system_prompt
    elif hasattr(tokenizer, "chat_template") and tokenizer.chat_template:
        tokenizer.chat_template = tokenizer.chat_template.replace(
            "{system_prompt}", new_system_prompt
        )
    else:
        print("This tokenizer does not support system prompts.")
        raise NotImplementedError

    return tokenizer
