import datasets
import trl
import torch
import numpy
import transformers
import subprocess
import importlib


def spawn_bash_command(bash_command):
    print(f"Executing {bash_command}")
    try:
        subprocess.run(bash_command, shell=True, check=True)
    except subprocess.CalledProcessError as err:
        print("error during execution:", err)
    except Exception as err:
        print("unexpected error:", err)


def clean_after_training_completion(output_dir):
    bash_command = f"./scripts/clean_past_steps.sh {output_dir} all"
    spawn_bash_command(bash_command)


def fix_pad_token(tokenizer, model_name, unsafe=False):
    if tokenizer.pad_token is None:
        if "Llama" in model_name:
            tokenizer.pad_token = "<|reserved_special_token_5|>"
        elif "Qwen" in model_name:
            tokenizer.pad_token = "<|fim_pad|>"
        else:
            raise NotImplementedError
    else:
        if not unsafe:
            assert tokenizer.pad_token_id != tokenizer.eos_token_id
    return tokenizer


def wrap_as_list(*args, **kwargs):
    to_return = []
    for element in args:
        to_return.append(element)
    for element in kwargs.values():
        to_return.append(element)
    return to_return


def load_function(path: str):
    module_name, attr = path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, attr)
