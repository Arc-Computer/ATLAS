from datasets import load_dataset, concatenate_datasets, Dataset
from huggingface_hub import hf_hub_download
import json
from .utils import make_masked_sft_collator
from .datasets_info import (
    DATA_CONFIGS, DataConfig, ADAPTIVE_TEACHING_SYSTEM_PROMPT)


def add_indices(ds):
    if "__index" not in ds.column_names:
        ds = ds.map(lambda x, i: {"__index": i}, with_indices=True)
    return ds


def get_process_line_fn(dataset_id_or_path):
    data: DataConfig = DATA_CONFIGS[dataset_id_or_path]
    system_prompt = data.system_prompt

    def process_line_fn(line, tokenizer):
        question_content, thought_process_and_solution = (
            data.extract_question_and_completion_from_line(line))
        messages = [
            {
                "role": "system",
                "content": system_prompt,
            },
            {
                "role": "user",
                "content": question_content,
            },
            {
                "role": "assistant",
                "content": thought_process_and_solution,
            }
        ]
        line_text = tokenizer.apply_chat_template(
            messages, tokenize=False, continue_final_message=False)
        return {"text": line_text}
    return process_line_fn


def load_formatted_sft_dataset(
        tokenizer,
        dataset_id_or_path,
        dataset_local_directory=None,
        train_split='train',
        val_split=None,
        process_line_fn=None,
        model_name_or_path=None,
        completion_only_training=True,
        custom_start_of_response=None,
        keep_columns=None,
        add_dataset_indices=False,
        artificial_epochs=None,
        **dataset_loading_kwargs,
):

    if dataset_id_or_path == "Arc-Intelligence/Arc-ATLAS-Teach-v0":
        sft_path = hf_hub_download(
            repo_id="Arc-Intelligence/Arc-ATLAS-Teach-v0",
            filename="training/sft.jsonl",
            repo_type="dataset"
        )

        with open(sft_path, 'r') as f:
            sft_data = [json.loads(line) for line in f]

        def process_arc_atlas_line(item, tokenizer):
            assistant_content = (
                f"Student Approach: {item['student_approach']}\n\n"
                f"Teacher Diagnosis: {item['teacher_diagnosis']}\n\n"
                f"Teacher Teaching: {item['teacher_teaching']}"
            )
            messages = [
                {
                    "role": "system",
                    "content": ADAPTIVE_TEACHING_SYSTEM_PROMPT,
                },
                {
                    "role": "user",
                    "content": item["problem_text"],
                },
                {
                    "role": "assistant",
                    "content": assistant_content,
                }
            ]
            line_text = tokenizer.apply_chat_template(
                messages, tokenize=False, continue_final_message=False)
            return {"text": line_text}

        processed_data = [process_arc_atlas_line(item, tokenizer) for item in sft_data]
        train_dataset = Dataset.from_list(processed_data)
        val_dataset = None
    else:
        if dataset_local_directory is None:
            dataset_local_directory = dataset_id_or_path
        dataset = load_dataset(dataset_local_directory, **dataset_loading_kwargs)
        train_dataset = dataset[train_split]
        if val_split is not None:
            val_dataset = dataset[val_split]
        else:
            val_dataset = None

    if add_dataset_indices:
        train_dataset = add_indices(train_dataset)

    if dataset_id_or_path != "Arc-Intelligence/Arc-ATLAS-Teach-v0" and process_line_fn is not None:
        if isinstance(process_line_fn, (list, tuple)):
            processed_train_datasets = []
            for fn in process_line_fn:
                processed = train_dataset.map(
                    lambda x, fn=fn: fn(x, tokenizer))
                processed_train_datasets.append(processed)
            train_dataset = concatenate_datasets(
                processed_train_datasets)
        else:
            print('not loading from cache')
            train_dataset = train_dataset.map(
                lambda x: process_line_fn(x,  tokenizer))

        if val_dataset is not None:
            if add_dataset_indices:
                val_dataset = add_indices(val_dataset)
            if process_line_fn is not None:
                if isinstance(process_line_fn, (list, tuple)):
                    processed_val_datasets = []
                    for fn in process_line_fn:
                        processed = val_dataset.map(
                            lambda x, fn=fn: fn(x, tokenizer))
                        processed_val_datasets.append(processed)
                    val_dataset = concatenate_datasets(
                        processed_val_datasets)
                else:
                    val_dataset = val_dataset.map(
                        lambda x: process_line_fn(x,  tokenizer))
    if keep_columns is not None:
        train_dataset = train_dataset.remove_columns(
            [col for col in train_dataset.column_names
             if col not in keep_columns])

    if artificial_epochs is not None:
        assert artificial_epochs == 1, (
            'Artificial epoch, moved to GRPO to avoid shuffling samples between'
            ' different epochs.')

        train_dataset = concatenate_datasets(
            [train_dataset]*artificial_epochs)
    out_data = dict(
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    if completion_only_training:
        out_data['data_collator'] = make_masked_sft_collator(
            tokenizer=tokenizer,
            model_name=model_name_or_path,
            custom_start_of_response=custom_start_of_response,
        )
    return out_data
