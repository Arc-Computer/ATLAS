import types

import pytest
torch = pytest.importorskip("torch")
from datasets import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import GKDConfig
from trl.trainer.utils import DataCollatorForChatML

from trainers.gkd_trainer import AlignedChatCollator, AtlasGKDTrainer
from trainers.postgres_runtime_dataset import session_to_conversation


def _ensure_pad_token(tokenizer):
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    return tokenizer


def _sample_messages():
    return [
        {"role": "system", "content": "Be precise."},
        {"role": "user", "content": "Say hello"},
        {"role": "assistant", "content": "Hello!"},
    ]


def test_session_to_conversation_adds_prompt_and_completion():
    session = types.SimpleNamespace()
    session.task = "Wave"
    session.plan = {"steps": [{"id": 1, "description": "greet"}]}
    session.trajectory_events = []
    session.steps = [
        {"guidance": ["Consider context."], "output": "Hello!", "trace": "result"},
    ]
    session.session_metadata = {}
    session.session_reward = None
    record = session_to_conversation(session)
    assert record is not None
    assert "prompt_text" in record and record["prompt_text"]
    assert record["completion_text"] == "Hello!"


def test_aligned_chat_collator_handles_dual_tokenizers():
    student_tok = _ensure_pad_token(AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-gpt2"))
    teacher_tok = _ensure_pad_token(AutoTokenizer.from_pretrained("hf-internal-testing/tiny-random-llama"))
    student_collator = DataCollatorForChatML(tokenizer=student_tok, max_length=128)
    teacher_collator = DataCollatorForChatML(tokenizer=teacher_tok, max_length=128)
    collator = AlignedChatCollator(student_collator, teacher_collator)

    sample = {
        "messages": _sample_messages(),
        "prompt_text": "system: Be precise.\nuser: Say hello",
        "completion_text": "Hello!",
    }
    batch = collator([sample])
    assert "teacher_input_ids" in batch
    assert batch["teacher_input_ids"].shape[0] == batch["input_ids"].shape[0]
    assert batch["teacher_prompts"].shape[0] == batch["prompts"].shape[0]
    assert batch["completion_text"] == ["Hello!"]


def test_atlas_gkd_trainer_aligns_teacher_logits(tmp_path):
    student_model_name = "hf-internal-testing/tiny-random-gpt2"
    teacher_model_name = "hf-internal-testing/tiny-random-llama"

    tokenizer = _ensure_pad_token(AutoTokenizer.from_pretrained(student_model_name))
    student_model = AutoModelForCausalLM.from_pretrained(student_model_name)
    teacher_model = AutoModelForCausalLM.from_pretrained(teacher_model_name)

    dataset = Dataset.from_list(
        [
            {
                "messages": _sample_messages(),
                "prompt_text": "system: Be precise.\nuser: Say hello",
                "completion_text": "Hello!",
            }
        ]
    )

    gkd_args = GKDConfig(
        output_dir=str(tmp_path / "outputs"),
        per_device_train_batch_size=1,
        per_device_eval_batch_size=1,
        max_steps=1,
        eval_strategy="no",
        save_strategy="no",
        logging_steps=1,
    )

    trainer = AtlasGKDTrainer(
        model=student_model,
        teacher_model=teacher_model,
        args=gkd_args,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=dataset,
        align_teacher_template=True,
        teacher_tokenizer_name_or_path=teacher_model_name,
    )

    batch = trainer.data_collator(
        [
            {
                "messages": _sample_messages(),
                "prompt_text": "system: Be precise.\nuser: Say hello",
                "completion_text": "Hello!",
            }
        ]
    )
    loss = trainer.compute_loss(trainer.model, batch)
    assert torch.isfinite(loss)
