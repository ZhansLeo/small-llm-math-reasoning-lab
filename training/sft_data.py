"""Tokenization and assistant-only causal-LM collation for SFT."""

import json
from pathlib import Path

import torch
from torch.utils.data import Dataset


class GSM8KSFTDataset(Dataset):
    def __init__(self, path, tokenizer, build_messages, prompt_name, max_seq_length):
        self.tokenizer = tokenizer
        self.build_messages = build_messages
        self.prompt_name = prompt_name
        self.max_seq_length = max_seq_length
        with Path(path).open("r", encoding="utf-8") as f:
            self.rows = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        user_messages = self.build_messages(row["question"], self.prompt_name)
        prompt = self.tokenizer.apply_chat_template(
            user_messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
        )["input_ids"]
        full_messages = user_messages + [
            {"role": "assistant", "content": row["assistant_answer"]}
        ]
        full = self.tokenizer.apply_chat_template(
            full_messages,
            tokenize=True,
            add_generation_prompt=False,
            return_dict=True,
        )["input_ids"]

        if full[: len(prompt)] != prompt:
            raise RuntimeError("Chat template prompt is not a prefix of the SFT example")
        if len(full) > self.max_seq_length:
            raise RuntimeError(
                f"Prepared example {row['dataset_index']} exceeds max_seq_length"
            )

        labels = [-100] * len(prompt) + full[len(prompt) :]
        if all(label == -100 for label in labels):
            raise RuntimeError("No assistant tokens remain for loss computation")
        return {
            "input_ids": full,
            "attention_mask": [1] * len(full),
            "labels": labels,
        }


class SFTDataCollator:
    def __init__(self, pad_token_id):
        self.pad_token_id = pad_token_id

    def __call__(self, features):
        max_length = max(len(item["input_ids"]) for item in features)
        input_ids = []
        attention_mask = []
        labels = []
        for item in features:
            padding = max_length - len(item["input_ids"])
            input_ids.append(item["input_ids"] + [self.pad_token_id] * padding)
            attention_mask.append(item["attention_mask"] + [0] * padding)
            labels.append(item["labels"] + [-100] * padding)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }
