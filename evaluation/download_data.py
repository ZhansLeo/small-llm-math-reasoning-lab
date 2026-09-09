from datasets import load_dataset


dataset = load_dataset(
    "openai/gsm8k",
    "main"
)

print(dataset)
print(dataset["train"][0])
print(dataset["test"][0])

# 保存成 JSONL
dataset["train"].to_json(
    "data/gsm8k/train.jsonl",
    force_ascii=False
)

dataset["test"].to_json(
    "data/gsm8k/test.jsonl",
    force_ascii=False
)

print("GSM8K downloaded.")