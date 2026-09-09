import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


model_name = "Qwen/Qwen2.5-0.5B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

messages = [
    {
        "role": "user",
        "content": "请介绍一下南京大学，不超过100字。"
    }
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(model.device)


# ============================================================
# 实验1：Greedy
# ============================================================

print("\n========== Greedy ==========")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=False,
)

generated_ids = outputs[
    0,
    inputs["input_ids"].shape[-1]:
]

print(
    tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
)


# ============================================================
# 实验2：Sampling + temperature=1.0 + top_p=0.5
# ============================================================

print("\n========== T=1.0 + top_p=0.5 ==========")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
    top_p=0.5,
)

generated_ids = outputs[
    0,
    inputs["input_ids"].shape[-1]:
]

print(
    tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
)


# ============================================================
# 实验3：Sampling + temperature=1.0 + top_p=0.8
# ============================================================

print("\n========== T=1.0 + top_p=0.8 ==========")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
    top_p = 0.8,
)

generated_ids = outputs[
    0,
    inputs["input_ids"].shape[-1]:
]

print(
    tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
)


# ============================================================
# 实验4：Sampling + temperature=1.0 + top_p=0.9
# ============================================================

print("\n========== T=1.0 + top_p=0.9 ==========")

outputs = model.generate(
    **inputs,
    max_new_tokens=100,
    do_sample=True,
    temperature=1.0,
    top_p = 0.9,
)

generated_ids = outputs[
    0,
    inputs["input_ids"].shape[-1]:
]

print(
    tokenizer.decode(
        generated_ids,
        skip_special_tokens=True
    )
)