import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import torch

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)

from prompts import build_messages


MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"


# ============================================================
# Device
# ============================================================

if hasattr(torch, "xpu") and torch.xpu.is_available():
    DEVICE = torch.device("xpu")
    print("Using Intel XPU")
else:
    DEVICE = torch.device("cpu")
    print("Using CPU")


# ============================================================
# Tokenizer
# ============================================================

tokenizer = None
model = None


def _load_model():
    """Lazy loading keeps imports and --help fast."""
    global tokenizer, model

    if tokenizer is not None and model is not None:
        return tokenizer, model

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype="auto",
        local_files_only=True,
    )
    model.to(DEVICE)
    model.eval()

    return tokenizer, model


# ============================================================
# Batch Inference
# ============================================================

@torch.no_grad()
def ask_batch(
    questions,
    max_new_tokens=1024,
    prompt_name="prompt_a",
):
    """
    一次处理多个 GSM8K 问题。

    输入：
        questions = [
            question1,
            question2,
            ...
        ]

    返回：
        responses = [
            response1,
            response2,
            ...
        ]
    """

    tokenizer, model = _load_model()

    messages_list = []

    for question in questions:

        messages = build_messages(question, prompt_name)

        messages_list.append(messages)

    # --------------------------------------------------------
    # Chat template
    # --------------------------------------------------------

    prompts = []

    for messages in messages_list:

        prompt = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        prompts.append(prompt)

    # --------------------------------------------------------
    # Batch tokenize
    # --------------------------------------------------------

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )

    inputs = {
        key: value.to(DEVICE)
        for key, value in inputs.items()
    }

    # --------------------------------------------------------
    # Generation
    # --------------------------------------------------------

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
    )

    # --------------------------------------------------------
    # Decode
    #
    # 因为使用 left padding，所有输入都被 pad 到相同长度
    # input_length。generate() 的输出是：
    #
    #   [pad ... pad | prompt | generated answer]
    #   |----- input_length -----|
    #
    # 每条输入对应的生成内容都从 input_length 开始，
    # 所以可以直接按固定长度切分。
    #
    # 注意：不能用 attention_mask[i].sum() 来切，
    # 那是有效 token 数（对 left padding 不等于输入区长度）。
    # --------------------------------------------------------

    input_length = inputs["input_ids"].shape[1]

    responses = []

    for i in range(len(questions)):

        generated_ids = outputs[
            i,
            input_length:
        ]

        response = tokenizer.decode(
            generated_ids,
            skip_special_tokens=True,
        )

        responses.append(
            response.strip()
        )

    return responses
