"""Versioned prompt registry shared by every inference backend."""

PROMPT_VERSION = "v1"


PROMPT_TEMPLATES = {
    "prompt_a": (
        "Solve the following math problem.\n\n"
        "{question}\n\n"
        "Return the final numeric answer on the last line in exactly this format:\n"
        "#### 42\n"
        "Replace 42 with the answer. Do not write anything after that line."
    ),
    "prompt_b": (
        "Solve the following math problem carefully.\n\n"
        "{question}\n\n"
        "Work through the problem step by step. Then return the final numeric "
        "answer on the last line in exactly this format:\n"
        "#### 42\n"
        "Replace 42 with the answer. Do not write anything after that line."
    ),
    "prompt_c": (
        "Solve the following math problem using this structure:\n"
        "1. Identify known quantities.\n"
        "2. Identify relationships.\n"
        "3. Identify the target.\n"
        "4. Perform calculations.\n"
        "5. Give the final answer.\n\n"
        "{question}\n\n"
        "Return the final numeric answer on the last line in exactly this format:\n"
        "#### 42\n"
        "Replace 42 with the answer. Do not write anything after that line."
    ),
}


def get_prompt_template(prompt_name):
    """Return a registered prompt template or raise a useful error."""
    try:
        return PROMPT_TEMPLATES[prompt_name]
    except KeyError as exc:
        choices = ", ".join(sorted(PROMPT_TEMPLATES))
        raise ValueError(
            f"Unknown prompt_name={prompt_name!r}. Choose one of: {choices}"
        ) from exc


def build_prompt(question, prompt_name="prompt_a"):
    return get_prompt_template(prompt_name).format(question=question)


def build_messages(question, prompt_name="prompt_a"):
    return [
        {
            "role": "user",
            "content": build_prompt(question, prompt_name),
        }
    ]


def prompt_metadata(prompt_name):
    return {
        "name": prompt_name,
        "version": PROMPT_VERSION,
        "template": get_prompt_template(prompt_name),
    }
