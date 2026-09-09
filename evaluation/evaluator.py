import re


EVALUATOR_VERSION = "v4"


def extract_gold(answer):
    """
    GSM8K 官方答案：
    #### 后面的内容就是标准答案。
    """

    if "####" not in answer:
        raise ValueError(
            "GSM8K answer does not contain ####"
        )

    gold = answer.split(
        "####",
        1
    )[1].strip()

    # GSM8K 最终答案通常是整数。
    gold = gold.replace(",", "")

    return gold


def extract_prediction(response):
    """
    尽可能可靠地提取模型最终答案。

    优先级：
    1. #### 42
    2. \\boxed{42}
    3. \\boxed{42 \\text{ min}}
    4. Final answer: 42 / Answer is 42
    5. 最后一行是一个纯数字
    6. 否则返回 None

    原则：不确定就返回 None（判为 invalid），绝不猜答案。
    """

    response = response.strip()

    # markdown 加粗只影响可读性，全局去掉。
    # 让 "answer is **56**" 也能匹配。
    response = response.replace("**", "")

    # ========================================================
    # 1. GSM8K 风格
    #    #### 42
    # ========================================================

    matches = re.findall(
        r"####\s*(-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+))",
        response
    )

    if matches:
        return matches[-1].replace(",", "")

    # ========================================================
    # 2. LaTeX \boxed{}
    #
    # 支持：
    # \boxed{42}
    # \boxed{180 \text{ min}}
    # \boxed{\$460}
    # ========================================================

    fraction_matches = re.findall(
        r"\\boxed\{\s*(-?)\\frac\{\s*(\d+)\s*\}"
        r"\{\s*(\d+)\s*\}"
        r"(?:\s*\\text\{[^}]*\})?\s*\}",
        response,
    )

    if fraction_matches:
        sign, numerator, denominator = fraction_matches[-1]
        return f"{sign}{numerator}/{denominator}"

    matches = re.findall(
        r"\\boxed\{\s*"
        r"(?:\\\$)?"
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?:\s*\\text\{[^}]*\})?"
        r"\s*\}",
        response
    )

    if matches:
        return matches[-1].replace(",", "")

    # ========================================================
    # 3. Final answer: 42
    #    Answer is 42
    # ========================================================

    pattern = (
        r"(?:final answer|answer)"
        r"\s*(?:is|:)\s*"
        r"(?:approximately\s+)?"
        r"(?:\\\(\s*)?"
        r"(?:\$|\\\$)?"
        r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
        r"(?:\s*\\\))?\s*%?"
    )

    matches = re.findall(
        pattern,
        response,
        flags=re.IGNORECASE
    )

    if matches:
        return matches[-1].replace(",", "")

    # ========================================================
    # 4. 最后一行是纯数字
    # ========================================================

    lines = [
        line.strip()
        for line in response.splitlines()
        if line.strip()
    ]

    if lines:

        last_line = lines[-1]

        # A single number in an explicit concluding sentence is reliable even
        # when the model did not obey the requested #### format, for example:
        #   "Therefore, Lloyd earned $130 for the first two weeks."
        # Reject lines with multiple numbers or an explicit disclaimer so this
        # cannot silently become a "last number in the reasoning" fallback.
        if re.match(r"^(?:therefore|thus|hence|so)\b", last_line, re.IGNORECASE):
            if not re.search(r"\b(?:however|error|cannot|can't|unclear)\b", last_line, re.IGNORECASE):
                concluding_numbers = re.findall(
                    r"(?<![\d.])-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?!\d)",
                    last_line,
                )
                if len(concluding_numbers) == 1:
                    return concluding_numbers[0].replace(",", "")

        match = re.fullmatch(
            r"-?\d+(?:,\d{3})*(?:\.\d+)?[。.]?",
            last_line
        )

        if match:
            return (
                last_line
                .rstrip("。.")
                .replace(",", "")
            )

        # 最后一行以提交式语句结尾：
        #   "... is 3." / "Total = $130,000"
        # 数字必须在行尾，且只接受 is / total / answer 作为信号。
        # 不接受裸的 "=" 或 ":"（容易抓推理中间步骤，如 "Step 1: 12"）。
        last = last_line

        m = re.search(
            r"\bis\b\s*\$?"
            r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
            r"\s*[。.]?$",
            last,
            flags=re.IGNORECASE,
        )

        if m:
            return m.group(1).replace(",", "")

        # 明确的句末提交式答案，允许普通单位和 LaTeX 行内括号：
        #   "... is 320 miles."
        #   "... is approximately \(8.57\) mph."
        m = re.search(
            r"\bis\s+(?:approximately\s+)?"
            r"(?:\\\(\s*)?(?:\$|\\\$)?"
            r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
            r"(?:\s*\\\))?"
            r"(?:\s+[A-Za-z][A-Za-z /-]*)?"
            r"\s*[。.]?$",
            last,
            flags=re.IGNORECASE,
        )

        if m:
            return m.group(1).replace(",", "")

        if re.search(
            r"\b(?:total|answer)\b",
            last,
            flags=re.IGNORECASE,
        ):

            m = re.search(
                r"(-?\d+(?:,\d{3})*(?:\.\d+)?)"
                r"\s*[。.]?$",
                last,
            )

            if m:
                return m.group(1).replace(",", "")

    # ========================================================
    # 5. 无法可靠判断
    # ========================================================

    return None


def is_format_compliant(response):
    """Whether the final non-empty line exactly follows `#### NUMBER`."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    if not lines:
        return False
    return re.fullmatch(
        r"####\s*-?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)",
        lines[-1],
    ) is not None

def _as_number(text):
    """尝试把字符串解析为数字，失败返回 None。"""
    try:
        normalized = text.strip().replace(",", "")
        if "/" in normalized:
            numerator, denominator = normalized.split("/", 1)
            return float(numerator) / float(denominator)
        return float(normalized)
    except (ValueError, ZeroDivisionError):
        return None


def check_answer(
    prediction,
    gold
):
    """
    先 Exact Match；两边都是数字时再做数值比较。
    例如 gold="18" 与 pred="18.0" 应算对。
    """

    if prediction is None:
        return False

    if (
        prediction.strip()
        == gold.strip()
    ):
        return True

    pred_num = _as_number(prediction)
    gold_num = _as_number(gold)

    if (
        pred_num is not None
        and gold_num is not None
    ):
        return pred_num == gold_num

    return False
