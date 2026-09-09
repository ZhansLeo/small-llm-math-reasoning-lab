from evaluator import (
    extract_prediction,
    check_answer,
    is_format_compliant,
)


# ============================================================
# extract_prediction
# ============================================================

extract_tests = [
    # 应成功提取
    ("#### 42", "42"),
    (r"\boxed{42}", "42"),
    (r"\boxed{180 \text{ min}}", "180"),
    (r"\boxed{\$460}", "460"),
    (r"\boxed{-\frac{3}{8}}", "-3/8"),
    ("The final answer is 123.", "123"),
    ("Answer: 88", "88"),
    ("Therefore, the answer is **56**.", "56"),
    ("The final answer is $18.", "18"),
    (r"The final answer is \(60\)%.", "60"),
    ("42", "42"),
    ("#### 1,200", "1200"),
    # 行尾提交式结尾
    (
        "Therefore, the total number of bolts needed "
        "to make one robe is **3**.",
        "3",
    ),
    ("Total cost = $130,000", "130000"),
    ("Thus, the distance covered by each train is 320 miles.", "320"),
    (r"Therefore, the speed is approximately \(8.57\) mph.", "8.57"),
    (r"Therefore, the percentage is \(20\%\).", "20"),
    ("Therefore, Lloyd earned **$130** for two weeks.", "130"),
    ("Thus, Steve needs to grow **28** vines each week.", "28"),
    ("#### .0333", ".0333"),
    # 不应猜，返回 None
    ("Step 1: 12", None),
    ("She sells 9 eggs. She makes 18 dollars.", None),
    ("First multiply 3 by 4, then add 2.", None),
    # 数字不在行尾，不应提取
    ("Janet sells 16 - 3 - 4 = 9 duck eggs a day.", None),
    ("Each duck lays 16 eggs per day.", None),
    ("I hope this helps.", None),
    # A concluding equation is a reliable submission signal; extract its RHS.
    ("Therefore, 3 times 4 is 12.", "12"),
    ("Therefore, the result is -43.55; however, there is an error.", None),
]

all_ok = True

for text, expected in extract_tests:

    got = extract_prediction(text)

    status = "OK" if got == expected else "FAIL"

    if got != expected:
        all_ok = False

    print(
        f"[{status}] {text!r} -> {got!r}"
        f"{'' if got == expected else '  (expected ' + repr(expected) + ')'}"
    )


# ============================================================
# check_answer
# ============================================================

check_tests = [
    ("18", "18", True),
    ("18.0", "18", True),
    ("1200", "1,200", True),
    ("18", "19", False),
    (None, "18", False),
    ("abc", "18", False),
    ("1/2", "0.5", True),
]

for pred, gold, expected in check_tests:

    got = check_answer(pred, gold)

    status = "OK" if got == expected else "FAIL"

    if got != expected:
        all_ok = False

    print(
        f"[{status}] check_answer({pred!r}, {gold!r}) -> {got!r}"
        f"{'' if got == expected else '  (expected ' + repr(expected) + ')'}"
    )


format_tests = [
    ("work\n#### 42", True),
    ("#### -1,200.5", True),
    ("#### .0333", True),
    (r"\boxed{42}", False),
    ("#### ANSWER\n42", False),
    ("#### 42\nextra", False),
]

for response, expected in format_tests:
    got = is_format_compliant(response)
    if got != expected:
        all_ok = False
    status = "OK" if got == expected else "FAIL"
    print(f"[{status}] format_compliant({response!r}) -> {got!r}")


print()
print("ALL PASSED" if all_ok else "SOME FAILED")

if not all_ok:
    raise AssertionError("Parser tests failed")
