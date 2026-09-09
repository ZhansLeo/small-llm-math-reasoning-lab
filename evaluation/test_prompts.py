from prompts import PROMPT_TEMPLATES, build_messages, build_prompt


def test_prompt_contract():
    for name in ("prompt_a", "prompt_b", "prompt_c"):
        text = build_prompt("What is 1 + 1?", name)
        assert "What is 1 + 1?" in text
        assert "#### 42" in text
        assert "#### ANSWER" not in text
        assert build_messages("Q", name)[0]["role"] == "user"

    baseline = PROMPT_TEMPLATES["prompt_a"].lower()
    assert "step by step" not in baseline
    assert "known quantities" not in baseline
    assert "step by step" in PROMPT_TEMPLATES["prompt_b"].lower()
    assert "identify known quantities" in PROMPT_TEMPLATES["prompt_c"].lower()


if __name__ == "__main__":
    test_prompt_contract()
    print("ALL PASSED")
