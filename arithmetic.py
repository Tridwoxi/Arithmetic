"""How good are LLMs at integer addition and multiplication without tool-calling? How
does this change with model and reasoning effort?"""

from pathlib import Path
from random import randint, seed

from openai import OpenAI
from openai.types import ReasoningEffort, ResponsesModel

## Configuration

MODELS: list[tuple[ResponsesModel, ReasoningEffort]] = [
    ("gpt-4o-mini-2024-07-18", None),
    ("gpt-5-nano-2025-08-07", "minimal"),
    ("gpt-5-nano-2025-08-07", "high"),
    ("gpt-6-luna", "none"),
    ("gpt-6-luna", "medium"),
    ("gpt-6-luna", "max"),
]

DIGITS_PROGRESSION = [1, 2, 4, 8, 16, 32, 64]
TRIALS = 10

ROOT = Path(__file__).parent

## Setup

client = OpenAI()
seed("arithmetic")


def generate_n_digit_number(num_digits: int) -> int:
    if num_digits == 1:
        return randint(0, 9)
    return randint(10 ** (num_digits - 1), 10**num_digits - 1)


def is_correct(response: str, ground_truth: int) -> bool:
    cleaned = "".join(filter(lambda char: char in "0123456789 ", response))
    numbers = cleaned.split(" ")
    return str(ground_truth) in numbers


## Experiment

for operator in ["plus", "times"]:
    for num_digits in DIGITS_PROGRESSION:
        for trial in range(TRIALS):

            a = generate_n_digit_number(num_digits)
            b = generate_n_digit_number(num_digits)
            prompt = f"calculate {a} {operator} {b}"
            ground_truth = a + b if operator == "plus" else a * b

            for model, effort in MODELS:
                response = client.responses.create(
                    model=model,
                    input=prompt,
                    reasoning={"effort": effort},
                )
                output_text = response.output_text
                tokens = -1 if response.usage is None else response.usage.output_tokens
                verdict = is_correct(output_text, ground_truth)
                report = f"""MODEL: {model}
EFFORT: {effort}
PROMPT: {prompt}
OUTPUT_TEXT: {output_text}
VERDICT: {verdict}
TOKENS: {tokens}
{"=" * 80}
"""
                path = (ROOT / model).with_suffix(".txt")
                with path.open("a") as file:
                    file.write(report)
                print(report)
