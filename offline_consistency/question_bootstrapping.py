from vllm import LLM, SamplingParams
import json
import sys
import pandas as pd
import random
import argparse
from tqdm import tqdm

# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=64, n=10, stop=["\n8", "###"])


# Create an LLM.
MODEL_PATH = "meta-llama/Llama-2-7b-hf"
llm = LLM(model=MODEL_PATH, dtype="half")


def ask_llm(prompt):
    outputs = llm.generate([prompt], sampling_params)[0]
    return [output.__dict__ for output in outputs.outputs]


def generate_prompt(questions):
    assert len(questions) == 5
    prefix = "Please ask some objective questions of similar difficulty to [Seed Questions].\n\n### [Seed Questions]\n"

    text = ""
    for i, question in enumerate(questions, 1):
        text += f"{i}. {question}\n"

    return prefix + text + "6."


def main(N, seed_question_file):
    df = pd.read_json(seed_question_file, lines=True)
    all_questions = (
        df["question"].apply(lambda x: x + "?" if x[-1] != "?" else x).tolist()
    )

    output_file_name = f"bootstrap_question_total_{N}.jsonl"

    with open(output_file_name, "w") as f:
        for _ in tqdm(range(N)):
            selected_questions = random.sample(all_questions, 5)
            prompt = generate_prompt(selected_questions)
            raw_responses = ask_llm(prompt)
            output_item = dict(prompt=prompt, raw_responses=raw_responses)
            f.write(json.dumps(output_item) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Bootstrap Questions")
    parser.add_argument("--N", type=int, default=100000, help="Number of iterations")
    parser.add_argument(
        "--seed_question_file",
        type=str,
        help="Path to a JSON Lines formatted file containing a seed set of questions for Bootstrapping.",
    )
    args = parser.parse_args()

    main(args.N)
