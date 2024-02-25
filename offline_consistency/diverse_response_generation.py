import argparse
from vllm import LLM, SamplingParams
import json
from tqdm import tqdm
import sys
import random


prompt_templates = [
    "### Instruction\nCompose a concise answer within a single sentence.\n\n### Question\n{question}\n\n### Answer\n",
    "### Instruction\nGenerate a brief response in just one sentence.\n\n### Question\n{question}\n\n### Answer\n",
    "### Instruction\nGive a helpful answer.\n\n### Question\n{question}\n\n### Answer\n",
]


# Create a sampling params object.
sampling_params = SamplingParams(max_tokens=64, n=10, stop=["\n\n", "Question"])

# Create an LLM.
MODEL_PATH = "meta-llama/Llama-2-7b-hf"
llm = LLM(model=MODEL_PATH, dtype="half")


def ask_llm(prompt):
    outputs = llm.generate([prompt], sampling_params)[0]
    return [output.__dict__ for output in outputs.outputs]


def load_jsonline(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main(input_file, output_file):
    dataset = load_jsonline(input_file)

    with open(output_file, "w") as f:
        for item in tqdm(dataset):
            prompt_template = random.choice(prompt_templates)
            prompt = prompt_template.format(question=item["question"])
            item["raw_responses"] = ask_llm(prompt)
            f.write(json.dumps(item) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate diverse responses using LLM."
    )

    parser.add_argument(
        "--input_file", type=str, help="Input file containing the set of questions"
    )
    parser.add_argument(
        "--output_file", type=str, help="Output file path for storing model responses"
    )
    args = parser.parse_args()

    main(args.input_file, args.output_file)
