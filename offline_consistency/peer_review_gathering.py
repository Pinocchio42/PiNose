import json
from tqdm import tqdm
import sys
import random
from vllm import LLM, SamplingParams
import argparse
import random


prompt_template = """Assess the connection between the two responses to the initial query, taking into account the potential scenarios of Endorsement, Contradiction, and Impartiality.
{seeds}

### Input

- **Question:** {question}
- **First Response:** {answer}
- **Second Response:** {pred_answer}

### Output
Judgement:"""


with open("prompts/nli_seeds.json") as f:
    nli_all_seeds = json.load(f)

# Create a sampling params object.
sampling_params = SamplingParams(
    max_tokens=2,
    stop=["\n\n", "###"],
)

MODEL_PATH = "meta-llama/Llama-2-7b-hf"
llm = LLM(model=MODEL_PATH, dtype="half")


def ask_llm(prompt):
    outputs = llm.generate([prompt], sampling_params)[0]
    return [output.__dict__ for output in outputs.outputs]


def generate_prompt_with_seeds(question, answer, predicted_answer):
    selected_seeds = random.choices(nli_all_seeds, k=4)
    prompt_text = prompt_template.format(
        seeds="\n\n".join(selected_seeds),
        question=question,
        answer=answer,
        predicted_answer=predicted_answer,
    )
    return prompt_text


def execute_single_review_round(question, reference_response, peer_response):
    prompt_text = generate_prompt_with_seeds(
        question, peer_response, reference_response
    )
    judgement_token = ask_llm(prompt_text)[0]["token_ids"][0]
    # Endorsement, Contradiction, and Impartiality
    # First token: 2796, 1281, 14305
    # Consistent, Non-Consistent, and Neutral
    judgement_token2label = {
        2796: "Consistent",
        1281: "Non-Consistent",
        14305: "Neutral",
    }
    return judgement_token2label.get(judgement_token, "Neutral")


def conduct_peer_reviews(
    question, reference_response, all_peer_responses, num_review_rounds
):
    peer_review_results = []
    for peer_response in all_peer_responses:
        peer_review_rounds = []
        for _ in range(num_review_rounds):
            peer_review_rounds.append(
                execute_single_review_round(question, reference_response, peer_response)
            )
        peer_review_results.append(
            dict(peer_response=peer_response, review_rounds=peer_review_rounds)
        )
    return peer_review_results


def load_jsonline(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main(input_file, output_file, num_peer_responses, num_review_rounds):
    dataset = load_jsonline(input_file)

    with open(output_file, "w") as f:
        for item in tqdm(dataset):
            question = item["question"]
            raw_responses = [x["text"] for x in item["raw_responses"]]
            random.shuffle(raw_responses)
            reference_response = raw_responses[0]
            all_peer_responses = raw_responses[1 : 1 + num_peer_responses]
            peer_review_results = conduct_peer_reviews(
                question, reference_response, all_peer_responses, num_review_rounds
            )
            new_item = dict(
                question=question,
                response=reference_response,
                peer_review_results=peer_review_results,
            )
            f.write(json.dumps(new_item) + "\n")
            f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate diverse responses using LLM."
    )

    parser.add_argument(
        "--input_file",
        type=str,
        help="Input file containing the set of question and responses",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        help="Output file path for storing model peer_reviews",
    )
    parser.add_argument("--num_review_rounds", type=int, help="number of review rounds")
    parser.add_argument(
        "--num_peer_responses", type=int, help="number of peer responses"
    )
    args = parser.parse_args()

    main(
        args.input_file,
        args.output_file,
        args.num_peer_responses,
        args.num_review_rounds,
    )
