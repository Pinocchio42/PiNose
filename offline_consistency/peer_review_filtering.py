import json
import argparse
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple



def count_votes(votes: List[str]) -> Tuple[int, int, int]:
    """Count the number of each type of vote."""
    support = votes.count("Consistent")
    oppose = votes.count("Non-Consistent")
    neutral = votes.count("Neutral")
    return support, oppose, neutral


def check_vote_validity_and_type(votes: List[str], threshold: int) -> Tuple[bool, str]:
    """Check if a peer response is valid based on the votes and threshold, and return the type of the most voted option."""
    support, oppose, neutral = count_votes(votes)
    max_votes = max(support, oppose, neutral)
    if max_votes >= threshold:
        if support == max_votes:
            return True, "Consistent"
        elif oppose == max_votes:
            return True, "Non-Consistent"
        else:
            return True, "Neutral"
    else:
        return False, None


def categorize_peer_responses(
    peer_review_results: List[Dict],
    review_number_threshold: int,
    peer_response_number_threshold: int,
) -> Optional[bool]:
    """Categorize peer responses based on the votes and thresholds."""
    peer_labels = []
    for peer_response in peer_review_results:
        review_rounds = peer_response["review_rounds"]
        valid, label = check_vote_validity_and_type(
            review_rounds, review_number_threshold
        )
        if valid:
            peer_labels.append(label)
    _, label = check_vote_validity_and_type(peer_labels, peer_response_number_threshold)
    label_to_bool = {"Consistent": True, "Non-Consistent": False, "Neutral": None}
    return label_to_bool.get(label, None)


def load_jsonline(file_path):
    data = []
    with open(file_path, "r") as file:
        for line in file:
            data.append(json.loads(line))
    return data


def main(
    input_file, output_file, review_number_threshold, peer_response_number_threshold
):
    dataset = load_jsonline(input_file)

    with open(output_file, "w") as f:
        for item in tqdm(dataset):
            label = categorize_peer_responses(
                item["peer_review_results"],
                review_number_threshold,
                peer_response_number_threshold,
            )
            if label is not None:
                new_item = dict(
                    question=item["question"], response=item["response"], label=label
                )
                f.write(json.dumps(new_item) + "\n")
                f.flush()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process peer review results and categorize them based on thresholds."
    )
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="The path to the input JSONL file containing peer review results.",
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="The path to the output JSONL file where the processed data will be written.",
    )
    parser.add_argument(
        "--peer_response_number_threshold",
        type=int,
        required=True,
        help="The minimum number of consistent responses required.",
    )
    parser.add_argument(
        "--review_number_threshold",
        type=int,
        required=True,
        help="The minimum number of reviews a peer must have to be considered.",
    )

    args = parser.parse_args()

    main(
        args.input_file,
        args.output_file,
        args.review_number_threshold,
        args.peer_response_number_threshold,
    )
