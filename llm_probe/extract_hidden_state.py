import argparse
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# To support pandas' progress_apply
tqdm.pandas()

torch.set_grad_enabled(False)

MODEL_PATH = "meta-llama/Llama-2-7b-hf"

llm = AutoModel.from_pretrained(MODEL_PATH, torch_dtype=torch.half).cuda().eval()

tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

def get_hidden_state(prompt, layer):
    inputs = torch.tensor(tokenizer.encode(prompt)).unsqueeze(0).cuda()
    outputs = llm(inputs, output_hidden_states=True)
    return outputs["hidden_states"][layer][0].cpu()

prompt_template = "\n### Instruction\nCompose a concise answer within a single sentence.\n\n### Question\n{question}\n\n### Answer\n{answer}"

def process_and_extract_features_from_jsonlines(input_path):
    dataframe = pd.read_json(input_path, lines=True)
    
    dataframe["prompt"] = dataframe.progress_apply(
        lambda row: prompt_template.format(
            question=row["question"], answer=row["response"]
        ),
        axis=1,
    )
    
    dataframe["feature"] = dataframe["prompt"].progress_apply(
        lambda x: get_hidden_state(x, 16)[-1].numpy().astype("float")
    )
    
    return dataframe


def main():
    parser = argparse.ArgumentParser(description='Process JSON lines and extract features.')
    parser.add_argument('--input_file', type=str, help='Input JSON lines file path')
    parser.add_argument('--output_file', type=str, help='Output Parquet file path')
    
    args = parser.parse_args()
    
    dataframe = process_and_extract_features_from_jsonlines(args.input_file)
    
    dataframe.to_parquet(args.output_file)

if __name__ == "__main__":
    main()