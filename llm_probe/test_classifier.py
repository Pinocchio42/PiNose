import torch
import argparse
import pandas as pd
import numpy as np
from classifier_modeling import SimpleMLP
from sklearn.metrics import roc_auc_score

def load_data(file_path):
    df = pd.read_parquet(file_path)
    features = np.vstack(df['feature'].values)  
    labels = df['label'].values
    return features, labels

def load_model(ckpt_path):
    input_size = 4096
    hidden_size = 256
    num_classes = 2
    model = SimpleMLP(input_size, hidden_size, num_classes)
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict)
    return model.eval()

def get_auc(model, features, labels):
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        probs = model(features).softmax(dim=-1)
    return roc_auc_score(labels, probs[:,1])

def get_acc(model, features, labels, threshold):
    features = torch.tensor(features, dtype=torch.float32)
    with torch.no_grad():
        probs = model(features).softmax(dim=-1)
    preds = (probs[:,1]>threshold).numpy()
    return (preds==labels).mean()


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test data.')
    parser.add_argument('model_path', type=str, help='Path to the model checkpoint.')
    parser.add_argument('test_data_path', type=str, help='Path to the test data.')
    parser.add_argument('threshold', type=float, help='Threshold for classification.')
    args = parser.parse_args()

    model = load_model(args.model_path)
    features, labels = load_data(args.test_data_path)

    auc = get_auc(model, features, labels)
    acc = get_acc(model, features, labels, args.threshold)

    print(f'AUC: {auc}')
    print(f'Accuracy: {acc}')

if __name__ == "__main__":
    main()
