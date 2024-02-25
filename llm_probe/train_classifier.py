import torch
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from classifier_modeling import SimpleMLP

def load_data(file_path):
    df = pd.read_parquet(file_path)
    features = np.vstack(df['feature'].values)  
    labels = df['label'].values
    return features, labels

def train(model, dataloader, criterion, optimizer, num_epochs, device):
    for epoch in range(num_epochs):
        for i, (features, labels) in enumerate(dataloader):
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if (i+1) % 10 == 0:
                print(f'Epoch [{epoch+1}/{num_epochs}], Step [{i+1}/{len(dataloader)}], Loss: {loss.item()}')

def main():
    
    input_size = 4096
    hidden_size = 256
    num_classes = 2
    learning_rate = 0.001
    num_epochs = 5
    batch_size = 64

    if torch.cuda.is_available():
        device = torch.device("cuda")
        print("Using CUDA")
    else:
        device = torch.device("cpu")
        print("Using CPU")
    
    features, labels = load_data('Train.parquet')
    features_val, labels_val = load_data('Validation.parquet')

    features = torch.tensor(features, dtype=torch.float32)
    labels = torch.tensor(labels, dtype=torch.long)
    features_val = torch.tensor(features_val, dtype=torch.float32)
    labels_val = torch.tensor(labels_val, dtype=torch.long)

    
    train_dataset = TensorDataset(features, labels)
    val_dataset = TensorDataset(features_val, labels_val)
    train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(dataset=val_dataset, batch_size=batch_size, shuffle=False)

    
    model = SimpleMLP(input_size, hidden_size, num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train(model, train_loader, criterion, optimizer, num_epochs, device)

    torch.save(model.cpu().state_dict(), 'model.pth')
    

if __name__ == "__main__":
    main()