import torch
import torch.nn as nn
from ensemble_model.lstm_model import CrowdLSTM

def train_lstm(X, y, epochs=10):
    model = CrowdLSTM(X.shape[2])
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(epochs):
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

    torch.save(model.state_dict(), "models/lstm_model.pth")
    return model
