import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class TransformerCNN(nn.Module):
    def __init__(self, input_dim, num_classes=2):
        super(TransformerCNN, self).__init__()
        # CNN part
        self.conv = nn.Conv1d(1, 16, kernel_size=3, padding=1)

        # Transformer part
        encoder_layer = nn.TransformerEncoderLayer(d_model=16, nhead=2, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=2)

        self.fc = nn.Linear(16 * (input_dim // 1), num_classes) # simplification

    def forward(self, x):
        x = x.unsqueeze(1) # [batch, 1, input_dim]
        x = F.relu(self.conv(x)) # [batch, 16, input_dim]
        x = x.permute(0, 2, 1) # [batch, input_dim, 16] for transformer
        x = self.transformer(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class AudioTransformerCNN:
    def __init__(self, input_dim, num_classes=2):
        self.model = TransformerCNN(input_dim, num_classes)
        self.input_dim = input_dim

    def train(self, X, y):
        X_tensor = torch.FloatTensor(X)
        y_tensor = torch.LongTensor(y)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        self.model.train()
        for epoch in range(10):
            optimizer.zero_grad()
            outputs = self.model(X_tensor)
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()

    def predict(self, X):
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            outputs = self.model(X_tensor)
            return torch.argmax(outputs, dim=1).numpy()
