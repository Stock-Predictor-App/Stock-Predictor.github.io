import torch
import torch.nn as nn

class StockPredictionNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(StockPredictionNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Using an LSTM since we are dealing with sequential/time-series differentials
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))
        
        # Decode the hidden state of the last time step
        out = self.fc(out[:, -1, :])
        return out
