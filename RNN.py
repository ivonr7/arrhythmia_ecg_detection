import torch
import torch.nn as nn

#hyperparameters
# input_size: 12 features
# hidden_size: how much the model remembers and each moment
# num_classes = 2 - binary classification
# used in training (not needed here): batch_size, num_epochs, learning_rate 

#inheriting from nn.Module
class ECG_RNN(nn.Module):
    def __init__(self, input_size=12, hidden_size=128, num_layers=2, num_classes=2, dropout=0.2):
        super().__init__()

        # LSTM: input shape = (batch, seq_len, input_size)
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
        )#586 time steps of 12 dimensional data -> processed one row at a time

        #batch_first: x.shape == (batch_size, seq_len, input_size)
        #dropouts between layers -> prevent overfitting

        # Final classifier
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        # x shape: (batch_size = 32, seq_len= 586 (window size: 293x2), channels = 12)
        out, (h_n, c_n) = self.lstm(x)

        # Use the final hidden state from last LSTM layer
        last_hidden = h_n[-1]     # shape: (batch, hidden_size)

        logits = self.fc(last_hidden)  # shape: (batch_size = 32, num_classes)
        return logits

#many to one RNN (586 time steps each with 12 values -> output one class label)
model = ECG_RNN()
print(model)