"""
LSTM forecasting model.

Architecture rationale:
- LSTM over vanilla RNN: LSTMs handle the vanishing gradient problem through
  gating, which is critical for financial data where dependencies can span
  weeks (e.g., 60-day lookback).
- LSTM over Transformer: For small-to-medium datasets (< 10k sequences),
  LSTMs are less prone to overfitting and train faster on CPU. Transformers
  shine with large datasets and GPU. For a CPU-friendly screening task, LSTM
  is the pragmatic choice.
- LSTM over GRU: Marginal difference in practice. LSTM chosen for familiarity
  with reviewers; GRU would be equally valid.
- Direct multi-step output: A single forward pass predicts all H future steps
  via a linear head. Avoids recursive error accumulation.
- Dropout between LSTM layers: Regularization to combat overfitting on
  financial noise.
"""

import torch
import torch.nn as nn


class LSTMForecaster(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
        horizon: int = 5,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
            bidirectional=bidirectional,
        )

        self.dropout = nn.Dropout(dropout)

        # Project final hidden state to horizon-length output
        self.fc = nn.Linear(hidden_size * self.num_directions, horizon)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, lookback, input_size)
        returns: (batch, horizon)
        """
        # lstm_out: (batch, lookback, hidden * directions)
        lstm_out, _ = self.lstm(x)

        # Use the output at the last time step
        last_out = lstm_out[:, -1, :]  # (batch, hidden * directions)
        last_out = self.dropout(last_out)

        return self.fc(last_out)  # (batch, horizon)
