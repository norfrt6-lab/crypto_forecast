# Models

## LSTM

Long Short-Term Memory network for time series forecasting.

- Input: Sequence of features over a lookback window
- Hidden layers: Configurable via `config.yaml`
- Output: Next-day price prediction

## Baseline

Simple baseline models for comparison:
- **Last Value**: Predicts previous day's close
- **Moving Average**: Predicts N-day moving average
