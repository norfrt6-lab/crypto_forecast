# Architecture

## Overview

```
main.py          Entry point
config.yaml      Model and data configuration
src/
  data_loader.py Data fetching from Binance API
  dataset.py     PyTorch Dataset wrapper
  features.py    Technical indicator computation
  preprocessing.py  Data scaling and splitting
  trainer.py     Training loop
  evaluation.py  Metrics and evaluation
  visualization.py  Plotting predictions
  models/
    lstm.py      LSTM model definition
    baseline.py  Baseline models
```

## Data Flow

1. `data_loader` fetches OHLCV data
2. `features` computes technical indicators
3. `preprocessing` scales and splits data
4. `dataset` wraps data for PyTorch
5. `trainer` runs the training loop
6. `evaluation` computes metrics
7. `visualization` plots results
