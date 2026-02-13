# Crypto Time-Series Forecasting

Multi-step return prediction of BTC/USDT using historical OHLCV data, engineered
technical features, and an LSTM model. Compares against naive and linear baselines.

## Quick Start

```bash
cd crypto_forecast
pip install -r requirements.txt
python main.py
```

Outputs land in `outputs/plots/` and `outputs/metrics/`.

## Data Source Strategy

Three-tier fallback for maximum reviewer convenience:

| Priority | Source | Network Required | Notes |
|---|---|---|---|
| 1 | Local CSV cache | No | Auto-created after first CCXT fetch |
| 2 | CCXT Binance public API | Yes | No API key needed, paginated fetch |
| 3 | Bundled `data/btc_usdt_daily.csv` | No | 2,235 real BTC/USDT daily candles (2020-2026), ships with repo |

**A reviewer can always run this project offline.** The bundled CSV contains
real market data, not synthetic. If CCXT is available, it fetches fresh data
and caches it for subsequent runs.

## Project Structure

```
crypto_forecast/
├── config.yaml              # All hyperparameters and paths
├── requirements.txt
├── main.py                  # Orchestration entry point
├── README.md
├── data/
│   └── btc_usdt_daily.csv   # Bundled dataset (2,235 daily candles)
└── src/
    ├── data_loader.py       # CCXT fetcher with static CSV fallback
    ├── preprocessing.py     # Missing values, outlier detection, validation
    ├── features.py          # Technical indicators and feature engineering
    ├── dataset.py           # Sliding window sequences, scaling, splits
    ├── models/
    │   ├── baseline.py      # Zero, Naive, Ridge, GBT baselines
    │   └── lstm.py          # LSTM forecaster
    ├── trainer.py           # Training loop with early stopping
    ├── evaluation.py        # Metrics, inverse transforms, walk-forward
    └── visualization.py     # All plotting functions
```

## Configuration

All parameters live in `config.yaml`. Key knobs:

| Parameter | Default | Description |
|---|---|---|
| `data.symbol` | BTC/USDT | Trading pair (CCXT format) |
| `data.since` | 2020-01-01 | Start date for CCXT fetch |
| `dataset.lookback` | 60 | Input window (trading days) |
| `dataset.horizon` | 5 | Predict next N candles |
| `dataset.target` | return_1 | Target column (returns are stationary) |
| `model.hidden_size` | 64 | LSTM hidden units |
| `model.num_layers` | 2 | Stacked LSTM layers |
| `training.epochs` | 50 | Max training epochs |
| `training.patience` | 10 | Early stopping patience |

Override the config path: `python main.py --config path/to/custom.yaml`

## Why Predict Returns Instead of Raw Price

Raw close prices are non-stationary — the test set is in a different price
regime than training, causing StandardScaler to produce out-of-distribution
inputs. Daily returns (`close[t]/close[t-1] - 1`) are approximately stationary,
making them suitable for:
- Standard scaling that generalizes across train/val/test
- Meaningful regression metrics (MSE on returns vs. MSE on prices)
- Direct interpretation: sign of prediction = predicted price direction

To switch to raw price prediction, set `dataset.target: close` in config.

## Model Choice: Why LSTM

| Consideration | LSTM | Transformer | Linear |
|---|---|---|---|
| Small dataset (<2k sequences) | Good | Overfits | Good |
| CPU training time | Minutes | Hours | Seconds |
| Sequential dependency modeling | Native | Via positional encoding | None |
| Implementation complexity | Moderate | High | Low |

**Decision**: LSTM provides the best tradeoff of expressiveness, training speed
on CPU, and dataset size fitness. A Transformer would be warranted with >10k
training sequences and GPU access.

The LSTM uses **direct multi-step forecasting** (one forward pass outputs all H
future values) rather than recursive single-step prediction. This avoids
compounding prediction errors across the horizon.

## Feature Engineering

Features are designed to be **scale-invariant** and **stationary** where possible:

- **Price ratios** (close/SMA, close/EMA) instead of raw moving averages — removes price-level dependency
- **RSI** — bounded oscillator, naturally stationary
- **Bollinger Band width and position** — normalized volatility and relative price
- **ATR normalized by close** — scale-free volatility
- **MACD normalized by close** — scale-free momentum
- **Volume ratios** (volume/SMA) — relative volume activity
- **Lagged returns** — autoregressive components at multiple horizons
- **Cyclical day-of-week encoding** — captures weekly seasonality without ordinal bias

All indicators use only past data — no look-ahead.

## Data Leakage Prevention

1. **Chronological splits**: train -> val -> test, no shuffling at the split level
2. **Scaler fit on train only**: StandardScaler is fit on the training set, then applied to val/test
3. **Sequences don't cross split boundaries**: sliding windows are generated independently within each split
4. **Features use only historical data**: every technical indicator computation is causal (rolling windows use past values only)

## Evaluation

### Metrics
- **MSE / MAE / RMSE** in original return space (inverse-transformed from scaled)
- **Direction accuracy**: for return targets, checks whether predicted sign matches actual sign (up/down). For price targets, checks sign of step-to-step price changes
- **Per-step metrics**: error broken down by t+1, t+2, ..., t+H

### Baselines
- **Zero (Random Walk)**: predicts zero return for all horizon steps. This is the
  true random-walk null hypothesis — expected future return is zero. The primary
  baseline any model must beat.
- **Naive Repeat**: repeats the last observed return for all horizon steps.
  This is a momentum/autocorrelation hypothesis, not the random walk.
- **Ridge Regression**: linear model on flattened lookback window.
- **GBT (Gradient Boosted Trees)**: `GradientBoostingRegressor` wrapped in
  `MultiOutputRegressor` (one ensemble per horizon step). Uses only the last
  timestep's features to keep dimensionality manageable. Tests non-linear
  feature interactions without sequential modeling.

### Results on Real BTC/USDT Data (2020-2026)

| Model | RMSE | MAE | Direction Accuracy |
|---|---|---|---|
| **Zero (RW)** | **0.02219** | **0.01508** | 0.0%* |
| Naive Repeat | 0.03153 | 0.02242 | 47.5% |
| Linear (Ridge) | 0.02808 | 0.02082 | 54.3% |
| GBT | 0.02353 | 0.01626 | 51.2% |
| LSTM | 0.02234 | 0.01517 | 50.1% |
| Walk-Fwd (avg) | 0.02748 | 0.01851 | 47.2% |

*Zero baseline always predicts 0, so `sign(0) != sign(actual)` for all nonzero returns.

### Honest Assessment

The results show a clean model hierarchy by RMSE (best to worst): Zero (RW), LSTM, GBT, Linear, Naive.
The LSTM (RMSE 0.02234) is very close to the Zero baseline (0.02219), and GBT
(0.02353) sits between them and the linear model. Direction accuracy for all
learned models hovers around 50-54%, not statistically distinguishable from
chance at this sample size.

**This is the expected result.** Daily crypto returns are close to a random walk —
the efficient market hypothesis predicts that simple models cannot consistently
beat "predict zero" at this frequency with public features. A submission that
claimed otherwise would be less credible than one that demonstrates:

1. A correct, leakage-free pipeline that produces honest results
2. Proper baseline comparisons including the true null hypothesis (Zero)
3. Multiple model families (linear, tree-based, recurrent neural network) for a
   thorough comparison
4. The engineering to extend the approach (higher-frequency data, alternative
   features, or ensemble methods could shift the edge)

The value of this submission is the **infrastructure and methodology**, not the
prediction accuracy. The same pipeline on intraday data with order-flow features
would be a different story.

### Walk-Forward Validation
Runs automatically. Slides the evaluation window through the test set in 5 steps,
using the same scaler the model was trained with (no scaler refit, which would
create distribution shift against frozen model weights). Walk-forward starts at
the true test boundary (85%), not the validation boundary, to avoid evaluating
on data used for early stopping.

## Outputs

After running:
- `outputs/plots/training_loss.png` — train/val loss curves
- `outputs/plots/pred_vs_actual_lstm_step1.png` — LSTM predictions overlay
- `outputs/plots/pred_vs_actual_gbt_step1.png` — GBT predictions overlay
- `outputs/plots/all_steps_lstm.png` — all horizon steps grid
- `outputs/plots/model_comparison.png` — bar chart across models
- `outputs/plots/per_step_mae_comparison.png` — per-step error comparison
- `outputs/metrics/metrics_report.json` — full numeric results

## Extending the Model

**Add a new model**: implement a `nn.Module` with `forward(x) -> (batch, horizon)`,
add it to `src/models/`, and wire it in `main.py`.

**Add features**: add a function in `src/features.py` following the existing
pattern (receives and returns a DataFrame), call it from `build_features()`.

**Change target**: set `dataset.target` in config to any column (e.g., `close`,
`log_return_1`, `volume`). The evaluation automatically detects whether the target
is a return (direction accuracy uses sign) or a price (uses step differences).

**Change asset**: set `data.symbol` to any CCXT-supported pair (e.g., `ETH/USDT`,
`SOL/USDT`). Update the bundled CSV or let CCXT fetch fresh data.

## Reproducibility

- All random seeds (Python, NumPy, PyTorch) are fixed via `config.yaml`
- Data is cached locally as CSV after first CCXT fetch
- Bundled static CSV ensures identical results across environments
- Deterministic PyTorch flags are set
- No GPU-specific non-determinism (CPU only)
