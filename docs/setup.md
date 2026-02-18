# Setup Guide

## Prerequisites
- Python 3.9+
- pip

## Installation

```bash
git clone https://github.com/norfrt6-lab/crypto_forecast.git
cd crypto_forecast
pip install -r requirements.txt
```

## Configuration

Edit `config.yaml` to adjust model parameters:
- `epochs`: Training epochs (default: 50)
- `batch_size`: Batch size (default: 32)
- `sequence_length`: Lookback window (default: 60)

## Usage

```bash
python main.py
```
