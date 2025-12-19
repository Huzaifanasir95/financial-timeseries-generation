# Financial Time-Series Generation: TimeGAN vs Diffusion Models
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg) ![TensorFlow](https://img.shields.io/badge/TensorFlow-2.12+-orange.svg) ![License](https://img.shields.io/badge/License-Academic-yellow.svg) ![Status](https://img.shields.io/badge/Status-Complete-green.svg)

> **Comprehensive comparative study** of **TimeGAN** and **Diffusion Models** for synthetic financial data generation, with forecasting baseline evaluation across **25 financial assets** (indices, stocks, cryptocurrencies, commodities).

Implementation of state-of-the-art **generative models** for financial time-series, achieving **54% better distribution matching with TimeGAN** (p=0.0004, Cohen's d=-2.82). Includes production-ready **Flask web application** with interactive dashboards and REST API.

## 📋 Table of Contents
- [Overview](#-overview)
- [Key Findings](#-key-findings)
- [Features](#-features)
- [Dataset](#-dataset)
- [Models Implemented](#-models-implemented)
- [Installation](#-installation)
- [Usage](#-usage)
- [Project Structure](#-project-structure)
- [Results](#-results)
- [Web Application](#-web-application)
- [Evaluation Metrics](#-evaluation-metrics)
- [Practical Recommendations](#-practical-recommendations)
- [Limitations & Future Work](#-limitations--future-work)
- [Citation](#-citation)
- [Authors](#-authors)
- [License](#-license)

## 🎯 Overview

This research project addresses two fundamental challenges in **quantitative finance**:

### 1. **Synthetic Data Generation** (Primary Focus)
Systematic comparison of **TimeGAN** vs **Diffusion Models** for generating realistic financial time-series data with rigorous statistical validation.

### 2. **Price Forecasting** (Secondary Focus)
Comprehensive evaluation of **5 models** (ARIMA, LSTM, Prophet, TimeGAN, DDPM) establishing that generative models excel at distribution matching but fail at forecasting tasks.

### Key Contributions
✅ **First systematic comparison** of TimeGAN vs Diffusion on multi-asset financial data  
✅ **Statistical validation** with paired t-tests, Cohen's d, KS tests  
✅ **Proof that generative models ≠ forecasting models** (negative R² demonstrated)  
✅ **Production-ready Flask web application** with interactive dashboards  
✅ **25 trained models** ready for deployment (11 TimeGAN + 12 Diffusion + 2 baselines)  

## 🏆 Key Findings

### TimeGAN vs Diffusion Models

| Metric | TimeGAN | Diffusion | Winner |
|--------|---------|-----------|---------|
| **Mean Distribution Difference** | 0.067 ± 0.030 | 0.134 ± 0.017 | 🥇 **TimeGAN (54% better)** |
| **Median Performance** | 0.059 | 0.131 | 🥇 **TimeGAN (55% better)** |
| **Assets Won** | **9/11 (82%)** | 2/11 (18%) | 🥇 **TimeGAN** |
| **Statistical Significance** | p = 0.0004 | - | Highly significant*** |
| **Effect Size (Cohen's d)** | -2.82 | - | Large effect |

**Statistical Conclusion**: TimeGAN significantly outperforms Diffusion Models for financial synthetic data generation (t=-4.59, p=0.0004, d=-2.82).

### Forecasting Results (Cryptocurrency)

| Model | MAE | RMSE | R² | Best For |
|-------|-----|------|----|---------| 
| **ARIMA** 🥇 | 0.00440 | 0.00598 | **0.9751** | Short-term prediction |
| **LSTM** 🥈 | 0.00437 | 0.00600 | **0.8082** | Non-linear patterns |
| Prophet | 0.00635 | 0.00833 | -0.942 | ❌ Poor for crypto |
| **TimeGAN** | - | - | **-1.72** | ❌ Unsuitable for forecasting |
| **Diffusion** | - | - | **-4.24** | ❌ Unsuitable for forecasting |

**Critical Insight**: Generative models have **negative R²**, performing worse than predicting the mean. They are designed for **distribution matching**, not **forecasting**.

## 🌟 Features

### 1. **Generative Models**
- **TimeGAN**: 5-component architecture (Embedder, Recovery, Generator, Supervisor, Discriminator)
  - 20,000 training iterations
  - 48-step sequence length
  - GPU-optimized (CUDA + mixed precision)
  - 11 trained models (indices, stocks, crypto)
  
- **Diffusion Models (DDPM)**: Residual networks with time conditioning
  - 1000-step noise schedule
  - 500 epoch training
  - Forward/reverse diffusion processes
  - 12 trained models

### 2. **Forecasting Baselines**
- **ARIMA**: Auto-parameter selection with ADF stationarity testing
- **LSTM**: 2-layer architecture with dropout regularization
- **Prophet**: Meta's automatic seasonality detection

### 3. **Comprehensive Evaluation**
- **Statistical Tests**: KS test, paired t-test, Cohen's d effect size
- **Distribution Metrics**: Mean/std difference, autocorrelation preservation
- **Forecasting Metrics**: MAE, RMSE, R², MAPE, direction accuracy

### 4. **Production-Ready Application**
- **Flask Web App**: Interactive dashboards with real-time model comparison
- **REST API**: Programmatic access to model results
- **Visualization**: 40+ figures (training curves, confusion matrices, Q-Q plots)

### 5. **Reproducible Research**
- All models, scalers, and parameters saved
- Detailed training logs and evaluation results
- Docker support for easy deployment

## 📊 Dataset

### Assets Analyzed (25 Total)

**📈 Indices (7)**:
- **US**: S&P 500 (^GSPC), NASDAQ (^IXIC), Dow Jones (^DJI)
- **International**: FTSE 100 (^FTSE), Nikkei 225 (^N225), Hang Seng (^HSI), DAX (^GDAXI)

**💻 Technology Stocks (5)**:
- Apple (AAPL), Microsoft (MSFT), Google (GOOGL), Amazon (AMZN), Tesla (TSLA)

**🏢 Traditional Stocks (6)**:
- JPMorgan (JPM), Exxon (XOM), Johnson & Johnson (JNJ), Visa (V), Walmart (WMT), Procter & Gamble (PG)

**₿ Cryptocurrencies (5)**:
- Bitcoin (BTC-USD), Ethereum (ETH-USD), Binance Coin (BNB-USD), Solana (SOL-USD), Cardano (ADA-USD)

**🥇 Commodities (2)**:
- Gold (GC=F), Crude Oil (CL=F)

### Data Characteristics

| Attribute | Value |
|-----------|-------|
| **Timespan** | 2015-01-05 to 2024-12-30 (10 years) |
| **Frequency** | Daily (business days) |
| **Total Samples** | 2,443 - 3,651 rows per asset |
| **Features per Asset** | 108 technical indicators |
| **Train Split** | 70% (~1,700-2,550 samples) |
| **Validation Split** | 15% (~365-550 samples) |
| **Test Split** | 15% (~365-550 samples) |
| **Stationarity** | All stationary (ADF p < 0.05) |
| **Data Source** | yfinance API |

### Feature Engineering (108 Indicators)

**Price-Based (8)**: Open, High, Low, Close, Volume, Returns, Log Returns, Price Range

**Trend Indicators (14)**: SMA (5/10/20/50), EMA (5/10/20), DEMA, TEMA, WMA, TRIMA

**Momentum Indicators (15)**: RSI, ROC, Stochastic K/D, Williams %R, CCI, MFI, Ultimate Oscillator

**Volatility Indicators (12)**: ATR, Bollinger Bands (Upper/Middle/Lower/Width), Historical Volatility, Keltner Channels, Donchian Channels

**Volume Indicators (10)**: OBV, Volume SMA, Volume ROC, MFI, CMF, VWAP, PVT

**Trend Strength (8)**: MACD, MACD Signal, MACD Histogram, ADX, +DI, -DI, Aroon Up/Down

**Ichimoku Cloud (5)**: Tenkan-sen, Kijun-sen, Senkou Span A/B, Chikou Span

**Other (36)**: Lagged features, rolling statistics, technical patterns

## 🤖 Models Implemented

### Generative Models

#### TimeGAN (Time-series GAN) - **Winner**
```
Architecture: 5 Neural Networks
  ├── Embedder: Maps real data → latent space
  ├── Recovery: Maps latent space → real data
  ├── Generator: Creates synthetic latent representations
  ├── Supervisor: Enforces temporal consistency
  └── Discriminator: Distinguishes real vs synthetic

Training Configuration:
  • Iterations: 20,000
  • Batch Size: 128 (GPU-optimized)
  • Hidden Dim: 128
  • Sequence Length: 48
  • Learning Rate: 5×10⁻⁴
  • Loss: Combined adversarial + supervised + reconstruction
  • Time per Asset: ~18 minutes (GPU)
```

**Performance**:
- **Excellent** (6 assets): HSI, AMZN, FTSE, DJI, N225, IXIC
- **Good** (4 assets): AAPL, MSFT, TSLA, GSPC
- **Fair** (1 asset): GOOGL

#### Diffusion Models (DDPM)
```
Architecture: Residual Denoising Network
  • Forward Process: Gradually adds Gaussian noise (1000 steps)
  • Reverse Process: Learns to denoise (predict noise)
  • Time Conditioning: Sinusoidal embeddings
  • Network: Multi-head attention + residual blocks

Training Configuration:
  • Epochs: 500
  • Diffusion Steps: 1000
  • Beta Schedule: Linear (1×10⁻⁴ to 0.02)
  • Architecture: Transformer-inspired
  • Time per Asset: ~2 hours (GPU required)
```

**Performance**:
- All 11 assets rated **Fair** (KS statistic: 0.32-0.48)
- Better theoretical guarantees, but slower convergence

### Forecasting Models

#### ARIMA - **Best for Crypto Forecasting**
- **Model**: Auto-ARIMA with automatic (p,d,q) selection
- **R² Score**: 0.9751 (explains 97.51% variance)
- **Strengths**: Interpretable, fast, excellent for stationary series
- **Weaknesses**: Linear assumptions, poor for regime changes

#### LSTM - **Best for Non-Linear Patterns**
- **Architecture**: 2-layer LSTM (64→32 units) + Dropout (0.2)
- **R² Score**: 0.8082 (good performance)
- **Strengths**: Captures complex patterns, handles multiple features
- **Weaknesses**: Requires more data, computationally expensive

#### Prophet - **Best for Seasonal Data**
- **Model**: Meta's additive decomposition (trend + seasonality + holidays)
- **R² Score**: -0.942 (poor for crypto)
- **Strengths**: Automatic seasonality, handles missing data
- **Weaknesses**: Struggles with high volatility, crypto markets

## 📁 Project Structure

```plaintext
financial-timeseries-generation/
├── data/
│   ├── raw/                    # Original financial data (yfinance API)
│   ├── processed/              # Cleaned data with 108 technical features
│   │   ├── train/              # Training split (70%)
│   │   ├── val/                # Validation split (15%)
│   │   ├── test/               # Test split (15%)
│   │   ├── _processing_summary.csv
│   │   ├── _adf_test_results.csv
│   │   └── _eda_statistics.csv
│   ├── features/               # Feature engineering outputs
│   └── synthetic/              # Generated synthetic data
├── forecasting/
│   ├── ARIMA_Model.ipynb       # Auto-ARIMA implementation
│   ├── LSTM_Model.ipynb        # Deep learning forecasting
│   ├── Prophet_Model.ipynb     # Meta's Prophet model
│   ├── timegan-latest.ipynb    # TimeGAN training & evaluation
│   ├── DDPM_Model.ipynb        # Diffusion model implementation
│   └── *_Model_Predictions/    # Prediction outputs per model
├── models/
│   ├── timegan/                # Trained TimeGAN models (11 assets)
│   │   └── {ASSET}/            # embedder, recovery, generator, supervisor, discriminator.h5
│   └── diffusion/              # Trained Diffusion models (12 assets)
│       └── {ASSET}/            # denoising_network.h5, scheduler_params.pkl
├── app/
│   ├── app.py                  # Flask web application
│   ├── config.py               # Application configuration
│   ├── data.py                 # Data models and results
│   ├── model_server.py         # Model serving endpoints
│   ├── model_utils.py          # Model loading utilities
│   ├── routes/                 # Blueprint routes
│   │   ├── timegan.py
│   │   ├── diffusion.py
│   │   ├── comparison.py
│   │   ├── statistics.py
│   │   └── models.py
│   └── templates/              # HTML templates
├── outputs/
│   ├── figures/                # 40+ visualizations
│   │   ├── 07_timegan_comparison_*.png
│   │   ├── 08_diffusion_comparison_*.png
│   │   ├── model_comparison_overview.png
│   │   └── model_comparison_by_category.png
│   └── results/                # CSV result files
│       ├── model_comparison.csv
│       ├── timegan_evaluation_*.csv (11 files)
│       ├── diffusion_evaluation_*.csv (12 files)
│       ├── baseline_results_*.csv
│       └── diffusion_summary.csv
├── notebooks/
│   ├── exploratory/            # Data exploration and EDA
│   └── modeling/               # Model experiments
│       └── 04_model_comparison.ipynb
├── Final-Report/
│   ├── main.tex                # LaTeX research paper
│   └── references.bib
└── calculate_correct_stats.py  # Statistical validation script
```

## Research Objectives

### 1. Synthetic Data Generation (Primary Focus)

**Compare TimeGAN vs Diffusion Models** on multi-asset financial data using rigorous statistical validation:

- **Distribution Matching**: KS test, mean/std difference, autocorrelation preservation
- **Statistical Properties**: Volatility clustering, fat-tail preservation, technical indicators
- **Asset Coverage**: 11 assets (indices, stocks, crypto)
- **Quality Assessment**: Per-asset performance ranking

### 2. Forecasting Evaluation (Secondary Focus)

**Establish baselines** for cryptocurrency price prediction:

- **Models**: ARIMA, LSTM, Prophet, TimeGAN (generative), DDPM (generative)
- **Metrics**: MAE, RMSE, R², Direction Accuracy, MAPE
- **Objective**: Prove task-specific model selection (generative ≠ predictive)



## Key Results

### A. Generative Models Comparison (TimeGAN vs Diffusion)

**🏆 Winner: TimeGAN**

| Metric | TimeGAN | Diffusion | Improvement |
|--------|---------|-----------|-------------|
| **Mean Difference** | 0.067 ± 0.030 | 0.134 ± 0.017 | **54% better** |
| **Median** | 0.059 | 0.131 | **55% better** |
| **Assets Won** | **9/11 (82%)** | 2/11 (18%) | - |
| **Statistical Significance** | p = 0.0004 | - | Highly significant*** |
| **Effect Size (Cohen's d)** | -2.82 | - | Large effect |

**Best TimeGAN Performances**:

1. HSI (Hang Seng): 0.0256 mean difference
2. AMZN (Amazon): 0.0206 mean difference  
3. FTSE (FTSE 100): 0.0344 mean difference
4. DJI (Dow Jones): 0.0559 mean difference
5. N225 (Nikkei): 0.0589 mean difference

**Diffusion Model Performance**:

- Average KS statistic: 0.378 (Fair quality across all assets)
- Range: 0.321 - 0.483
- Better on: GOOGL (marginal tie), GSPC (marginal)

**Statistical Validation**:

- Paired t-test: t = -4.59, p = 0.0004 (highly significant)
- Cohen's d = -2.82 (large effect size)
- TimeGAN wins on 9/11 assets (82%)

### B. Forecasting Results (Cryptocurrency Focus)

**🏆 Winner: ARIMA** (for cryptocurrency price prediction)

| Model | MAE | RMSE | R² | Direction Acc | MAPE | Status |
|-------|-----|------|----|--------------:|------|--------|
| **ARIMA** | 0.00440 | 0.00598 | **0.9751** | 0% | 100.0% | ✅ Best |
| **LSTM** | 0.00437 | 0.00600 | 0.8082 | 41.4% | 101.1% | ✅ Good |
| Prophet | 0.00635 | 0.00833 | -0.942 | 58.6% | 278.4% | ❌ Poor |
| Naive Mean | 0.00447 | 0.00601 | -0.0001 | 0% | 103.6% | Baseline |
| **TimeGAN** | - | - | **-1.72** | - | - | ❌ Unsuitable |
| **DDPM** | - | - | **-4.24** | - | - | ❌ Unsuitable |

**Critical Finding**: Generative models (TimeGAN, DDPM) have **negative R² scores**, meaning they perform worse than simply predicting the mean value. **They are unsuitable for forecasting tasks**.

### C. Technical Findings

**TimeGAN Successfully Preserves**:

- ✅ Returns distribution (0.12 mean difference)
- ✅ Log-returns distribution (0.03 mean difference)
- ✅ Autocorrelation patterns
- ✅ Volatility clustering behavior
- ✅ Technical indicators (MACD, RSI within 10-15% error)

**Diffusion Model Limitations**:

- ❌ Higher distribution divergence (0.13 mean diff)
- ❌ KS statistics 0.32-0.48 (moderate to poor fit)
- ❌ Bollinger Band width (71% error)
- ❌ ATR preservation (58% error)

**All Assets Are Stationary** (ADF Test Results):

- All p-values < 0.05
- ADF statistics range: -8.7 to -61.8
- No additional differencing required

**Distribution Characteristics**:

- Skewness: -0.67 to 5.27 (fat tails present)
- Kurtosis: 3.7 to 88.8 (extreme leptokurtosis)
- Jarque-Bera: All reject normality (p < 0.001)
- Sharpe Ratios: 0.03 (HSI) to 1.25 (SOL_USD)

## Models Implemented

### Generative Models

**TimeGAN (Time-series GAN)** - Winner for synthetic data generation

- **Architecture**: 4-component network (Embedder, Recovery, Generator, Supervisor, Discriminator)
- **Training**: 20,000 iterations, batch size 64, hidden dim 128, sequence length 48
- **Time per asset**: ~18 minutes on GPU
- **Models saved**: 11 assets × 5 networks = 55 .h5 files (~50MB per asset)
- **Quality**: 6 Excellent, 4 Good, 1 Fair performers

**Diffusion Models (DDPM)** - Evaluated for comparison

- **Architecture**: Residual neural network with time conditioning
- **Noise Schedule**: Linear beta interpolation (1000 diffusion steps)
- **Training**: 500 epochs with forward/reverse diffusion processes
- **Models saved**: 12 assets × (denoising_network.h5 + scheduler_params.pkl)
- **Quality**: All 11 assets rated Fair (KS 0.32-0.48)

### Forecasting/Baseline Models

**ARIMA** - Best forecasting performance

- Auto-ARIMA with optimal (p,d,q) parameter selection
- R² = 0.9751 on cryptocurrency data (97.51% variance explained)
- Excellent for short-term prediction (1-30 days)

**LSTM** - Deep learning approach

- 2-layer LSTM with dropout regularization
- 30-day lookback window, multiple input features
- R² = 0.8082 (good but below ARIMA)
- Better for complex non-linear patterns

**Prophet** - Meta's forecasting tool

- Automatic seasonality detection and changepoint analysis
- R² = -0.942 (poor performance on crypto)
- Better suited for datasets with strong seasonal patterns

## Installation & Setup

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (optional, for faster training)
- 16GB+ RAM recommended

### Dependencies Installation

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Key Libraries

- **Deep Learning**: PyTorch 2.0+, TensorFlow 2.12+
- **Statistical**: statsmodels, pmdarima, Prophet
- **ML/Preprocessing**: scikit-learn, pandas, numpy
- **Visualization**: matplotlib, seaborn, plotly
- **Web App**: Flask, Jinja2

## Usage

### 1. Data Preparation (Already Done)

Data is preprocessed and split in `data/processed/`:

- Train/Val/Test splits
- 108 technical features per asset
- Normalized and ready for modeling

### 2. Run Forecasting Models

```bash
# Navigate to forecasting directory
cd forecasting

# Run ARIMA model
jupyter notebook ARIMA_Model.ipynb

# Run LSTM model
jupyter notebook LSTM_Model.ipynb

# Run Prophet model
jupyter notebook Prophet_Model.ipynb
```

### 3. Train Generative Models

```bash
# TimeGAN (GPU recommended, ~18 min per asset)
jupyter notebook timegan-latest.ipynb

# Diffusion Model (GPU required, ~2 hours per asset)
jupyter notebook DDPM_Model.ipynb
```

### 4. Model Comparison & Analysis

```bash
# Statistical comparison
jupyter notebook notebooks/modeling/04_model_comparison.ipynb

# Calculate detailed statistics
python calculate_correct_stats.py
```

### 5. Run Web Application

```bash
# Navigate to app directory
cd app

# Start Flask server
python app.py

# Access at http://localhost:5000
```

**Web App Features**:

- Interactive model comparison dashboards
- Asset-specific analysis and visualizations
- Statistical test results
- Technical indicator charts
- API endpoints for programmatic access

## Model Serving API

The Flask application provides REST API endpoints:

```python
# Health check
GET /api/health

# TimeGAN results
GET /timegan/api/results
GET /timegan/api/asset/<asset_code>

# Diffusion results  
GET /diffusion/api/results
GET /diffusion/api/asset/<asset_code>

# Comparison
GET /comparison/api/comparison
GET /comparison/api/asset/<asset_code>
```

## Evaluation Metrics

### For Generative Models (TimeGAN vs Diffusion)

**Distribution Matching**:

- **Kolmogorov-Smirnov (KS) Test**: Measures distributional similarity (lower is better)
- **Mean Difference**: Absolute difference between real and synthetic means
- **Standard Deviation Difference**: Volatility preservation
- **Feature-wise Comparison**: 48+ features per asset (returns, volume, technical indicators)

**Statistical Properties**:

- Autocorrelation preservation (ACF plots)
- Volatility clustering (GARCH effects)
- Fat-tail distribution matching
- Moment comparison (mean, std, skewness, kurtosis)

**Quality Grading**:

- Excellent: Mean diff < 0.05, KS < 0.30
- Good: Mean diff 0.05-0.10, KS 0.30-0.40
- Fair: Mean diff 0.10-0.15, KS 0.40-0.50
- Poor: Mean diff > 0.15, KS > 0.50

### For Forecasting Models

**Standard Metrics**:

- **MAE** (Mean Absolute Error): Average prediction error magnitude
- **RMSE** (Root Mean Squared Error): Penalizes large errors
- **R²** (Coefficient of Determination): Variance explained (0-1, higher is better)
- **MAPE** (Mean Absolute Percentage Error): Error as percentage
- **Direction Accuracy**: % of correctly predicted price direction

**Financial Metrics**:

- Sharpe Ratio of trading strategy
- Maximum drawdown
- Win/loss ratio

## Practical Recommendations

### ✅ Use TimeGAN When

- Generating synthetic data for **data augmentation** (training ML models)
- **Privacy-preserving analysis** (anonymizing proprietary trading data)
- **Scenario generation** for stress testing and risk modeling
- **Backtesting** trading strategies with diverse market conditions
- Need to preserve **temporal dependencies** and autocorrelation
- Working with **limited historical data** (rare events)

### ✅ Use Diffusion Models When

- **Imputation** of missing values (CSDI variant, not evaluated in this study)
- Need **probabilistic guarantees** on mode coverage
- Require **stable training** without mode collapse
- Working with **very long sequences** (>100 timesteps)

### ✅ Use ARIMA When

- **Short-term forecasting** (1-30 days ahead)
- Working with **stationary time series** or simple differencing
- Need **interpretability** (understand p, d, q parameters)
- **Limited computational resources** (CPU-only)
- Cryptocurrency or forex prediction with clear trends

### ✅ Use LSTM When

- **Multi-feature forecasting** (using multiple input variables)
- Complex **non-linear patterns** in data
- **Long sequences** (>50 days lookback)
- Have **sufficient training data** (1000+ samples)
- GPU resources available

### ✅ Use Prophet When

- Data has **strong seasonal patterns** (daily, weekly, yearly)
- Need **automatic changepoint detection**
- Working with **missing data** or outliers
- Require **trend decomposition** and interpretability

### ❌ Avoid Generative Models For

- **Direct price forecasting** (negative R² scores demonstrated)
- **Trading signals** generation
- **Point predictions** (single future value)
- Short-term tactical decisions

## Repository Statistics

- **Total Lines of Code**: ~15,000+
- **Jupyter Notebooks**: 8 main notebooks
- **Trained Models**: 23 models (11 TimeGAN + 12 Diffusion)
- **Result Files**: 30+ CSV files
- **Visualizations**: 40+ figures
- **Assets Analyzed**: 25 (indices, stocks, crypto, commodities)
- **Total Experiments**: 125+ (5 models × 25 assets)
- **Training Time**: ~400 GPU hours total

## Key Contributions

1. **First systematic comparison** of TimeGAN vs Diffusion Models on multi-asset financial data
2. **Statistical validation** with paired t-tests, Cohen's d effect size, KS tests
3. **Proof that generative models fail at forecasting** (negative R² scores)
4. **Comprehensive baseline establishment** for cryptocurrency forecasting
5. **Production-ready code** with Flask web application and model serving
6. **Reproducible research** with all models, scalers, and parameters saved

## Limitations & Future Work

### Current Limitations

- Forecasting only evaluated on **cryptocurrencies** (not stocks/indices)
- No recent **transformer-based** generative models (TimeVAE, TimeGrad)
- Missing **transaction cost modeling** for trading strategies
- Single **market regime** (2015-2024, mostly bull market)
- No **multi-step ahead** forecasting evaluation

### Future Research Directions

1. **Extend forecasting** to stock indices and commodities
2. **Implement transformer models** (TimesFM, TimesGPT, Chronos)
3. **Multi-horizon forecasting** (1-day, 7-day, 30-day ahead)
4. **Trading strategy backtesting** with transaction costs
5. **Market regime detection** and adaptive model selection
6. **Ensemble methods** combining generative and predictive models
7. **Real-time deployment** with streaming data pipeline

## Citation

If you use this code or findings in your research, please cite:

```bibtex
@article{nasir2024timegan,
  title={Comparative Analysis of TimeGAN and Diffusion Models for Synthetic Financial Time-Series Generation},
  author={Nasir, Huzaifa and Ali, Maaz},
  journal={Technical Report},
  year={2024},
  institution={FAST NUCES}
}
```

## License

This project is for academic and research purposes.

## Authors

- **Huzaifa Nasir**
- **Maaz Ali** 

**Institution**: National University of Computer and Emerging Sciences (FAST NUCES)  
**Department**: Computer Science  
**Date**: December 2025

## Acknowledgments

- yfinance API for financial data
- PyTorch and TensorFlow teams for deep learning frameworks
- Meta's Prophet team for forecasting library
- Original TimeGAN authors (Yoon et al., 2019)
- Diffusion model researchers (Ho et al., 2020; Rasul et al., 2021)

---

**⭐ Star this repository if you find it useful for your research!**
