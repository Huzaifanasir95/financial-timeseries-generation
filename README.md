# Synthetic Financial Time-Series Generation

A comprehensive comparative study of TimeGAN vs Diffusion Models for synthetic data generation and evaluation of forecasting models (ARIMA, LSTM, Prophet) across 25 financial assets.

## Project Overview

This research project addresses two fundamental challenges in quantitative finance:

1. **Synthetic Data Generation**: Comparing TimeGAN vs Diffusion Models for creating realistic financial time-series data
2. **Price Forecasting**: Evaluating 5 models (ARIMA, LSTM, Prophet, TimeGAN, DDPM) for trading and prediction tasks

**Key Finding**: TimeGAN outperforms Diffusion Models by 54% for synthetic data generation (p=0.0004), while ARIMA dominates cryptocurrency forecasting with R²=0.975. Critically, generative models (TimeGAN, DDPM) excel at distribution matching but fail at forecasting (negative R² scores).

## Project Structure

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

## Dataset

### Assets Analyzed (25 total)

**Indices (7)**:
- US: S&P 500 (GSPC), NASDAQ (IXIC), Dow Jones (DJI)
- International: FTSE 100, Nikkei 225 (N225), Hang Seng (HSI), DAX (GDAXI)

**Technology Stocks (5)**:
- AAPL, MSFT, GOOGL, AMZN, TSLA

**Traditional Stocks (6)**:
- JPM, XOM, JNJ, V, WMT, PG

**Cryptocurrencies (5)**:
- BTC-USD, ETH-USD, BNB-USD, SOL-USD, ADA-USD

**Commodities (2)**:
- Gold (GC=F), Crude Oil (CL=F)

### Data Characteristics

- **Timespan**: 2015-01-05 to 2024-12-30 (10 years)
- **Frequency**: Daily
- **Features**: 108 technical indicators per asset
  - Price-based: OHLCV, Returns, Log Returns, Price Range
  - Trend: SMA (5/10/20/50), EMA (5/10/20)
  - Momentum: RSI, ROC, Stochastic, Williams %R
  - Volatility: ATR, Bollinger Bands, Historical Volatility, Keltner Channels
  - Volume: OBV, Volume SMA, Volume ROC, MFI, CMF
  - Trend Strength: MACD, ADX, Ichimoku
- **Splits**: Train (70%), Validation (15%), Test (15%)
- **Stationarity**: All assets stationary (ADF test p < 0.05)
- **Total samples**: 2,443 - 3,651 rows depending on asset

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
