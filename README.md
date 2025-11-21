# Synthetic Financial Time-Series Generation

A comparative study of diffusion models and GANs for generating synthetic financial market data.

## Project Overview

This project investigates modern generative models (diffusion models and GANs) for creating synthetic financial time-series data. The goal is to evaluate their effectiveness in preserving statistical properties of real markets while enabling data augmentation for improved forecasting and risk management.

## Project Structure

```
financial-timeseries-generation/
├── data/
│   ├── raw/              # Original financial data from APIs
│   ├── processed/        # Cleaned and preprocessed data
│   ├── synthetic/        # Generated synthetic data
│   └── external/         # External datasets
├── notebooks/
│   ├── exploratory/      # Data exploration and EDA
│   ├── baseline_models/  # ARIMA, LSTM, Prophet experiments
│   ├── generative_models/# GAN and Diffusion model experiments
│   └── evaluation/       # Model evaluation and comparison
├── src/
│   ├── data/            # Data loading and preprocessing modules
│   ├── models/
│   │   ├── baseline/    # Traditional forecasting models
│   │   └── generative/  # GAN and Diffusion implementations
│   ├── evaluation/      # Metrics and evaluation functions
│   ├── utils/           # Helper functions
│   └── visualization/   # Plotting and visualization tools
├── experiments/
│   ├── configs/         # Experiment configuration files
│   ├── results/         # Experiment results and metrics
│   └── logs/            # Training logs
├── models/
│   ├── saved_models/    # Trained model weights
│   └── checkpoints/     # Training checkpoints
├── outputs/
│   ├── figures/         # Generated plots and visualizations
│   ├── tables/          # Results tables
│   └── reports/         # Analysis reports
├── tests/
│   ├── unit/            # Unit tests
│   └── integration/     # Integration tests
└── Report/              # LaTeX project report

```

## Research Objectives

1. **Model Exploration**: Compare diffusion models and GANs for financial time-series synthesis
2. **Comprehensive Benchmarking**: Evaluate against ARIMA, LSTM, Prophet, TimesFM, and TimesGPT
3. **Financial Data Analysis**: Use STL decomposition and statistical tests
4. **Practical Applications**: Demonstrate utility in risk management and forecasting

## Getting Started

### Prerequisites

- Python 3.8+
- pip or conda for package management

### Installation

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Quick Start

1. **Download Data**: Run `notebooks/exploratory/01_data_collection.ipynb`
2. **Explore Data**: Run `notebooks/exploratory/02_eda_and_stl.ipynb`
3. **Baseline Models**: Start with `notebooks/baseline_models/`
4. **Generative Models**: Progress to `notebooks/generative_models/`

## Models Implemented

### Baseline Models
- ARIMA (AutoRegressive Integrated Moving Average)
- LSTM (Long Short-Term Memory Networks)
- Prophet (Facebook's forecasting tool)
- TimesFM (Google's Time-Series Foundation Model)
- TimesGPT (Nixtla's GPT for time-series)

### Generative Models
- TimeGAN (Time-series Generative Adversarial Network)
- DDPM (Denoising Diffusion Probabilistic Models)
- CSDI (Conditional Score-based Diffusion for Imputation)

## Evaluation Metrics

- **Forecasting**: MAPE, MAE, RMSE
- **Statistical**: KS test, distribution matching, moment comparison
- **Financial**: Volatility clustering, fat-tail preservation, VaR estimation

## Authors

- Huzaifa Nasir (i221053@nu.edu.pk)
- Maaz Ali (i221042@nu.edu.pk)

Department of Computer Science  
National University of Computer and Emerging Sciences

## License

This project is for academic purposes as part of a Generative AI course project.
