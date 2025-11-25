# Images Directory

This directory contains all 40 visualizations used in the research paper.

## Directory Structure

### Data Exploration (11 images)
- `01_raw_data_overview.png` - Raw price series for all assets (2015-2024)
- `02_data_splits.png` - Train/validation/test temporal splits
- `02_return_distributions.png` - Daily return distributions by category
- `03_autocorrelation_analysis.png` - ACF for returns and squared returns
- `03_correlation_matrix.png` - Feature correlation heatmap
- `03_normalized_prices.png` - Normalized price series
- `03_rolling_volatility.png` - 30-day rolling volatility
- `03_stl_decomposition_GSPC.png` - STL decomposition for S&P 500
- `03_stl_decomposition_BTC_USD.png` - STL decomposition for Bitcoin
- `04_feature_correlations.png` - Inter-asset feature correlations
- `04_technical_indicators.png` - Technical indicators (RSI, MACD, Bollinger)

### Baseline Comparisons (2 images)
- `05_baseline_comparison_GSPC.png` - ARIMA/LSTM/Prophet on S&P 500
- `06_baseline_all_assets_comparison.png` - MAPE/MAE across all assets

### TimeGAN Results (11 images)
Individual asset comparisons showing real vs synthetic distributions:
- `07_timegan_comparison_AAPL.png` - Apple
- `07_timegan_comparison_AMZN.png` - Amazon
- `07_timegan_comparison_DJI.png` - Dow Jones
- `07_timegan_comparison_FTSE.png` - FTSE 100
- `07_timegan_comparison_GOOGL.png` - Alphabet
- `07_timegan_comparison_GSPC.png` - S&P 500
- `07_timegan_comparison_HSI.png` - Hang Seng
- `07_timegan_comparison_IXIC.png` - NASDAQ
- `07_timegan_comparison_MSFT.png` - Microsoft
- `07_timegan_comparison_N225.png` - Nikkei 225
- `07_timegan_comparison_TSLA.png` - Tesla

### Diffusion Model Results (14 images)
Individual asset comparisons (includes Bitcoin):
- `08_diffusion_comparison_AAPL.png` - Apple
- `08_diffusion_comparison_AMZN.png` - Amazon
- `08_diffusion_comparison_BTC_USD.png` - Bitcoin
- `08_diffusion_comparison_DJI.png` - Dow Jones
- `08_diffusion_comparison_FTSE.png` - FTSE 100
- `08_diffusion_comparison_GOOGL.png` - Alphabet
- `08_diffusion_comparison_GSPC.png` - S&P 500
- `08_diffusion_comparison_HSI.png` - Hang Seng
- `08_diffusion_comparison_IXIC.png` - NASDAQ
- `08_diffusion_comparison_MSFT.png` - Microsoft
- `08_diffusion_comparison_N225.png` - Nikkei 225
- `08_diffusion_comparison_TSLA.png` - Tesla
- `diffusion_training_GSPC.png` - Training progression for S&P 500
- `diffusion_summary.png` - Performance summary across 11 assets

### Comparative Analysis (2 images)
- `model_comparison_overview.png` - 4-panel main comparison
- `model_comparison_by_category.png` - Category-based analysis

## Image Specifications

- **Format**: PNG with transparency
- **Resolution**: 300 DPI (publication quality)
- **Color Scheme**: Seaborn 'husl' palette
- **Typical Dimensions**: 1600x1000 to 2400x1500 pixels
- **Total Size**: ~200 MB

## Usage in LaTeX

All images are referenced in `main.tex` using relative paths:
```latex
\includegraphics[width=0.95\textwidth]{images/filename.png}
```

## Regeneration

To regenerate all figures, run the Jupyter notebooks in order:
1. `01_data_exploration.ipynb`
2. `02_timegan_model.ipynb`
3. `03_diffusion_model.ipynb`
4. `04_model_comparison.ipynb`

All notebooks save figures to `../outputs/figures/` which can then be copied here.

## Categories Summary

| Category | Count | Purpose |
|----------|-------|---------|
| Data Exploration | 11 | Understanding dataset characteristics |
| Baselines | 2 | Context for generative model evaluation |
| TimeGAN | 11 | Individual asset performance |
| Diffusion | 14 | Individual asset + training/summary |
| Comparison | 2 | Head-to-head model comparison |
| **Total** | **40** | **Complete visual documentation** |
