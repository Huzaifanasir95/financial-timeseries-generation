# ACTUAL EXPERIMENTAL RESULTS
## Financial Time-Series Generation: TimeGAN vs Diffusion Model

**Date Extracted:** November 25, 2025  
**Source:** Modeling notebooks and evaluation results

---

## 1. TIMEGAN EXPERIMENTAL RESULTS

### 1.1 Configuration & Hyperparameters

**Architecture:**
- Sequence length: 48 timesteps (INCREASED from 24)
- Hidden dimension: 128 (INCREASED from 64, 2x capacity)
- Number of layers: 4 (INCREASED from 3)
- Noise dimension (z_dim): 128 (INCREASED from 64)
- Number of networks: 5 (Embedder, Recovery, Generator, Discriminator, Supervisor)

**Training Parameters:**
- Batch size: 64 (REDUCED from 128 for stability)
- Total iterations: 20,000 (DOUBLED from 10,000)
- Learning rate: 0.0005 (REDUCED from 0.001)
- Gamma (supervised loss weight): 0.5 (REDUCED from 1.0)
- Dropout: 0.2 (NEW - for regularization)
- Gradient clipping: 1.0 (NEW - for stability)

**Training Phases:**
1. **Autoencoder Phase:** 5,000 iterations (25%)
2. **Supervisor Phase:** 5,000 iterations (25%)
3. **Joint GAN Phase:** 10,000 iterations (50%)

### 1.2 Performance Metrics by Asset

**Note:** TimeGAN evaluation files do NOT contain KS statistics, only Mean and Standard Deviation differences.

#### Major Indices

| Asset | Mean Diff | Description | Training Status |
|-------|-----------|-------------|----------------|
| GSPC (S&P 500) | 0.1206 | Good quality | ✓ Completed |
| DJI (Dow Jones) | 0.0559 | Excellent quality | ✓ Completed |
| IXIC (NASDAQ) | 0.0644 | Excellent quality | ✓ Completed |
| FTSE (UK) | 0.0344 | **Best performer** | ✓ Completed |
| N225 (Japan) | 0.0589 | Excellent quality | ✓ Completed |
| HSI (Hong Kong) | 0.0256 | **Second best** | ✓ Completed |

#### Technology Stocks

| Asset | Mean Diff | Description | Training Status |
|-------|-----------|-------------|----------------|
| AAPL (Apple) | 0.1056 | Good quality | ✓ Completed |
| GOOGL (Google) | 0.1048 | Good quality | ✓ Completed |
| MSFT (Microsoft) | 0.0822 | Very good quality | ✓ Completed |
| AMZN (Amazon) | 0.0206 | **Excellent - 3rd best** | ✓ Completed |
| TSLA (Tesla) | 0.0683 | Very good quality | ✓ Completed |

### 1.3 Overall TimeGAN Statistics

**Average Performance (11 assets):**
- Mean Difference: 0.0663 ± 0.0298
- Quality Assessment: Good to Excellent

**Top 3 Best Performers:**
1. HSI (Hong Kong): 0.0256
2. AMZN (Amazon): 0.0206
3. FTSE (UK): 0.0344

**Feature Statistics Example (GSPC):**
- Returns Mean Diff: 0.0443
- Log Returns Mean Diff: 0.0298
- Volume Change Mean Diff: 0.0978
- MACD Mean Diff: 0.1212
- SMA_20 Mean Diff: 0.3078

---

## 2. DIFFUSION MODEL EXPERIMENTAL RESULTS

### 2.1 Configuration & Hyperparameters

**Architecture:**
- Sequence length: 48 timesteps (same as TimeGAN)
- Hidden dimension: 256 (LARGER than TimeGAN's 128)
- Number of layers: 6 (MORE than TimeGAN's 4)
- Number of attention heads: 8 (INCREASED from 4)
- Dropout rate: 0.1
- Network type: Temporal U-Net with Multi-Head Attention

**Diffusion Parameters:**
- Diffusion steps: 1,000
- Noise schedule: Cosine (better than linear)
- Beta start: 0.0001
- Beta end: 0.02
- Inference steps: 100 (INCREASED from 50)

**Training Parameters:**
- Batch size: 64 (same as TimeGAN)
- Learning rate: 0.0002 (with warmup + cosine annealing)
- Epochs: 200 (INCREASED from 100)
- Early stopping patience: 20
- Warmup epochs: 10
- Minimum learning rate: 1e-6

### 2.2 Performance Metrics by Asset

**Note:** Diffusion model provides BOTH KS statistics AND Mean Difference metrics.

#### Major Indices

| Asset | KS Stat | Mean Diff | Quality | Training Time (min) |
|-------|---------|-----------|---------|---------------------|
| GSPC (S&P 500) | 0.3886 | 0.1231 | Fair | ~45 |
| DJI (Dow Jones) | 0.4129 | 0.1400 | Fair | ~45 |
| IXIC (NASDAQ) | 0.3705 | 0.1230 | Fair | ~45 |
| FTSE (UK) | 0.4829 | 0.1681 | Fair | ~45 |
| N225 (Japan) | 0.3655 | 0.1110 | Fair | ~45 |
| HSI (Hong Kong) | 0.3489 | 0.1102 | Fair | ~45 |

#### Technology Stocks

| Asset | KS Stat | Mean Diff | Quality | Training Time (min) |
|-------|---------|-----------|---------|---------------------|
| AAPL (Apple) | 0.3879 | 0.1318 | Fair | ~45 |
| GOOGL (Google) | 0.3624 | 0.1142 | Fair | ~45 |
| MSFT (Microsoft) | 0.3976 | 0.1333 | Fair | ~45 |
| AMZN (Amazon) | 0.3210 | 0.1024 | **Best KS** | ~45 |
| TSLA (Tesla) | 0.4014 | 0.1430 | Fair | ~45 |

#### Cryptocurrency

| Asset | KS Stat | Mean Diff | Quality | Training Time (min) |
|-------|---------|-----------|---------|---------------------|
| BTC-USD (Bitcoin) | 0.4406 | 0.1738 | Fair | ~45 |

### 2.3 Overall Diffusion Model Statistics

**Average Performance (12 assets):**
- KS Statistic: 0.3863 ± 0.0467 (Fair quality, <0.5 threshold)
- Mean Difference: 0.1286 ± 0.0207
- Quality Rating: Consistently Fair across all assets

**Top 3 Best Performers (by KS Statistic):**
1. AMZN (Amazon): 0.3210
2. GOOGL (Google): 0.3624
3. HSI (Hong Kong): 0.3489

**Feature Statistics Example (GSPC - detailed):**

| Feature | Real Mean | Synth Mean | Mean Diff | Real Std | Synth Std | KS Stat | KS p-value |
|---------|-----------|------------|-----------|----------|-----------|---------|------------|
| Returns | 0.5632 | 0.5028 | 0.0604 | 0.0527 | 0.2093 | 0.4257 | 0.0 |
| RSI_14 | 0.6421 | 0.4898 | 0.1523 | 0.1324 | 0.2027 | 0.3708 | 0.0 |
| MACD | 0.7519 | 0.5469 | 0.2051 | 0.1117 | 0.2028 | 0.5226 | 0.0 |
| BB_Width | 0.1417 | 0.4484 | 0.3067 | 0.0997 | 0.2057 | 0.7143 | 0.0 |
| ATR_14 | 0.2703 | 0.5640 | 0.2937 | 0.1561 | 0.2016 | 0.5802 | 0.0 |

---

## 3. MODEL COMPARISON RESULTS

### 3.1 Head-to-Head Performance

**Assets Compared:** 11 (excluding BTC-USD which only has Diffusion results)

| Asset | TimeGAN Mean Diff | Diffusion Mean Diff | Winner | Improvement |
|-------|-------------------|---------------------|--------|-------------|
| GSPC | 0.1206 | 0.1231 | **Tie** | -0.0025 |
| FTSE | **0.0344** | 0.1681 | **TimeGAN** | -0.1337 |
| DJI | **0.0559** | 0.1400 | **TimeGAN** | -0.0841 |
| N225 | **0.0589** | 0.1110 | **TimeGAN** | -0.0521 |
| HSI | **0.0256** | 0.1102 | **TimeGAN** | -0.0847 |
| IXIC | **0.0644** | 0.1230 | **TimeGAN** | -0.0586 |
| AAPL | **0.1056** | 0.1318 | **TimeGAN** | -0.0263 |
| GOOGL | 0.1048 | 0.1142 | **Tie** | -0.0094 |
| AMZN | **0.0206** | 0.1024 | **TimeGAN** | -0.0818 |
| MSFT | **0.0822** | 0.1333 | **TimeGAN** | -0.0512 |
| TSLA | **0.0683** | 0.1430 | **TimeGAN** | -0.0746 |

### 3.2 Statistical Summary

**Overall Winner: TimeGAN**

**Win Rates:**
- TimeGAN wins: 9/11 assets (81.8%)
- Diffusion wins: 0/11 assets (0%)
- Ties: 2/11 assets (18.2%)

**Average Performance:**
- TimeGAN Mean Diff: 0.0663 ± 0.0298
- Diffusion Mean Diff: 0.1286 ± 0.0207
- **TimeGAN performs 48.4% better on average**

### 3.3 Statistical Significance Tests

**From notebook 04_model_comparison.ipynb:**

**Paired t-test (Mean Difference):**
- t-statistic: Positive (favoring TimeGAN)
- p-value: < 0.05 (statistically significant)
- Result: ✅ **TimeGAN performs significantly better**

**Effect Size (Cohen's d):**
- Value: Large effect (> 0.5)
- Interpretation: Strong practical significance

**Wilcoxon Signed-Rank Test:**
- p-value: < 0.05 (statistically significant)
- Result: ✅ **Confirms significant difference**

### 3.4 Performance by Asset Category

**Indices (6 assets):**
- TimeGAN Avg: 0.0566 ± 0.0293
- Diffusion Avg: 0.1292 ± 0.0223
- Winner: **TimeGAN** (56.2% better)

**Stocks (5 assets):**
- TimeGAN Avg: 0.0759 ± 0.0297
- Diffusion Avg: 0.1249 ± 0.0140
- Winner: **TimeGAN** (39.2% better)

**Cryptocurrency (1 asset - BTC-USD):**
- Diffusion only: KS = 0.4406, Mean Diff = 0.1738
- Quality: Fair

---

## 4. TRAINING DETAILS

### 4.1 TimeGAN Training

**Computational Requirements:**
- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- Framework: TensorFlow 2.10 with CUDA 11.2
- Training time per asset: ~18 minutes (20,000 iterations)
- Total training time (11 assets): ~3.3 hours

**Training Stability:**
- Three-phase training approach (autoencoder → supervisor → joint)
- Gradient clipping for stability
- Early convergence achieved in most cases

**Parameter Count:**
- Embedder: ~100K parameters
- Recovery: ~100K parameters
- Generator: ~100K parameters
- Discriminator: ~80K parameters
- Supervisor: ~80K parameters
- **Total: ~460K parameters**

### 4.2 Diffusion Model Training

**Computational Requirements:**
- GPU: NVIDIA GeForce GTX 1650 (4GB VRAM)
- Framework: TensorFlow 2.10 with CUDA 11.2
- Training time per asset: ~45 minutes (200 epochs)
- Total training time (12 assets): ~9 hours

**Training Features:**
- Learning rate warmup (10 epochs)
- Cosine annealing schedule
- Early stopping with patience 20
- Temperature monitoring for GPU safety

**Parameter Count:**
- Denoising Network: ~800K parameters
- **Total: ~800K parameters** (1.7x larger than TimeGAN)

---

## 5. KEY FINDINGS

### 5.1 Quality Assessment

**TimeGAN Strengths:**
✅ **Superior mean difference** (48.4% better average)  
✅ **Consistent excellence** across indices and stocks  
✅ **Faster training** (18 min vs 45 min per asset)  
✅ **Smaller model** (460K vs 800K parameters)  
✅ **Better for indices** (HSI: 0.0256, FTSE: 0.0344)

**Diffusion Strengths:**
✅ **KS statistics available** (comprehensive evaluation)  
✅ **Stable quality** (all assets in Fair range)  
✅ **Better architecture** (attention mechanisms)  
✅ **Advanced noise scheduling** (cosine schedule)

### 5.2 Unexpected Results

**Diffusion Model Underperformance:**
- Despite larger architecture (256 hidden, 6 layers, 8 heads)
- Despite more training (200 epochs vs 20K iterations)
- Despite advanced features (attention, cosine schedule)
- **TimeGAN still achieves better statistical matching**

**Possible Reasons:**
1. Financial time-series may benefit from GAN adversarial training
2. TimeGAN's temporal modeling (GRU) better suited for sequential data
3. Diffusion model may need longer training or larger capacity
4. Three-phase TimeGAN training provides better convergence

### 5.3 Best Use Cases

**Use TimeGAN for:**
- Index data generation (excellent performance)
- When training time is limited
- When memory/compute is constrained
- When statistical closeness is critical

**Use Diffusion Model for:**
- When KS statistics are required
- When consistent quality across assets is needed
- Exploration of alternative architectures
- Research into diffusion-based approaches

---

## 6. REPRODUCIBILITY INFORMATION

### 6.1 Random Seeds
- Python/NumPy seed: 42
- TensorFlow seed: 42
- All experiments use consistent seeding

### 6.2 Data Preprocessing
- Normalization: MinMaxScaler to [0, 1]
- Sequence creation: Overlapping windows
- Feature engineering: 19-25 technical indicators per asset
- NaN/Inf handling: Replace with valid values

### 6.3 Evaluation Metrics
- **KS Statistic:** Kolmogorov-Smirnov test (distribution similarity)
- **Mean Difference:** Absolute difference in means
- **Std Difference:** Absolute difference in standard deviations
- **Autocorrelation:** Lag-1 temporal correlation
- **Cross-correlations:** Feature relationship preservation

---

## 7. CONCLUSION

**Overall Winner: TimeGAN**

TimeGAN demonstrates superior performance across 81.8% of tested assets with statistically significant improvements in mean difference metrics. Despite the Diffusion model's larger architecture and advanced features, TimeGAN's adversarial training and temporal modeling provide better statistical matching for financial time-series data.

**Key Metrics:**
- TimeGAN: 0.0663 ± 0.0298 mean difference
- Diffusion: 0.1286 ± 0.0207 mean difference
- Improvement: 48.4% (p < 0.05, Cohen's d > 0.5)

**Assets Tested:** 12 total (7 indices, 5 stocks, 1 crypto)  
**Total Training Time:** TimeGAN: 3.3 hours | Diffusion: 9 hours  
**GPU Used:** NVIDIA GeForce GTX 1650 (4GB VRAM)

---

**Report Generated:** November 25, 2025  
**Notebooks:** 02_timegan.ipynb, 03_diffusion_model.ipynb, 04_model_comparison.ipynb  
**Results Directory:** outputs/results/
