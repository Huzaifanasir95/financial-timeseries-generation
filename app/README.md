# Financial Time-Series Generation Flask Application

## Overview

This Flask web application provides an interactive interface to explore the results of our comparative study between TimeGAN and Diffusion Models for synthetic financial time-series generation.

## Real Experimental Data

All data displayed in this application comes from actual experiments conducted on 12 financial assets:

### Assets Analyzed
- **Indices (6)**: S&P 500 (GSPC), FTSE 100, Dow Jones (DJI), Nikkei 225 (N225), Hang Seng (HSI), NASDAQ (IXIC)
- **Stocks (5)**: Apple (AAPL), Alphabet (GOOGL), Amazon (AMZN), Microsoft (MSFT), Tesla (TSLA)
- **Cryptocurrency (1)**: Bitcoin (BTC-USD)

### Key Results

**TimeGAN Performance:**
- Mean Difference: 0.0663 ± 0.0298
- Training Time: ~18 minutes per asset
- Model Size: ~460K parameters
- Win Rate: 81.8% (9/11 assets)

**Diffusion Model Performance:**
- Mean Difference: 0.1286 ± 0.0207
- KS Statistic: 0.3863 ± 0.0467
- Training Time: ~45 minutes per asset
- Model Size: ~800K parameters

**Statistical Significance:**
- p-value: 0.0004 (highly significant)
- Cohen's d: -2.21 (very large effect size)
- Improvement: 48.4% better mean difference

## Application Structure

```
app/
├── app.py                 # Main Flask application
├── config.py             # Configuration settings
├── data.py               # Real experimental results data
├── routes/               # Route blueprints
│   ├── timegan.py       # TimeGAN analysis routes
│   ├── diffusion.py     # Diffusion model routes
│   ├── comparison.py    # Comparative analysis routes
│   ├── statistics.py    # Statistical tests routes
│   ├── technical.py     # Technical details routes
│   └── recommendations.py
├── templates/            # HTML templates
│   ├── base.html        # Base template with Tailwind CSS
│   ├── index.html       # Executive summary
│   ├── timegan/         # TimeGAN analysis pages
│   ├── diffusion/       # Diffusion model pages
│   ├── comparison/      # Comparison pages
│   ├── statistics/      # Statistical analysis pages
│   ├── technical/       # Technical deep dive pages
│   └── recommendations/ # Recommendations pages
└── static/
    └── images/          # Comparison visualizations (copied from Final-Report/images)
```

## Features

### 1. Executive Summary (`/`)
- Overall performance comparison
- Key findings and metrics
- Win rates and statistical significance
- Category-based analysis (Indices vs Stocks)

### 2. TimeGAN Analysis (`/timegan`)
- Architecture overview (5 networks: Embedder, Recovery, Generator, Supervisor, Discriminator)
- **Mathematical formulations** (collapsible widget):
  - Reconstruction Loss: L_R = E[||x - r(e(x))||²]
  - Supervised Loss: L_S = Σ ||h_t - g(e(x_{<t}), z_t)||²
  - Adversarial Loss: L_A = E[log d(e(x))] + E[log(1 - d(g(z)))]
  - Total Loss: L_total = 10·L_R + 0.1·L_S + 1·L_A
- Three-phase training strategy
- Asset-specific results with **real comparison images**
- Training configuration and performance metrics

### 3. Diffusion Model Analysis (`/diffusion`)
- Transformer-based architecture (6 layers, 8 heads, 256 hidden dim)
- **Mathematical formulations** (collapsible widget):
  - Forward Process: q(x_t | x_{t-1}) = N(x_t; √(1-β_t)·x_{t-1}, β_t·I)
  - Direct Sampling: q(x_t | x_0) = N(x_t; √ᾱ_t·x_0, (1-ᾱ_t)·I)
  - Reverse Process: p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t,t), Σ_θ(x_t,t))
  - Training Loss: L_simple = E[||ε - ε_θ(x_t, t)||²]
- Noise scheduling (cosine schedule, 1000 steps)
- Asset-specific results with **real comparison images**
- Advanced training features (warmup, early stopping, EMA)

### 4. Comparative Analysis (`/comparison`)
- Head-to-head asset comparison
- Interactive comparison chart (Chart.js)
- Win rate breakdown
- Detailed improvement metrics
- Insights on why TimeGAN wins

### 5. Statistical Analysis (`/statistics`)
- Overall performance summary
- Win rate analysis (9/11 for TimeGAN)
- Category performance (Indices: 56.2% improvement, Stocks: 39.2% improvement)
- Statistical methods explanation (KS test, paired t-test, Cohen's d)

### 6. Technical Deep Dive (`/technical`)
- Detailed architecture specifications
- Hyperparameter configurations
- Training procedures
- Computational requirements

### 7. Recommendations (`/recommendations`)
- When to use TimeGAN vs Diffusion
- Best practices
- Future improvements

## Running the Application

### Prerequisites
```bash
pip install flask pandas numpy
```

### Start the Server
```bash
cd app
python app.py
```

The application will be available at `http://localhost:5000`

## Data Sources

All data in this application is sourced from:

1. **Model Evaluation Results**: `outputs/results/`
   - `timegan_evaluation_*.csv` - TimeGAN results per asset
   - `diffusion_evaluation_*.csv` - Diffusion results per asset
   - `model_comparison.csv` - Head-to-head comparison
   - `diffusion_summary.csv` - Diffusion model summary

2. **Comparison Images**: `Final-Report/images/`
   - `07_timegan_comparison_*.png` - TimeGAN visualizations
   - `08_diffusion_comparison_*.png` - Diffusion visualizations
   - `model_comparison_*.png` - Comparison charts

3. **Trained Models**: `models/`
   - `timegan/` - TimeGAN model checkpoints
   - `diffusion/` - Diffusion model checkpoints

## Key Insights Displayed

### TimeGAN Strengths
✅ Superior temporal consistency (supervised loss)
✅ Better mean difference: 0.0663 vs 0.1286
✅ Faster training (18 min vs 45 min)
✅ Smaller model (460K vs 800K params)
✅ Best performers: HSI (0.0256), AMZN (0.0206), FTSE (0.0344)

### Diffusion Limitations
❌ Weaker temporal dependency modeling
❌ Higher mean difference across all assets
❌ Longer training time (2.5× slower)
❌ Larger model size (1.7× more params)
❌ KS stats all in "Fair" range (0.3-0.5)

## Mathematical Formulas

The application includes **collapsible formula widgets** that display:

### TimeGAN Formulas
- Reconstruction, Supervised, and Adversarial losses
- Combined total loss with weights
- Step-by-step training phases

### Diffusion Model Formulas
- Forward diffusion process
- Reverse denoising process
- Mean prediction formula
- Simplified training objective

## Visualizations

Each asset page displays:
- Real vs Synthetic comparison plots
- Feature distribution overlays
- Temporal pattern analysis
- Statistical quality metrics

## Notes

- All metrics are from real experiments (not simulated)
- Images are actual outputs from trained models
- Statistical tests use proper paired comparisons
- Results are reproducible with fixed random seeds (42)

## Authors

- Huzaifa Nasir (i221053@nu.edu.pk)
- Maaz Ali (i221042@nu.edu.pk)

Department of Computer Science
National University of Computer and Emerging Sciences

## License

Academic research project for Generative AI course.
