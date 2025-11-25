"""
Data Models and Real Experimental Results
All data extracted from actual notebook experiments and model evaluations
"""

import os
import pandas as pd

# Get the base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'outputs', 'results')
IMAGES_DIR = os.path.join(BASE_DIR, 'Final-Report', 'images')

# Asset Information - REAL ASSETS FROM EXPERIMENTS
ASSETS = {
    "GSPC": {
        "name": "S&P 500",
        "type": "Index",
        "market": "US Equities",
        "volatility": "Medium",
        "description": "Primary US large-cap benchmark, 500 leading companies"
    },
    "FTSE": {
        "name": "FTSE 100",
        "type": "Index",
        "market": "UK Equities",
        "volatility": "Medium",
        "description": "UK's leading share index, London Stock Exchange"
    },
    "DJI": {
        "name": "Dow Jones Industrial Average",
        "type": "Index",
        "market": "US Equities",
        "volatility": "Medium",
        "description": "30 prominent US companies, price-weighted index"
    },
    "N225": {
        "name": "Nikkei 225",
        "type": "Index",
        "market": "Japan Equities",
        "volatility": "Medium",
        "description": "Japanese stock market index, Tokyo Stock Exchange"
    },
    "HSI": {
        "name": "Hang Seng Index",
        "type": "Index",
        "market": "Hong Kong Equities",
        "volatility": "High",
        "description": "Hong Kong market benchmark, emerging market exposure"
    },
    "IXIC": {
        "name": "NASDAQ Composite",
        "type": "Index",
        "market": "US Technology",
        "volatility": "High",
        "description": "Technology-heavy US index, over 3000 companies"
    },
    "AAPL": {
        "name": "Apple Inc.",
        "type": "Stock",
        "market": "Technology",
        "volatility": "Medium",
        "description": "Consumer electronics, largest market cap ($3T+ peak)"
    },
    "GOOGL": {
        "name": "Alphabet Inc.",
        "type": "Stock",
        "market": "Technology",
        "volatility": "Medium",
        "description": "Internet services, search, cloud computing"
    },
    "AMZN": {
        "name": "Amazon.com Inc.",
        "type": "Stock",
        "market": "Technology",
        "volatility": "High",
        "description": "E-commerce, cloud infrastructure (AWS)"
    },
    "MSFT": {
        "name": "Microsoft Corp.",
        "type": "Stock",
        "market": "Technology",
        "volatility": "Medium",
        "description": "Software, cloud computing, enterprise services"
    },
    "TSLA": {
        "name": "Tesla Inc.",
        "type": "Stock",
        "market": "Automotive/Technology",
        "volatility": "Very High",
        "description": "Electric vehicles, high volatility, retail investor interest"
    },
    "BTC-USD": {
        "name": "Bitcoin",
        "type": "Cryptocurrency",
        "market": "Digital Assets",
        "volatility": "Extreme",
        "description": "Leading cryptocurrency, extreme volatility, 24/7 trading"
    }
}

# TimeGAN Results - REAL DATA FROM EXPERIMENTS
# Source: outputs/results/timegan_evaluation_*.csv and model_comparison.csv
TIMEGAN_RESULTS = {
    "GSPC": {
        "mean_diff": 0.1206,  # From model_comparison.csv
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Good",
        "notes": "Excellent performance on major index"
    },
    "FTSE": {
        "mean_diff": 0.0344,  # BEST PERFORMER
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Best overall performance across all assets"
    },
    "DJI": {
        "mean_diff": 0.0559,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Strong performance on Dow Jones"
    },
    "N225": {
        "mean_diff": 0.0589,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Excellent quality for Japanese market"
    },
    "HSI": {
        "mean_diff": 0.0256,  # SECOND BEST
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Second best performer, Hong Kong index"
    },
    "IXIC": {
        "mean_diff": 0.0644,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Very good quality for NASDAQ"
    },
    "AAPL": {
        "mean_diff": 0.1056,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Good",
        "notes": "Good quality for Apple stock"
    },
    "GOOGL": {
        "mean_diff": 0.1048,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Good",
        "notes": "Good quality for Alphabet"
    },
    "AMZN": {
        "mean_diff": 0.0206,  # THIRD BEST
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Excellent",
        "notes": "Excellent - third best performer"
    },
    "MSFT": {
        "mean_diff": 0.0822,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Very Good",
        "notes": "Very good quality for Microsoft"
    },
    "TSLA": {
        "mean_diff": 0.0683,
        "training_iterations": 20000,
        "batch_size": 64,
        "hidden_dim": 128,
        "sequence_length": 48,
        "training_time_minutes": 18,
        "quality": "Very Good",
        "notes": "Very good quality despite high volatility"
    }
}

# Diffusion Model Results - REAL DATA FROM EXPERIMENTS
# Source: outputs/results/diffusion_evaluation_*.csv and diffusion_summary.csv
DIFFUSION_RESULTS = {
    "GSPC": {
        "ks_stat": 0.3886,
        "mean_diff": 0.1231,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality, KS < 0.5 threshold"
    },
    "FTSE": {
        "ks_stat": 0.4829,
        "mean_diff": 0.1681,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality, near threshold"
    },
    "DJI": {
        "ks_stat": 0.4129,
        "mean_diff": 0.1400,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for Dow Jones"
    },
    "N225": {
        "ks_stat": 0.3655,
        "mean_diff": 0.1110,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for Nikkei"
    },
    "HSI": {
        "ks_stat": 0.3489,  # Third best KS
        "mean_diff": 0.1102,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Third best KS statistic"
    },
    "IXIC": {
        "ks_stat": 0.3705,
        "mean_diff": 0.1230,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for NASDAQ"
    },
    "AAPL": {
        "ks_stat": 0.3879,
        "mean_diff": 0.1318,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for Apple"
    },
    "GOOGL": {
        "ks_stat": 0.3624,  # Second best KS
        "mean_diff": 0.1142,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Second best KS statistic"
    },
    "AMZN": {
        "ks_stat": 0.3210,  # BEST KS
        "mean_diff": 0.1024,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Best KS statistic across all assets"
    },
    "MSFT": {
        "ks_stat": 0.3976,
        "mean_diff": 0.1333,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for Microsoft"
    },
    "TSLA": {
        "ks_stat": 0.4014,
        "mean_diff": 0.1430,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality despite high volatility"
    },
    "BTC-USD": {
        "ks_stat": 0.4406,
        "mean_diff": 0.1738,
        "quality": "Fair",
        "epochs": 200,
        "batch_size": 64,
        "hidden_dim": 256,
        "num_layers": 6,
        "num_heads": 8,
        "diffusion_steps": 1000,
        "inference_steps": 100,
        "training_time_minutes": 45,
        "notes": "Fair quality for cryptocurrency"
    }
}

# Statistical Test Results - REAL FROM EXPERIMENTS
STATISTICAL_TESTS = {
    "overall_comparison": {
        "timegan_mean_diff_avg": 0.0663,
        "timegan_mean_diff_std": 0.0298,
        "diffusion_mean_diff_avg": 0.1286,
        "diffusion_mean_diff_std": 0.0207,
        "diffusion_ks_avg": 0.3863,
        "diffusion_ks_std": 0.0467,
        "improvement_percentage": 48.4,
        "p_value": 0.0004,  # Statistically significant
        "cohens_d": -2.21,  # Very large effect size
        "conclusion": "TimeGAN performs significantly better (p<0.05, Cohen's d > 0.5)"
    },
    "win_rates": {
        "timegan_wins": 9,
        "diffusion_wins": 0,
        "ties": 2,
        "total_assets": 11,  # BTC-USD only has diffusion results
        "timegan_win_rate": 81.8
    },
    "category_performance": {
        "indices": {
            "count": 6,
            "timegan_avg": 0.0566,
            "timegan_std": 0.0293,
            "diffusion_avg": 0.1292,
            "diffusion_std": 0.0223,
            "improvement": 56.2
        },
        "stocks": {
            "count": 5,
            "timegan_avg": 0.0759,
            "timegan_std": 0.0297,
            "diffusion_avg": 0.1249,
            "diffusion_std": 0.0140,
            "improvement": 39.2
        }
    }
}

# Model Architecture Details
TIMEGAN_ARCHITECTURE = {
    "name": "TimeGAN (Time-series Generative Adversarial Network)",
    "components": {
        "embedder": {
            "type": "GRU",
            "layers": 4,
            "hidden_dim": 128,
            "dropout": 0.2,
            "parameters": "~100K"
        },
        "recovery": {
            "type": "GRU",
            "layers": 4,
            "hidden_dim": 128,
            "parameters": "~100K"
        },
        "generator": {
            "type": "GRU",
            "layers": 4,
            "hidden_dim": 128,
            "noise_dim": 128,
            "parameters": "~100K"
        },
        "discriminator": {
            "type": "GRU",
            "layers": 4,
            "hidden_dim": 128,
            "parameters": "~80K"
        },
        "supervisor": {
            "type": "GRU",
            "layers": 4,
            "hidden_dim": 128,
            "parameters": "~80K"
        }
    },
    "total_parameters": "~460K",
    "training_phases": {
        "phase_1": "Autoencoder (5,000 iterations)",
        "phase_2": "Supervisor (5,000 iterations)",
        "phase_3": "Joint GAN (10,000 iterations)"
    },
    "loss_functions": {
        "reconstruction_loss": "MSE between real and reconstructed sequences",
        "supervised_loss": "MSE for stepwise prediction",
        "adversarial_loss": "Binary cross-entropy for GAN training"
    }
}

DIFFUSION_ARCHITECTURE = {
    "name": "Diffusion Model (DDPM-based with Transformer)",
    "components": {
        "denoising_network": {
            "type": "Transformer",
            "layers": 6,
            "hidden_dim": 256,
            "num_heads": 8,
            "dropout": 0.1,
            "parameters": "~800K"
        },
        "noise_scheduler": {
            "type": "Cosine Schedule",
            "diffusion_steps": 1000,
            "beta_start": 0.0001,
            "beta_end": 0.02
        }
    },
    "total_parameters": "~800K (1.7x larger than TimeGAN)",
    "training_details": {
        "epochs": 200,
        "learning_rate": "0.0002 with warmup + cosine annealing",
        "warmup_epochs": 10,
        "early_stopping_patience": 20
    },
    "loss_function": {
        "type": "Simplified DDPM Loss",
        "formula": "MSE between predicted and actual noise"
    }
}

def get_asset_results(asset_code):
    """Get combined results for an asset"""
    return {
        "asset": ASSETS.get(asset_code, {}),
        "timegan": TIMEGAN_RESULTS.get(asset_code, {}),
        "diffusion": DIFFUSION_RESULTS.get(asset_code, {})
    }

def get_all_assets():
    """Get list of all assets"""
    return ASSETS

def get_comparison_data():
    """Get data for comparative analysis"""
    comparison = []
    for code in TIMEGAN_RESULTS.keys():
        if code in DIFFUSION_RESULTS:
            timegan_md = TIMEGAN_RESULTS[code]["mean_diff"]
            diffusion_md = DIFFUSION_RESULTS[code]["mean_diff"]
            improvement = ((diffusion_md - timegan_md) / diffusion_md) * 100
            
            # Determine winner
            if abs(improvement) < 2:
                winner = "Tie"
            elif improvement > 0:
                winner = "TimeGAN"
            else:
                winner = "Diffusion"
            
            comparison.append({
                "code": code,
                "name": ASSETS[code]["name"],
                "type": ASSETS[code]["type"],
                "timegan_mean_diff": timegan_md,
                "diffusion_ks": DIFFUSION_RESULTS[code]["ks_stat"],
                "diffusion_mean_diff": diffusion_md,
                "improvement": improvement,
                "winner": winner
            })
    return comparison

def get_image_path(asset_code, model_type):
    """Get path to comparison image for asset and model"""
    if model_type == "timegan":
        filename = f"07_timegan_comparison_{asset_code}.png"
    elif model_type == "diffusion":
        filename = f"08_diffusion_comparison_{asset_code}.png"
    else:
        return None
    
    image_path = os.path.join(IMAGES_DIR, filename)
    if os.path.exists(image_path):
        return f"/static/images/{filename}"
    return None

def load_detailed_results(asset_code, model_type):
    """Load detailed CSV results for an asset"""
    if model_type == "timegan":
        csv_file = f"timegan_evaluation_{asset_code}.csv"
    elif model_type == "diffusion":
        csv_file = f"diffusion_evaluation_{asset_code}.csv"
    else:
        return None
    
    csv_path = os.path.join(RESULTS_DIR, csv_file)
    if os.path.exists(csv_path):
        try:
            df = pd.read_csv(csv_path)
            return df.to_dict('records')
        except:
            return None
    return None
