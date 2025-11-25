"""
Model Loading and Inference Utilities
Loads trained TimeGAN and Diffusion models for inference and inspection
"""

import os
import pickle
import numpy as np
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
MODELS_DIR = BASE_DIR / 'models'
TIMEGAN_DIR = MODELS_DIR / 'timegan'
DIFFUSION_DIR = MODELS_DIR / 'diffusion'

def get_available_models():
    """Get list of available trained models"""
    timegan_assets = []
    diffusion_assets = []
    
    if TIMEGAN_DIR.exists():
        timegan_assets = [d.name for d in TIMEGAN_DIR.iterdir() if d.is_dir()]
    
    if DIFFUSION_DIR.exists():
        diffusion_assets = [d.name for d in DIFFUSION_DIR.iterdir() if d.is_dir()]
    
    return {
        'timegan': sorted(timegan_assets),
        'diffusion': sorted(diffusion_assets)
    }

def get_timegan_model_info(asset_code):
    """Get information about TimeGAN model files for an asset"""
    model_dir = TIMEGAN_DIR / asset_code
    
    if not model_dir.exists():
        return None
    
    networks = ['embedder', 'recovery', 'generator', 'supervisor', 'discriminator']
    model_info = {
        'asset': asset_code,
        'model_type': 'TimeGAN',
        'networks': {},
        'total_size_mb': 0
    }
    
    for network in networks:
        model_path = model_dir / f"{network}.h5"
        if model_path.exists():
            size_mb = model_path.stat().st_size / (1024 * 1024)
            model_info['networks'][network] = {
                'file': f"{network}.h5",
                'path': str(model_path),
                'size_mb': round(size_mb, 2),
                'exists': True
            }
            model_info['total_size_mb'] += size_mb
    
    model_info['total_size_mb'] = round(model_info['total_size_mb'], 2)
    return model_info

def get_diffusion_model_info(asset_code):
    """Get information about Diffusion model files for an asset"""
    model_dir = DIFFUSION_DIR / asset_code
    
    if not model_dir.exists():
        return None
    
    model_info = {
        'asset': asset_code,
        'model_type': 'Diffusion',
        'files': {},
        'total_size_mb': 0,
        'scheduler_params': None
    }
    
    # Denoising network
    denoising_path = model_dir / "denoising_network.h5"
    if denoising_path.exists():
        size_mb = denoising_path.stat().st_size / (1024 * 1024)
        model_info['files']['denoising_network'] = {
            'file': 'denoising_network.h5',
            'path': str(denoising_path),
            'size_mb': round(size_mb, 2),
            'exists': True
        }
        model_info['total_size_mb'] += size_mb
    
    # Scheduler params
    scheduler_path = model_dir / "scheduler_params.pkl"
    if scheduler_path.exists():
        size_kb = scheduler_path.stat().st_size / 1024
        model_info['files']['scheduler_params'] = {
            'file': 'scheduler_params.pkl',
            'path': str(scheduler_path),
            'size_kb': round(size_kb, 2),
            'exists': True
        }
        
        # Load scheduler params
        try:
            with open(scheduler_path, 'rb') as f:
                model_info['scheduler_params'] = pickle.load(f)
        except Exception as e:
            model_info['scheduler_params'] = {'error': str(e)}
    
    model_info['total_size_mb'] = round(model_info['total_size_mb'], 2)
    return model_info

def load_timegan_models(asset_code):
    """
    Load all TimeGAN networks for an asset
    Returns dict with loaded Keras models (requires tensorflow)
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {'error': 'TensorFlow not installed. Cannot load models.'}
    
    model_dir = TIMEGAN_DIR / asset_code
    if not model_dir.exists():
        return {'error': f'Model directory not found for {asset_code}'}
    
    networks = ['embedder', 'recovery', 'generator', 'supervisor', 'discriminator']
    loaded_models = {}
    
    for network in networks:
        model_path = model_dir / f"{network}.h5"
        if model_path.exists():
            try:
                loaded_models[network] = tf.keras.models.load_model(str(model_path))
            except Exception as e:
                loaded_models[network] = {'error': str(e)}
    
    return loaded_models

def load_diffusion_model(asset_code):
    """
    Load Diffusion model for an asset
    Returns dict with loaded Keras model and scheduler params
    """
    try:
        import tensorflow as tf
    except ImportError:
        return {'error': 'TensorFlow not installed. Cannot load models.'}
    
    model_dir = DIFFUSION_DIR / asset_code
    if not model_dir.exists():
        return {'error': f'Model directory not found for {asset_code}'}
    
    loaded = {}
    
    # Load denoising network
    denoising_path = model_dir / "denoising_network.h5"
    if denoising_path.exists():
        try:
            loaded['denoising_network'] = tf.keras.models.load_model(str(denoising_path))
        except Exception as e:
            loaded['denoising_network'] = {'error': str(e)}
    
    # Load scheduler params
    scheduler_path = model_dir / "scheduler_params.pkl"
    if scheduler_path.exists():
        try:
            with open(scheduler_path, 'rb') as f:
                loaded['scheduler_params'] = pickle.load(f)
        except Exception as e:
            loaded['scheduler_params'] = {'error': str(e)}
    
    return loaded

def get_model_architecture_summary(model):
    """
    Get architecture summary from a Keras model
    Returns dict with layer info
    """
    try:
        import tensorflow as tf
        
        if not isinstance(model, tf.keras.Model):
            return {'error': 'Not a valid Keras model'}
        
        layers_info = []
        for layer in model.layers:
            layer_info = {
                'name': layer.name,
                'type': layer.__class__.__name__,
                'output_shape': str(layer.output_shape),
                'params': layer.count_params()
            }
            layers_info.append(layer_info)
        
        return {
            'total_params': model.count_params(),
            'trainable_params': sum([layer.count_params() for layer in model.trainable_weights]),
            'num_layers': len(model.layers),
            'layers': layers_info
        }
    except Exception as e:
        return {'error': str(e)}

def generate_synthetic_sample_timegan(asset_code, num_samples=1, sequence_length=48):
    """
    Generate synthetic samples using trained TimeGAN
    Returns numpy array of synthetic sequences
    """
    try:
        import tensorflow as tf
        
        models = load_timegan_models(asset_code)
        if 'error' in models:
            return {'error': models['error']}
        
        if 'generator' not in models or 'recovery' not in models:
            return {'error': 'Generator or Recovery network not found'}
        
        generator = models['generator']
        recovery = models['recovery']
        
        # Generate random noise
        noise_dim = 128  # From your configuration
        noise = np.random.normal(0, 1, (num_samples, sequence_length, noise_dim))
        
        # Generate synthetic latent sequences
        synthetic_latent = generator.predict(noise, verbose=0)
        
        # Recover to data space
        synthetic_data = recovery.predict(synthetic_latent, verbose=0)
        
        return {
            'success': True,
            'synthetic_data': synthetic_data,
            'shape': synthetic_data.shape,
            'num_samples': num_samples,
            'sequence_length': sequence_length
        }
    except Exception as e:
        return {'error': str(e)}

def get_all_models_summary():
    """Get summary of all available models"""
    available = get_available_models()
    
    summary = {
        'timegan': {
            'count': len(available['timegan']),
            'assets': available['timegan'],
            'total_files': len(available['timegan']) * 5,  # 5 networks per asset
            'networks_per_asset': ['embedder', 'recovery', 'generator', 'supervisor', 'discriminator']
        },
        'diffusion': {
            'count': len(available['diffusion']),
            'assets': available['diffusion'],
            'total_files': len(available['diffusion']) * 2,  # 2 files per asset
            'files_per_asset': ['denoising_network.h5', 'scheduler_params.pkl']
        }
    }
    
    # Calculate total sizes
    timegan_total_mb = 0
    for asset in available['timegan']:
        info = get_timegan_model_info(asset)
        if info:
            timegan_total_mb += info['total_size_mb']
    
    diffusion_total_mb = 0
    for asset in available['diffusion']:
        info = get_diffusion_model_info(asset)
        if info:
            diffusion_total_mb += info['total_size_mb']
    
    summary['timegan']['total_size_mb'] = round(timegan_total_mb, 2)
    summary['diffusion']['total_size_mb'] = round(diffusion_total_mb, 2)
    summary['total_size_mb'] = round(timegan_total_mb + diffusion_total_mb, 2)
    
    return summary
