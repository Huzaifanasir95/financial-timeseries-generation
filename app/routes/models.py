"""
Model Inspection and Download Routes
"""

from flask import Blueprint, render_template, jsonify, send_file
from model_utils import (
    get_available_models,
    get_timegan_model_info,
    get_diffusion_model_info,
    get_all_models_summary
)
from data import ASSETS
import os

models_bp = Blueprint('models', __name__)

@models_bp.route('/')
def index():
    """Models overview page"""
    summary = get_all_models_summary()
    return render_template('models/index.html', 
                         summary=summary,
                         assets=ASSETS)

@models_bp.route('/api/summary')
def api_summary():
    """Get summary of all models"""
    return jsonify(get_all_models_summary())

@models_bp.route('/api/timegan/<asset_code>')
def api_timegan_info(asset_code):
    """Get TimeGAN model info for specific asset"""
    info = get_timegan_model_info(asset_code)
    if info is None:
        return jsonify({'error': 'Model not found'}), 404
    return jsonify(info)

@models_bp.route('/api/diffusion/<asset_code>')
def api_diffusion_info(asset_code):
    """Get Diffusion model info for specific asset"""
    info = get_diffusion_model_info(asset_code)
    if info is None:
        return jsonify({'error': 'Model not found'}), 404
    return jsonify(info)

@models_bp.route('/download/timegan/<asset_code>/<network>')
def download_timegan_network(asset_code, network):
    """Download specific TimeGAN network"""
    valid_networks = ['embedder', 'recovery', 'generator', 'supervisor', 'discriminator']
    if network not in valid_networks:
        return jsonify({'error': 'Invalid network name'}), 400
    
    model_path = os.path.join('models', 'timegan', asset_code, f'{network}.h5')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    return send_file(model_path, 
                    as_attachment=True,
                    download_name=f'timegan_{asset_code}_{network}.h5')

@models_bp.route('/download/diffusion/<asset_code>/denoising')
def download_diffusion_denoising(asset_code):
    """Download Diffusion denoising network"""
    model_path = os.path.join('models', 'diffusion', asset_code, 'denoising_network.h5')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Model file not found'}), 404
    
    return send_file(model_path,
                    as_attachment=True,
                    download_name=f'diffusion_{asset_code}_denoising.h5')

@models_bp.route('/download/diffusion/<asset_code>/scheduler')
def download_diffusion_scheduler(asset_code):
    """Download Diffusion scheduler params"""
    model_path = os.path.join('models', 'diffusion', asset_code, 'scheduler_params.pkl')
    if not os.path.exists(model_path):
        return jsonify({'error': 'Scheduler params not found'}), 404
    
    return send_file(model_path,
                    as_attachment=True,
                    download_name=f'diffusion_{asset_code}_scheduler.pkl')
