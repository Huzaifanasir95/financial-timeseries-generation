"""
Diffusion Model Analysis Routes
"""

from flask import Blueprint, render_template, jsonify
from data import ASSETS, DIFFUSION_RESULTS

diffusion_bp = Blueprint('diffusion', __name__)

@diffusion_bp.route('/')
def index():
    """Diffusion analysis page"""
    return render_template('diffusion/index.html', assets=ASSETS)

@diffusion_bp.route('/api/results')
def api_results():
    """Get all Diffusion results"""
    return jsonify(DIFFUSION_RESULTS)

@diffusion_bp.route('/api/asset/<asset_code>')
def api_asset(asset_code):
    """Get specific asset Diffusion results"""
    if asset_code in DIFFUSION_RESULTS:
        return jsonify({
            "asset": ASSETS[asset_code],
            "results": DIFFUSION_RESULTS[asset_code]
        })
    return jsonify({"error": "Asset not found"}), 404
