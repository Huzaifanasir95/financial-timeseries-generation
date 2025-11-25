"""
TimeGAN Analysis Routes
"""

from flask import Blueprint, render_template, jsonify
from data import ASSETS, TIMEGAN_RESULTS, get_asset_results

timegan_bp = Blueprint('timegan', __name__)

@timegan_bp.route('/')
def index():
    """TimeGAN analysis page"""
    return render_template('timegan/index.html', assets=ASSETS)

@timegan_bp.route('/api/results')
def api_results():
    """Get all TimeGAN results"""
    return jsonify(TIMEGAN_RESULTS)

@timegan_bp.route('/api/asset/<asset_code>')
def api_asset(asset_code):
    """Get specific asset TimeGAN results"""
    if asset_code in TIMEGAN_RESULTS:
        return jsonify({
            "asset": ASSETS[asset_code],
            "results": TIMEGAN_RESULTS[asset_code]
        })
    return jsonify({"error": "Asset not found"}), 404
