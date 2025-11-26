"""
Diffusion Model Analysis Routes
"""

from flask import Blueprint, render_template, jsonify
from data import ASSETS, DIFFUSION_RESULTS
import statistics

diffusion_bp = Blueprint('diffusion', __name__)

@diffusion_bp.route('/')
def index():
    """Diffusion analysis page"""
    # Calculate aggregated stats
    ks_stats = [r['ks_stat'] for r in DIFFUSION_RESULTS.values()]
    mean_diffs = [r['mean_diff'] for r in DIFFUSION_RESULTS.values()]
    diffusion_stats = {
        'ks_avg': statistics.mean(ks_stats),
        'ks_std': statistics.stdev(ks_stats),
        'mean_diff_avg': statistics.mean(mean_diffs),
        'mean_diff_std': statistics.stdev(mean_diffs),
        'assets_count': len(ks_stats)
    }
    return render_template('diffusion/index.html', assets=ASSETS, stats=diffusion_stats)

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
