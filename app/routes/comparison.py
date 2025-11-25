"""
Comparative Analysis Routes
"""

from flask import Blueprint, render_template, jsonify
from data import get_comparison_data, ASSETS, TIMEGAN_RESULTS, DIFFUSION_RESULTS

comparison_bp = Blueprint('comparison', __name__)

@comparison_bp.route('/')
def index():
    """Comparison analysis page"""
    comparison_data = get_comparison_data()
    return render_template('comparison/index.html', data=comparison_data)

@comparison_bp.route('/api/comparison')
def api_comparison():
    """Get comparison data"""
    return jsonify(get_comparison_data())

@comparison_bp.route('/api/asset/<asset_code>')
def api_asset_comparison(asset_code):
    """Get comparison for specific asset"""
    if asset_code in TIMEGAN_RESULTS and asset_code in DIFFUSION_RESULTS:
        return jsonify({
            "asset": ASSETS[asset_code],
            "timegan": TIMEGAN_RESULTS[asset_code],
            "diffusion": DIFFUSION_RESULTS[asset_code],
            "improvement": {
                "ks": ((DIFFUSION_RESULTS[asset_code]["ks"] - TIMEGAN_RESULTS[asset_code]["ks"]) / DIFFUSION_RESULTS[asset_code]["ks"]) * 100,
                "mean_diff": ((DIFFUSION_RESULTS[asset_code]["mean_diff"] - TIMEGAN_RESULTS[asset_code]["mean_diff"]) / DIFFUSION_RESULTS[asset_code]["mean_diff"]) * 100,
                "std_diff": ((DIFFUSION_RESULTS[asset_code]["std_diff"] - TIMEGAN_RESULTS[asset_code]["std_diff"]) / DIFFUSION_RESULTS[asset_code]["std_diff"]) * 100
            }
        })
    return jsonify({"error": "Asset not found"}), 404
