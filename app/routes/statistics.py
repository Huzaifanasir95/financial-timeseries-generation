"""
Statistical Analysis Routes
"""

from flask import Blueprint, render_template, jsonify
from data import STATISTICAL_TESTS

statistics_bp = Blueprint('statistics', __name__)

@statistics_bp.route('/')
def index():
    """Statistical analysis page"""
    return render_template('statistics/index.html', stats=STATISTICAL_TESTS)

@statistics_bp.route('/api/tests')
def api_tests():
    """Get all statistical tests"""
    return jsonify(STATISTICAL_TESTS)
