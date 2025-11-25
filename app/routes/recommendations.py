"""
Recommendations Routes
"""

from flask import Blueprint, render_template

recommendations_bp = Blueprint('recommendations', __name__)

@recommendations_bp.route('/')
def index():
    """Recommendations page"""
    return render_template('recommendations/index.html')
