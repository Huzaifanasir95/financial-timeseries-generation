"""
Technical Deep Dive Routes
"""

from flask import Blueprint, render_template

technical_bp = Blueprint('technical', __name__)

@technical_bp.route('/')
def index():
    """Technical deep dive page"""
    return render_template('technical/index.html')
