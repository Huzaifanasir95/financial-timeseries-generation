"""
Flask Application for Financial Timeseries Generation Analysis
TimeGAN vs Diffusion Models Comparative Study
"""

from flask import Flask, render_template, jsonify
from config import Config
from routes import timegan_bp, diffusion_bp, comparison_bp, statistics_bp, technical_bp, recommendations_bp
from routes.models import models_bp
from model_server import model_server_bp, model_server
from data import STATISTICAL_TESTS
import os

def create_app(config_class=Config):
    app = Flask(__name__)
    app.config.from_object(config_class)
    
    # Register blueprints
    app.register_blueprint(timegan_bp, url_prefix='/timegan')
    app.register_blueprint(diffusion_bp, url_prefix='/diffusion')
    app.register_blueprint(comparison_bp, url_prefix='/comparison')
    app.register_blueprint(statistics_bp, url_prefix='/statistics')
    app.register_blueprint(technical_bp, url_prefix='/technical')
    app.register_blueprint(recommendations_bp, url_prefix='/recommendations')
    app.register_blueprint(models_bp, url_prefix='/models')
    app.register_blueprint(model_server_bp)  # Model serving endpoints
    
    # Initialize model server on startup
    with app.app_context():
        model_server.load_models()
    
    @app.route('/')
    def index():
        """Executive Summary / Home Page"""
        return render_template('index.html', stats=STATISTICAL_TESTS)
    
    @app.route('/api/health')
    def health():
        """Health check endpoint"""
        return jsonify({"status": "healthy", "app": "Financial Timeseries Analysis"})
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(debug=True, host='0.0.0.0', port=5000)
