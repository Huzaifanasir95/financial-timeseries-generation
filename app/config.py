"""
Flask Configuration
"""

import os

class Config:
    """Base configuration"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    
    # Application settings
    APP_NAME = "Financial Timeseries Generation"
    APP_VERSION = "1.0.0"
    
    # Image paths
    IMAGES_FOLDER = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'Final-Report', 'images')
    
    # Color scheme - Professional and readable
    COLORS = {
        "primary": "#2563eb",      # Blue
        "success": "#10b981",      # Emerald green
        "warning": "#f59e0b",      # Amber
        "danger": "#ef4444",       # Red
        "dark": "#1e293b",         # Slate dark
        "light": "#f8fafc",        # Slate light
        "accent": "#06b6d4",       # Cyan
        "text_dark": "#0f172a",
        "text_light": "#475569",
        "bg_card": "#ffffff",
    }
    
    # Chart colors
    CHART_COLORS = {
        "timegan": "#2563eb",
        "diffusion": "#06b6d4",
        "real": "#10b981",
    }

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    TESTING = False

class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False
    TESTING = False

# Config dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}
