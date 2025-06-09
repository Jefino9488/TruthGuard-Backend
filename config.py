# flask_backend/config.py

import os
from dotenv import load_dotenv

# Load environment variables from .env.local
load_dotenv('.env')

class Config:
    """Base configuration."""
    SECRET_KEY = os.getenv('SECRET_KEY', 'a_very_secret_key_for_dev') # Replace with a strong secret key in production
    MONGO_URI = os.getenv('MONGODB_URI')
    NEWS_API_KEY = os.getenv('NEWS_API_KEY_SCRAPER') # Renamed from NEWS_API_KEY for clarity
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    FLASK_ENV = os.getenv('FLASK_ENV', 'development')
    FLASK_DEBUG = os.getenv('FLASK_DEBUG', 'False').lower() == 'true'
    SCRAPE_INTERVAL_HOURS = int(os.getenv('SCRAPE_INTERVAL_HOURS', 4))
    BATCH_SIZE_ANALYSIS = int(os.getenv('BATCH_SIZE_ANALYSIS', 10))

    if not MONGO_URI:
        raise ValueError("No MONGO_URI provided. Set MONGODB_URI env variable.")
    if not NEWS_API_KEY:
        raise ValueError("No NEWS_API_KEY provided. Set NEWS_API_KEY_SCRAPER env variable.")
    if not GOOGLE_API_KEY:
        raise ValueError("No GOOGLE_API_KEY provided. Set GOOGLE_API_KEY env variable.")

class DevelopmentConfig(Config):
    """Development configuration."""
    DEBUG = True

class ProductionConfig(Config):
    """Production configuration."""
    DEBUG = False
    FLASK_ENV = 'production'
    # Add any production-specific settings here

def get_config():
    env = os.getenv('FLASK_ENV', 'development')
    if env == 'production':
        return ProductionConfig()
    return DevelopmentConfig()