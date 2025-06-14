import os
from dotenv import load_dotenv

# Load .env file only in development environment
if os.environ.get('FLASK_ENV') != 'production':
    load_dotenv()

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'a_very_secret_key_for_dev')
    MONGO_URI = os.environ.get('MONGODB_URI')
    NEWS_API_KEY = os.environ.get('NEWS_API_KEY_SCRAPER')
    GOOGLE_API_KEY = os.environ.get('GOOGLE_API_KEY')
    FLASK_ENV = os.environ.get('FLASK_ENV', 'development')
    FLASK_DEBUG = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    SCRAPE_INTERVAL_HOURS = int(os.environ.get('SCRAPE_INTERVAL_HOURS', 4))
    BATCH_SIZE_ANALYSIS = int(os.environ.get('BATCH_SIZE_ANALYSIS', 10))

    if not MONGO_URI:
        raise ValueError("No MONGODB_URI provided in environment variables")
    if not NEWS_API_KEY:
        raise ValueError("No NEWS_API_KEY_SCRAPER provided in environment variables")
    if not GOOGLE_API_KEY:
        raise ValueError("No GOOGLE_API_KEY provided in environment variables")

class DevelopmentConfig(Config):
    DEBUG = True

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'

def get_config():
    env = os.environ.get('FLASK_ENV', 'development')
    if env == 'development':
        return DevelopmentConfig
    return ProductionConfig
