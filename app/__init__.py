from flask import Flask
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure
from typing import Optional

# Initialize these at module level
mongo_client: Optional[MongoClient] = None
db = None

def init_db(app) -> bool:
    """Initialize database connection with proper error handling"""
    global mongo_client, db
    try:
        if not app.config.get('MONGO_URI'):
            app.logger.error("MONGO_URI configuration is missing")
            return False

        mongo_client = MongoClient(app.config['MONGO_URI'])
        # Test the connection explicitly
        mongo_client.admin.command('ping')
        db = mongo_client.truthguard
        app.logger.info("Successfully connected to MongoDB")
        return True

    except ConnectionFailure as e:
        app.logger.error(f"Failed to connect to MongoDB: {e}")
        return False
    except Exception as e:
        app.logger.error(f"Unexpected error connecting to MongoDB: {e}")
        return False

def create_app(config_object):
    """Create and configure the Flask application"""
    app = Flask(__name__)

    # Configure logging first
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Load configuration
    try:
        app.config.from_object(config_object)
    except Exception as e:
        app.logger.error(f"Failed to load configuration: {e}")
        raise

    # Initialize database
    if not init_db(app):
        app.logger.error("Failed to initialize database. Application may not function correctly.")

    # Register blueprints and initialize routes
    try:
        from .routes.main import main_bp, init_route_dependencies
        app.register_blueprint(main_bp)

        # Initialize route dependencies within app context
        with app.app_context():
            init_route_dependencies(app)

    except Exception as e:
        app.logger.error(f"Failed to initialize application routes: {e}")
        raise

    @app.route('/')
    def index():
        return "TruthGuard Backend is running!"

    return app

# Clean up resources when the application exits
def cleanup():
    global mongo_client
    if mongo_client is not None:
        try:
            mongo_client.close()
        except Exception as e:
            logging.error(f"Error closing MongoDB connection: {e}")

import atexit
atexit.register(cleanup)

__all__ = ['db', 'create_app']
