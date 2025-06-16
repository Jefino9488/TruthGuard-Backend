from flask import Flask
import logging
from pymongo import MongoClient
from pymongo.errors import ConnectionFailure

# Initialize these at module level
mongo_client = None
db = None

def init_db(app):
    """Initialize database connection"""
    global mongo_client, db
    try:
        mongo_client = MongoClient(app.config['MONGO_URI'])
        # Test the connection
        mongo_client.admin.command('ping')
        db = mongo_client.truthguard
        app.logger.info("Successfully connected to MongoDB!")
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
    app.config.from_object(config_object)

    # Configure logging
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Initialize database
    if not init_db(app):
        app.logger.error("Failed to initialize database. Application may not function correctly.")

    # Register blueprints
    from .routes.main import main_bp, init_route_dependencies
    app.register_blueprint(main_bp)

    # Initialize route dependencies
    with app.app_context():
        init_route_dependencies(app)

    @app.route('/')
    def index():
        return "TruthGuard Backend is running!"

    return app

__all__ = ['db', 'create_app']
