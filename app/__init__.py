# flask_backend/app/__init__.py

from flask import Flask
import logging
from pymongo import MongoClient

# Initialize MongoDB client globally (or per request, but for simplicity, global for now)
mongo_client = None
db = None

def create_app(config_object):
    """
    Factory function to create and configure the Flask application.
    """
    app = Flask(__name__)
    app.config.from_object(config_object)

    # Configure logging for the Flask app
    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    # Initialize MongoDB
    global mongo_client, db
    try:
        mongo_client = MongoClient(app.config['MONGO_URI'])
        db = mongo_client.truthguard # 'truthguard' is your database name
        app.logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB: {e}")
        # Depending on your deployment, you might want to exit or handle gracefully
        # sys.exit(1) # This would exit the application if DB connection fails at startup

    # Register blueprints (routes will be added here later)
    from .routes.main import main_bp
    app.register_blueprint(main_bp)

    # Add a simple route to check if the app is running
    @app.route('/')
    def index():
        return "TruthGuard Backend is running!"

    return app