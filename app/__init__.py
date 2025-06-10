from flask import Flask
import logging
from pymongo import MongoClient

mongo_client = None
db = None

def create_app(config_object):
    app = Flask(__name__)
    app.config.from_object(config_object)

    logging.basicConfig(level=logging.INFO)
    app.logger.setLevel(logging.INFO)

    global mongo_client, db
    try:
        mongo_client = MongoClient(app.config['MONGO_URI'])
        db = mongo_client.truthguard
        app.logger.info("Successfully connected to MongoDB!")
    except Exception as e:
        app.logger.error(f"Failed to connect to MongoDB: {e}")

    from .routes.main import main_bp
    app.register_blueprint(main_bp)

    @app.route('/')
    def index():
        return "TruthGuard Backend is running!"

    return app