# flask_backend/run.py

from app import create_app
from config import get_config
import os

# Get configuration based on environment
config = get_config()

# Create the Flask app
app = create_app(config)

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)