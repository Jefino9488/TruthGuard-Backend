from functools import wraps
from flask import request, jsonify, current_app

def require_api_key(f):
    """Decorator to protect routes with an API key."""
    @wraps(f)
    def decorated_function(*args, **kwargs):
        api_key = request.headers.get('X-API-KEY')
        if not api_key or api_key != current_app.config.get('SECRET_API_KEY'):
            return jsonify({"error": "Unauthorized: Invalid or missing API key"}), 401
        return f(*args, **kwargs)
    return decorated_function
