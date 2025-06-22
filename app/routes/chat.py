from flask import Blueprint, request, jsonify, current_app, session
from ..tasks.analyzer import GeminiAnalyzerTask
from ..utils.chat_responses import format_chat_response, format_initial_greeting, format_search_response, format_continuation_response
import re
from urllib.parse import urlparse

chat_bp = Blueprint('chat', __name__)

def is_url(text: str) -> bool:
    """Check if the text is a URL."""
    try:
        result = urlparse(text)
        return all([result.scheme, result.netloc])
    except ValueError:
        return False

def extract_url(text: str) -> str:
    """Extract URL from text if present."""
    urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', text)
    return urls[0] if urls else None

@chat_bp.route('/chat', methods=['POST'])
def chat_endpoint():
    """Handle chat messages and return appropriate responses."""
    try:
        data = request.json
        message = data.get('message', '').strip()

        # Get session ID from request or create a new one
        session_id = data.get('session_id', request.cookies.get('session_id', None))

        # Store the previous message for context
        try:
            previous_message = session.get('previous_message', '')
        except:
            # If session is not available, use an empty string
            previous_message = ''

        if not message:
            # Reset previous message for new conversations
            try:
                session['previous_message'] = ''
            except:
                pass
            return jsonify({
                "response": format_initial_greeting(),
                "analysis": None
            })

        # Initialize analyzer
        analyzer = GeminiAnalyzerTask(
            current_app.config['MONGO_DB'],
            current_app.config['GOOGLE_API_KEY']
        )

        # Extract URL if present
        url = extract_url(message) if is_url(message) else None

        # Check if this is a short query that might be a continuation request
        if not url and len(message) < 20:
            # Try to use the continuation response
            cont_response = format_continuation_response(previous_message, message)
            if cont_response:
                # Store current message as previous for next request
                try:
                    session['previous_message'] = message
                except:
                    pass
                return jsonify({
                    "response": cont_response,
                    "analysis": None
                })

        # Perform analysis if content is provided
        analysis_result = analyzer.analyze_raw_content(
            title=url if url else message[:100],  # Use URL as title if present
            content=message
        )

        # Format response based on analysis results
        chat_response = format_chat_response(analysis_result)

        # Store current message as previous for next request
        try:
            session['previous_message'] = message
        except:
            pass

        return jsonify({
            "response": chat_response,
            "analysis": analysis_result
        })

    except Exception as e:
        current_app.logger.error(f"Error in chat endpoint: {e}")
        return jsonify({
            "response": "I apologize, but I encountered an error processing your request. Please try again.",
            "error": str(e)
        }), 500
