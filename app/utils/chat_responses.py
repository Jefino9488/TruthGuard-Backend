

def format_initial_greeting():
    """Format the initial greeting message."""
    return """**🤖 Welcome to TruthGuard AI!**

I'm powered by advanced AI and MongoDB Vector Search. Here's how I can help:

**🔍 Content Analysis:**
- "Analyze this headline: [your text]"
- Real-time bias and misinformation detection
- Sentiment and credibility analysis

**🔎 Database Search:**
- "Search for articles about [topic]"
- Vector-based semantic search
- Find similar content patterns

**📊 Educational Insights:**
- Ask about bias patterns and detection methods
- Learn about misinformation techniques
- Understand media credibility factors

**💾 MongoDB Features:**
- All analysis stored with vector embeddings
- Real-time aggregation and analytics
- Semantic search across content
- Pattern recognition and trends

**🚀 Try These Examples:**
- "Analyze this headline: Breaking news shocks the world"
- "What are common bias patterns?"
- "How do you detect misinformation?"

What would you like to explore first?"""

def format_chat_response(analysis_result):
    """Format the analysis result into a user-friendly response."""
    if not analysis_result or 'analysis' not in analysis_result:
        return "I couldn't analyze that content. Please try again with a longer text or a valid URL."

    analysis = analysis_result['analysis']

    # Check if analysis contains all required fields
    required_fields = [
        'bias_analysis', 'misinformation_analysis', 
        'credibility_assessment', 'confidence',
        'technical_analysis', 'narrative_analysis'
    ]

    for field in required_fields:
        if field not in analysis:
            return f"I was unable to complete the analysis. Some required information was missing. Please try again with more detailed content."

    try:
        # Extract key metrics
        bias_score = analysis['bias_analysis']['overall_score'] * 100
        misinfo_risk = analysis['misinformation_analysis']['risk_score'] * 100
        credibility = analysis['credibility_assessment']['overall_score'] * 100
        confidence = analysis['confidence'] * 100
    except (KeyError, TypeError) as e:
        # If any key is missing or there's a type error, return a helpful message
        return "I encountered an issue while analyzing this content. Please try again with more detailed text or a different query."

    # Safely get nested values with defaults
    def safe_get(d, keys, default=''):
        """Safely get a value from nested dictionaries."""
        try:
            result = d
            for key in keys:
                result = result[key]
            if isinstance(result, list) and not result:
                return default
            return result
        except (KeyError, TypeError, IndexError):
            return default

    # Get values safely
    key_topic = safe_get(analysis, ['technical_analysis', 'key_topics', 0], 'General content')
    primary_frame = safe_get(analysis, ['narrative_analysis', 'primary_frame'], 'General analysis')
    political_leaning = safe_get(analysis, ['bias_analysis', 'political_leaning'], 'Neutral')
    emotional_tone = safe_get(analysis, ['sentiment_analysis', 'emotional_tone'], 'Neutral')

    # Get narrative patterns
    narrative_patterns = safe_get(analysis, ['narrative_analysis', 'narrative_patterns'], [])
    patterns_text = ', '.join(narrative_patterns[:2]) if narrative_patterns else 'No specific patterns detected'

    # Get recommendations
    recommendations = safe_get(analysis, ['recommendations', 'critical_questions'], [])
    recommendation_text = recommendations[0] if recommendations else 'Consider checking multiple sources for verification'

    # Get model version
    model_version = safe_get(analysis, ['model_version'], 'TruthGuard AI')

    # Format the response
    response = f"""**Analysis Results:**

**Content Overview:**
{key_topic} - {primary_frame}

**Key Findings:**
- Bias: {bias_score:.1f}% ({political_leaning})
- Misinformation Risk: {misinfo_risk:.1f}%
- Credibility Score: {credibility:.1f}%
- Sentiment: {emotional_tone}

**Notable Patterns:**
{patterns_text}

**Recommendations:**
{recommendation_text}

Real-time Analysis Results:
Bias Score: {bias_score:.1f}%
Misinfo Risk: {misinfo_risk:.1f}%
Confidence: {confidence:.1f}%
Model: {model_version}
Stored: Yes ✅"""

    return response

def format_search_response(search_results):
    """Format search results into a user-friendly response."""
    if not search_results or len(search_results) == 0:
        return "I couldn't find any relevant articles in our database. Try a different search term or ask me to analyze specific content."

    response = "**Search Results:**\n\n"

    for i, result in enumerate(search_results[:3], 1):
        title = result.get('title', 'Untitled Article')
        source = result.get('source', {}).get('name', 'Unknown Source')
        published_date = result.get('published_at', 'Unknown Date')

        # Format date if available
        if isinstance(published_date, str):
            try:
                from datetime import datetime
                date_obj = datetime.fromisoformat(published_date.replace('Z', '+00:00'))
                published_date = date_obj.strftime('%B %d, %Y')
            except:
                pass

        # Get analysis if available
        analysis = result.get('analysis', {})
        bias_score = analysis.get('bias_analysis', {}).get('overall_score', 0) * 100
        misinfo_risk = analysis.get('misinformation_analysis', {}).get('risk_score', 0) * 100

        response += f"**{i}. {title}**\n"
        response += f"Source: {source} | Published: {published_date}\n"

        if analysis:
            response += f"Bias: {bias_score:.1f}% | Misinfo Risk: {misinfo_risk:.1f}%\n"

        response += "\n"

    response += "Would you like me to analyze any of these articles in detail?"

    return response

def format_continuation_response(previous_message, user_query):
    """Format a response that continues the conversation based on previous context."""
    # Check if the query is asking for continuation or more details
    continuation_phrases = ["tell me more", "continue", "go on", "elaborate", "explain further", 
                           "more details", "what else", "and then", "next", "more info"]

    is_continuation = any(phrase in user_query.lower() for phrase in continuation_phrases)

    if not is_continuation and len(user_query) < 10:
        # For very short queries that aren't explicit continuation requests
        return f"""I'd be happy to continue our conversation about the previous topic or analyze "{user_query}".

To continue our previous discussion, just say "continue" or "tell me more".
To analyze "{user_query}" as a new topic, please provide more context or a longer query.

What would you like me to do?"""

    if is_continuation:
        return "I'll continue with more details about our previous topic. What specific aspect would you like me to elaborate on?"

    # Default response for other cases
    return None  # Return None to let the regular analysis flow handle it
