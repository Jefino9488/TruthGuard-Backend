from datetime import datetime, timezone
from flask import Blueprint, request, jsonify, current_app
import threading
import hashlib
from newspaper import Article
from pymongo.errors import PyMongoError
import importlib
from typing import Optional

# Initialize the blueprint
main_bp = Blueprint('main', __name__)

# Initialize variables that will be set up during init_route_dependencies
article_service: Optional[object] = None  # Will be initialized as ArticleService
db: Optional[object] = None  # Will be initialized as pymongo.database.Database

def get_task_classes():
    """Dynamically import task classes to avoid circular imports"""
    tasks_module = importlib.import_module('app.tasks')
    return (
        getattr(tasks_module, 'NewsAPIFetcherTask'),
        getattr(tasks_module, 'GeminiAnalyzerTask')
    )

def init_route_dependencies(app):
    """Initialize dependencies after app context is created"""
    global article_service, db

    # Import and initialize database connection
    app_module = importlib.import_module('app')
    db = getattr(app_module, 'db')

    if not db:
        app.logger.error("Database connection not initialized")
        raise RuntimeError("Database connection not initialized")

    try:
        # Test database connection
        db.admin.command('ping')
    except Exception as e:
        app.logger.error(f"Failed to connect to database: {e}")
        raise RuntimeError(f"Database connection failed: {e}")

    # Import and initialize ArticleService
    try:
        ArticleService = importlib.import_module('app.services.article_service').ArticleService
        article_service = ArticleService(db)
    except Exception as e:
        app.logger.error(f"Failed to initialize ArticleService: {e}")
        raise

@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    try:
        # Check MongoDB connection by using the global db object
        if not db:
            raise RuntimeError("Database not initialized")
        db.admin.command('ping')
        mongo_status = "connected"
    except Exception as e:
        mongo_status = f"disconnected: {e}"

    try:
        # Check NewsAPI key (just check for presence, not validation)
        news_api_key_status = "present" if current_app.config['NEWS_API_KEY'] else "missing"
    except Exception:
        news_api_key_status = "missing"

    try:
        # Check Google AI API key (just check for presence)
        google_api_key_status = "present" if current_app.config['GOOGLE_API_KEY'] else "missing"
    except Exception:
        google_api_key_status = "missing"

    return jsonify({
        "status": "ok",
        "message": "TruthGuard Backend is healthy!",
        "dependencies": {
            "mongodb": mongo_status,
            "news_api_key": news_api_key_status,
            "google_ai_key": google_api_key_status
        },
        "timestamp": datetime.now(timezone.utc).isoformat()
    }), 200

@main_bp.route('/scrape', methods=['POST'])
def trigger_scrape():
    """
    API endpoint to trigger the news scraping task.
    """
    try:
        news_api_key = current_app.config['NEWS_API_KEY']
        NewsAPIFetcherTask = get_task_classes()[0]
        scraper = NewsAPIFetcherTask(db, news_api_key)

        thread = threading.Thread(target=scraper.run_scraper)
        thread.daemon = True # Allow main program to exit even if thread is running
        thread.start()

        return jsonify({"message": "News scraping initiated successfully!", "status": "processing"}), 202
    except Exception as e:
        current_app.logger.error(f"Error triggering scraping: {e}")
        return jsonify({"error": "Failed to initiate scraping task", "details": str(e)}), 500

@main_bp.route('/analyze', methods=['POST'])
def trigger_analysis():
    """
    API endpoint to trigger the AI analysis task.
    """
    try:
        google_api_key = current_app.config['GOOGLE_API_KEY']
        batch_size = current_app.config['BATCH_SIZE_ANALYSIS']
        GeminiAnalyzerTask = get_task_classes()[1]
        analyzer = GeminiAnalyzerTask(db, google_api_key)

        thread = threading.Thread(target=analyzer.run_analyzer, args=(batch_size,))
        thread.daemon = True # Allow main program to exit even if thread is running
        thread.start()

        return jsonify({"message": "AI analysis initiated successfully!", "status": "processing"}), 202
    except Exception as e:
        current_app.logger.error(f"Error triggering analysis: {e}")
        return jsonify({"error": "Failed to initiate analysis task", "details": str(e)}), 500

@main_bp.route('/analyze-manual', methods=['POST'])
def manual_analysis():
    """
    API endpoint to manually analyze a given headline, content, or URL.
    Expects JSON body: {"headline": "...", "content": "..."} OR {"url": "..."}
    """
    data = request.get_json()
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    headline = data.get('headline')
    content = data.get('content')
    url = data.get('url')

    article_to_analyze = {}

    try:
        if url:
            current_app.logger.info(f"Manual analysis: Scraping content from URL: {url}")
            temp_article = Article(url)
            temp_article.download()
            temp_article.parse()

            scraped_content = temp_article.text or temp_article.meta_description or temp_article.title
            scraped_title = temp_article.title

            # More robust source extraction
            scraped_source = "Unknown"
            if hasattr(temp_article, 'meta_site_name') and temp_article.meta_site_name:
                scraped_source = temp_article.meta_site_name
            elif temp_article.url:
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(temp_article.url)
                    scraped_source = parsed_url.netloc.replace('www.', '')
                except Exception:
                    pass

            if not scraped_content or len(scraped_content) < 50:
                return jsonify({"error": "Could not extract sufficient content from the provided URL (min 50 characters required)."}), 400

            article_to_analyze = {
                "article_id": hashlib.md5(url.encode()).hexdigest(),
                "title": scraped_title,
                "url": url,
                "source": scraped_source,
                "content": scraped_content,
                "published_at": datetime.now(timezone.utc).isoformat()
            }

        elif headline and content:
            current_app.logger.info(f"Manual analysis: Analyzing provided headline and content.")
            article_to_analyze = {
                "article_id": hashlib.md5((headline + content).encode()).hexdigest(),
                "title": headline,
                "content": content,
                "source": "Manual Input",
                "published_at": datetime.now(timezone.utc).isoformat()
            }

        else:
            return jsonify({"error": "Please provide either 'url' OR 'headline' and 'content' for manual analysis."}), 400

        google_api_key = current_app.config['GOOGLE_API_KEY']
        _, GeminiAnalyzerTask = get_task_classes()
        analyzer = GeminiAnalyzerTask(db, google_api_key)

        # Call the new raw content analysis method that returns embeddings
        analysis_result = analyzer.analyze_raw_content(
            title=article_to_analyze.get("title"),
            content=article_to_analyze.get("content")
        )

        if analysis_result:
            return jsonify({
                "success": True,
                "analysis": analysis_result, # This now includes embeddings as well
                "article_meta": {
                    "title": article_to_analyze.get("title"),
                    "source": article_to_analyze.get("source"),
                    "url": article_to_analyze.get("url"),
                    "published_at": article_to_analyze.get("published_at")
                }
            }), 200
        else:
            return jsonify({"success": False, "error": "Failed to perform manual analysis."}), 500

    except Exception as e:
        current_app.logger.error(f"Error during manual analysis: {e}", exc_info=True)
        return jsonify({"success": False, "error": "An unexpected error occurred during manual analysis.", "details": str(e)}), 500

@main_bp.route('/articles', methods=['GET'])
def get_articles():
    """
    API endpoint to retrieve a paginated list of articles.
    Query parameters: page, limit, sort_by, sort_order
    """
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    sort_by = request.args.get('sort_by', 'published_at', type=str)
    sort_order = request.args.get('sort_order', 'desc', type=str)

    # ArticleService is initialized globally in app/__init__.py for simplicity
    result = article_service.get_all_articles(page, limit, sort_by, sort_order)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@main_bp.route('/articles/<article_id>', methods=['GET'])
def get_article_detail(article_id):
    """
    API endpoint to retrieve details of a single article by its ID.
    """
    article = article_service.get_article_by_id(article_id)
    if article:
        return jsonify(article), 200
    return jsonify({"message": "Article not found"}), 404

@main_bp.route('/articles/search', methods=['GET'])
def search_articles_endpoint():
    """
    API endpoint to search articles.
    Query parameters: q (query string), page, limit, sort_by, sort_order
    """
    query = request.args.get('q', type=str)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    sort_by = request.args.get('sort_by', 'score', type=str)
    sort_order = request.args.get('sort_order', 'desc', type=str)

    if not query:
        return jsonify({"error": "Query parameter 'q' is required for search."}), 400

    result = article_service.search_articles(query, page, limit, sort_by, sort_order)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@main_bp.route('/articles/high-bias', methods=['GET'])
def get_high_bias_articles():
    """
    API endpoint to retrieve articles flagged with high bias.
    Query parameters: min_score, page, limit, sort_order
    """
    min_score = request.args.get('min_score', 0.7, type=float)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    sort_order = request.args.get('sort_order', 'desc', type=str)

    result = article_service.get_articles_by_bias_score(min_score, page, limit, sort_order)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@main_bp.route('/articles/misinformation-risk', methods=['GET'])
def get_misinformation_risk_articles():
    """
    API endpoint to retrieve articles flagged with high misinformation risk.
    Query parameters: min_risk, page, limit, sort_order
    """
    min_risk = request.args.get('min_risk', 0.6, type=float)
    page = request.args.get('page', 1, type=int)
    limit = request.args.get('limit', 10, type=int)
    sort_order = request.args.get('sort_order', 'desc', type=str)

    result = article_service.get_articles_by_misinformation_risk(min_risk, page, limit, sort_order)
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200

@main_bp.route('/dashboard-analytics', methods=['GET'])
def get_dashboard_analytics():
    """
    API endpoint to retrieve aggregated analytics data for the dashboard.
    """
    try:
        analytics_data = article_service.get_dashboard_analytics()
        if "error" in analytics_data:
            return jsonify({"success": False, "error": analytics_data["error"]}), 500
        return jsonify({"success": True, "data": analytics_data}), 200
    except PyMongoError as e:
        current_app.logger.error(f"MongoDB error fetching dashboard analytics: {e}")
        return jsonify({"success": False, "error": "Database error fetching analytics", "details": str(e)}), 500
    except Exception as e:
        current_app.logger.error(f"Error fetching dashboard analytics: {e}", exc_info=True)
        return jsonify({"success": False, "error": "An unexpected error occurred while fetching analytics", "details": str(e)}), 500


@main_bp.route('/vector-search', methods=['POST'])
def vector_search_endpoint():
    """
    API endpoint for MongoDB Atlas Vector Search.
    Expects JSON body: {"query": "search term"}
    """
    data = request.get_json()
    query_text = data.get('query')
    limit = data.get('limit', 10)

    if not query_text:
        return jsonify({"error": "Query 'q' is required for vector search."}), 400

    try:
        # Get the analyzer class dynamically
        _, GeminiAnalyzerTask = get_task_classes()

        google_api_key = current_app.config['GOOGLE_API_KEY']
        analyzer_task_instance = GeminiAnalyzerTask(db, google_api_key)

        query_embedding = analyzer_task_instance.generate_embedding(query_text)

        if query_embedding is None:
            return jsonify({"error": "Failed to generate embedding for the query."}), 500

        results = article_service.perform_vector_search(query_embedding, limit)

        if "error" in results:
            return jsonify({"success": False, "error": results["error"]}), 500
        return jsonify({"success": True, "data": results["articles"]}), 200

    except Exception as e:
        current_app.logger.error(f"Error during vector search: {e}", exc_info=True)
        return jsonify({"success": False, "error": "An error occurred during vector search.", "details": str(e)}), 500
