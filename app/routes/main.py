# flask_backend/app/routes/main.py
from datetime import datetime, timezone

from flask import Blueprint, request, jsonify, current_app
from app.tasks import NewsAPIFetcherTask, GeminiAnalyzerTask
from app.services import ArticleService
from app import db
import threading
import hashlib
from newspaper import Article

main_bp = Blueprint('main', __name__)

# Re-initialize the ArticleService here since it's used by multiple routes
# and relies on the `db` object which is initialized when `create_app` runs.
# This ensures it has access to the database.
article_service = ArticleService(db)  #


@main_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({"status": "ok", "message": "TruthGuard Backend is healthy!"}), 200


@main_bp.route('/scrape', methods=['POST'])
def trigger_scrape():
    """
    API endpoint to trigger the news scraping task.
    """
    try:
        news_api_key = current_app.config['NEWS_API_KEY']
        scraper = NewsAPIFetcherTask(db, news_api_key)

        thread = threading.Thread(target=scraper.run_scraper)
        thread.start()

        return jsonify({"message": "News scraping initiated successfully!", "status": "processing"}), 202  #
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
        analyzer = GeminiAnalyzerTask(db, google_api_key)

        thread = threading.Thread(target=analyzer.run_analyzer, args=(batch_size,))
        thread.start()

        return jsonify({"message": "AI analysis initiated successfully!", "status": "processing"}), 202  #
    except Exception as e:
        current_app.logger.error(f"Error triggering analysis: {e}")
        return jsonify({"error": "Failed to initiate analysis task", "details": str(e)}), 500  #


# NEW FEATURE: Manual Analysis Endpoint
@main_bp.route('/analyze-manual', methods=['POST'])
def manual_analysis():
    """
    API endpoint to manually analyze a given headline, content, or URL.
    Expects JSON body: {"headline": "...", "content": "..."} OR {"url": "..."}
    """
    data = request.get_json()  #
    if not data:
        return jsonify({"error": "Request must be JSON"}), 400

    headline = data.get('headline')
    content = data.get('content')
    url = data.get('url')

    article_to_analyze = {}
    temp_article_id = None

    try:
        if url:
            current_app.logger.info(f"Manual analysis: Scraping content from URL: {url}")  #
            temp_article = Article(url)
            temp_article.download()
            temp_article.parse()

            # Use content if available, fallback to title/description if not
            scraped_content = temp_article.text or temp_article.meta_description or temp_article.title
            scraped_title = temp_article.title

            # --- FIX STARTS HERE ---
            scraped_source = temp_article.meta_site_name  # Use meta_site_name
            if not scraped_source and temp_article.url:  # Fallback to parsing URL if meta_site_name is not available
                try:
                    from urllib.parse import urlparse
                    parsed_url = urlparse(temp_article.url)
                    scraped_source = parsed_url.netloc
                except Exception:
                    scraped_source = "Unknown"
            elif not scraped_source:
                scraped_source = "Unknown"
            # --- FIX ENDS HERE ---

            if not scraped_content or len(scraped_content) < 50:  # Minimum content length for meaningful analysis
                return jsonify({"error": "Could not extract sufficient content from the provided URL."}), 400

            article_to_analyze = {
                "article_id": hashlib.md5(url.encode()).hexdigest(),  # Generate a unique ID for this ad-hoc analysis #
                "title": scraped_title,
                "url": url,
                "source": scraped_source,
                "content": scraped_content,
                "processing_status": "manual_pending",  # Indicate it's for manual analysis #
                "published_at": datetime.now(timezone.utc).isoformat()  # Use current time for manual analysis #
            }
            temp_article_id = article_to_analyze["article_id"]

        elif headline and content:
            current_app.logger.info(f"Manual analysis: Analyzing provided headline and content.")
            article_to_analyze = {
                "article_id": hashlib.md5((headline + content).encode()).hexdigest(),  #
                "title": headline,
                "content": content,
                "source": "Manual Input",
                "processing_status": "manual_pending",
                "published_at": datetime.now(timezone.utc).isoformat()  #
            }
            temp_article_id = article_to_analyze["article_id"]

        else:
            return jsonify(
                {"error": "Please provide either 'url' OR 'headline' and 'content' for manual analysis."}), 400

        google_api_key = current_app.config['GOOGLE_API_KEY']  #
        analyzer = GeminiAnalyzerTask(db, google_api_key)  #

        # Call the new raw content analysis method
        analysis_result = analyzer.analyze_raw_content(  #
            title=article_to_analyze.get("title"),
            content=article_to_analyze.get("content")
        )

        if analysis_result:
            return jsonify(analysis_result), 200
        else:
            return jsonify({"error": "Failed to perform manual analysis."}), 500  #

    except Exception as e:
        current_app.logger.error(f"Error during manual analysis: {e}", exc_info=True)
        return jsonify({"error": "An unexpected error occurred during manual analysis.", "details": str(e)}), 500


@main_bp.route('/articles', methods=['GET'])
def get_articles():
    """
    API endpoint to retrieve a paginated list of articles.
    Query parameters: page, limit, sort_by, sort_order
    """
    page = request.args.get('page', 1, type=int)  #
    limit = request.args.get('limit', 10, type=int)  #
    sort_by = request.args.get('sort_by', 'published_at', type=str)  #
    sort_order = request.args.get('sort_order', 'desc', type=str)  #

    service = ArticleService(db)
    result = service.get_all_articles(page, limit, sort_by, sort_order)  #
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200


@main_bp.route('/articles/<article_id>', methods=['GET'])
def get_article_detail(article_id):
    """
    API endpoint to retrieve details of a single article by its ID.
    """
    service = ArticleService(db)
    article = service.get_article_by_id(article_id)  #
    if article:
        return jsonify(article), 200
    return jsonify({"message": "Article not found"}), 404


@main_bp.route('/articles/search', methods=['GET'])
def search_articles_endpoint():
    """
    API endpoint to search articles.
    Query parameters: q (query string), page, limit, sort_by, sort_order
    """
    query = request.args.get('q', type=str)  #
    page = request.args.get('page', 1, type=int)  #
    limit = request.args.get('limit', 10, type=int)  #
    sort_by = request.args.get('sort_by', 'score', type=str)  # Default sort by text score for search #
    sort_order = request.args.get('sort_order', 'desc', type=str)  #

    if not query:
        return jsonify({"error": "Query parameter 'q' is required for search."}), 400

    service = ArticleService(db)
    result = service.search_articles(query, page, limit, sort_by, sort_order)  #
    if "error" in result:
        return jsonify(result), 500  #
    return jsonify(result), 200


@main_bp.route('/articles/high-bias', methods=['GET'])
def get_high_bias_articles():
    """
    API endpoint to retrieve articles flagged with high bias.
    Query parameters: min_score, page, limit, sort_order
    """
    min_score = request.args.get('min_score', 0.7, type=float)  #
    page = request.args.get('page', 1, type=int)  #
    limit = request.args.get('limit', 10, type=int)  #
    sort_order = request.args.get('sort_order', 'desc', type=str)  #

    service = ArticleService(db)
    result = service.get_articles_by_bias_score(min_score, page, limit, sort_order)  #
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200


@main_bp.route('/articles/misinformation-risk', methods=['GET'])
def get_misinformation_risk_articles():
    """
    API endpoint to retrieve articles flagged with high misinformation risk.
    Query parameters: min_risk, page, limit, sort_order
    """
    min_risk = request.args.get('min_risk', 0.6, type=float)  #
    page = request.args.get('page', 1, type=int)  #
    limit = request.args.get('limit', 10, type=int)  #
    sort_order = request.args.get('sort_order', 'desc', type=str)  #

    service = ArticleService(db)
    result = service.get_articles_by_misinformation_risk(min_risk, page, limit, sort_order)  #
    if "error" in result:
        return jsonify(result), 500
    return jsonify(result), 200
