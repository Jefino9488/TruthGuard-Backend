import os
from datetime import datetime
import time
import logging
import hashlib
from newspaper import Article
from sentence_transformers import SentenceTransformer
import numpy as np
from newsapi import NewsApiClient
from concurrent.futures import ThreadPoolExecutor, as_completed
from pymongo.errors import PyMongoError  # Import for MongoDB specific errors

logger = logging.getLogger(__name__)

CATEGORIES = ["business", "technology", "science", "health", "general"]
TOPICS = ["misinformation", "fact checking", "media bias", "artificial intelligence", "politics", "climate"]


class NewsAPIFetcherTask:
    def __init__(self, db_client, api_key_news, model_path='all-MiniLM-L6-v2'):
        self.db = db_client
        self.collection = self.db.articles
        self.newsapi = NewsApiClient(api_key=api_key_news)
        logger.info(f"Loading sentence transformer model: {model_path}...")
        self.model = SentenceTransformer(model_path)

        # Ensure indexes for full-text search (if not already created)
        logger.info("Creating MongoDB text index on 'title' and 'content' fields if not exist...")
        try:
            self.collection.create_index([("title", "text"), ("content", "text")], default_language='english')
        except PyMongoError as e:
            if "Index with name" not in str(e):  # Ignore error if index already exists
                logger.error(f"Failed to create text index: {e}")

        # IMPORTANT: For $vectorSearch, you MUST manually create a Vector Search Index
        # in your MongoDB Atlas cluster UI or via Atlas CLI/API.
        # Example definition (if your embedding dimension is 384):
        # {
        #   "mappings": {
        #     "dynamic": true,
        #     "fields": {
        #       "content_embedding": {
        #         "type": "knnVector",
        #         "dimensions": 384,
        #         "similarity": "cosine"
        #       },
        #       "title_embedding": {
        #         "type": "knnVector",
        #         "dimensions": 384,
        #         "similarity": "cosine"
        #       },
        #       "analysis_embedding": { # If you also embed the AI analysis summary
        #         "type": "knnVector",
        #         "dimensions": 384,
        #         "similarity": "cosine"
        #       }
        #     }
        #   }
        # }
        logger.info(
            "Remember to manually create MongoDB Atlas Vector Search Index (e.g., 'vector_index') for embeddings.")

        self.stats = {
            'categories_processed': 0,
            'topics_processed': 0,
            'articles_found': 0,
            'articles_stored': 0,
            'duplicates_skipped': 0,
            'errors': 0,
            'embeddings_generated': 0
        }

    def fetch_top_headlines(self, country="us", category=None, page_size=20):
        """Fetch top headlines from News API using newsapi-python."""
        try:
            params = {
                "country": country,
                "page_size": page_size
            }
            if category:
                params["category"] = category

            response = self.newsapi.get_top_headlines(**params)

            if response["status"] != "ok":
                logger.error(f"News API Error (Top Headlines): {response.get('message', 'Unknown error')}")
                return []

            logger.info(f"Fetched {len(response['articles'])} top headlines for category: {category or 'all'}")
            return response["articles"]

        except Exception as e:
            logger.error(f"Error fetching top headlines: {e}")
            self.stats['errors'] += 1
            return []

    def fetch_everything(self, query, language="en", sort_by="publishedAt", page_size=20):
        """Fetch articles matching query from News API using newsapi-python."""
        try:
            response = self.newsapi.get_everything(
                q=query,
                language=language,
                sort_by=sort_by,
                page_size=page_size
            )

            if response["status"] != "ok":
                logger.error(f"News API Error (Everything): {response.get('message', 'Unknown error')}")
                return []

            logger.info(f"Fetched {len(response['articles'])} articles for query: {query}")
            return response["articles"]

        except Exception as e:
            logger.error(f"Error fetching articles: {e}")
            self.stats['errors'] += 1
            return []

    def extract_full_content(self, url):
        """Extract full article content using newspaper3k."""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text if article.text else ""
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""

    def generate_embedding(self, text):
        """Generate vector embedding for text using sentence-transformers."""
        try:
            # Maximum input length for all-MiniLM-L6-v2 is 256 tokens.
            # Truncate to ensure it fits, or use a model with larger context.
            max_char_length = 500  # Using 500 characters as a rough safe limit
            if len(text) > max_char_length:
                text = text[:max_char_length]
            embedding = self.model.encode(text, convert_to_numpy=True)
            self.stats['embeddings_generated'] += 1
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def process_article(self, article):
        """Process a single article from News API."""
        try:
            title = article.get("title", "")
            url = article.get("url", "")
            source_name = article.get("source", {}).get("name", "Unknown")
            published_at_str = article.get("publishedAt", "")
            description = article.get("description", "")
            category = article.get("category", "general")  # Assuming category might be passed or derived

            if not title or not url:
                logger.warning(f"Skipping article with missing title or URL")
                return None

            article_id = hashlib.md5(url.encode()).hexdigest()

            # Check for existing article by article_id or URL
            existing = self.collection.find_one({"$or": [{"article_id": article_id}, {"url": url}]})
            if existing:
                logger.info(f"Article already exists: {title[:50]}...")
                self.stats['duplicates_skipped'] += 1
                return None

            content = self.extract_full_content(url)

            if not content and description:
                content = description

            if len(content) < 200:  # Minimum content length for meaningful analysis
                logger.warning(f"Skipping article with insufficient content: {title[:50]}...")
                return None

            # Convert published_at to datetime object if possible, otherwise keep as string or None
            published_at_dt = None
            if published_at_str:
                try:
                    published_at_dt = datetime.fromisoformat(published_at_str.replace('Z', '+00:00'))
                except ValueError:
                    logger.warning(f"Could not parse published_at: {published_at_str}")

            content_embedding = self.generate_embedding(content)
            title_embedding = self.generate_embedding(title)

            article_doc = {
                "article_id": article_id,
                "title": title,
                "url": url,
                "source": source_name,
                "published_at": published_at_dt,  # Store as datetime object
                "content": content,
                "description": description,
                "scraped_at": datetime.utcnow(),
                "processing_status": "pending",  # Initial status for AI analysis
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
                "word_count": len(content.split()),
                "content_embedding": content_embedding,  # Store embeddings for vector search
                "title_embedding": title_embedding,  # Store embeddings for vector search
                "data_source": "news_api",
                "category": category  # Add category for better topic analysis
            }
            return article_doc

        except Exception as e:
            logger.error(f"Error processing article: {e}", exc_info=True)
            self.stats['errors'] += 1
            return None

    def store_articles(self, articles):
        """Store articles in MongoDB."""
        if not articles:
            return 0

        try:
            # Filter out None articles (from skipping) before bulk insert
            valid_articles = [a for a in articles if a is not None]
            if not valid_articles:
                return 0

            inserted_count = 0
            # Use insert_many for efficiency
            result = self.collection.insert_many(valid_articles, ordered=False)  # ordered=False to continue on error
            inserted_count = len(result.inserted_ids)
            logger.info(f"Stored {inserted_count} new articles in MongoDB.")

            self.stats['articles_stored'] += inserted_count
            return inserted_count

        except PyMongoError as e:
            logger.error(f"MongoDB error storing articles: {e}")
            self.stats['errors'] += 1
            return 0
        except Exception as e:
            logger.error(f"Error storing articles: {e}", exc_info=True)
            self.stats['errors'] += 1
            return 0

    def run_scraper(self):
        """Run the complete fetching process."""
        logger.info("Starting TruthGuard News API fetching task...")

        # Process by categories first
        for category in CATEGORIES:
            logger.info(f"Fetching top headlines for category: {category}")
            articles_from_category = self.fetch_top_headlines(category=category)

            processed_articles_batch = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit articles for processing, ensuring category is passed
                future_to_article = {
                    executor.submit(self.process_article, {**article, 'category': category}): article
                    for article in articles_from_category
                }
                for future in as_completed(future_to_article):
                    processed = future.result()
                    if processed:
                        processed_articles_batch.append(processed)
                        self.stats['articles_found'] += 1

            self.store_articles(processed_articles_batch)
            self.stats['categories_processed'] += 1
            time.sleep(1)  # Respect News API rate limits

        # Process by topics
        for topic in TOPICS:
            logger.info(f"Fetching articles for topic: {topic}")
            articles_from_topic = self.fetch_everything(query=topic)

            processed_articles_batch = []
            with ThreadPoolExecutor(max_workers=5) as executor:
                # Submit articles for processing, ensuring topic is passed
                future_to_article = {
                    executor.submit(self.process_article, {**article, 'category': topic}): article
                    for article in articles_from_topic
                }
                for future in as_completed(future_to_article):
                    processed = future.result()
                    if processed:
                        processed_articles_batch.append(processed)
                        self.stats['articles_found'] += 1

            self.store_articles(processed_articles_batch)
            self.stats['topics_processed'] += 1
            time.sleep(1)  # Respect News API rate limits

        logger.info(f"Fetching complete. Stats: {self.stats}")
        return self.stats
