# flask_backend/app/tasks/scraper.py

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

logger = logging.getLogger(__name__)

# News API categories and topics
CATEGORIES = ["business", "technology", "science", "health", "general"]
TOPICS = ["misinformation", "fact checking", "media bias", "artificial intelligence", "politics", "climate"]

class NewsAPIFetcherTask:
    def __init__(self, db_client, api_key_news, model_path='all-MiniLM-L6-v2'):
        self.db = db_client
        self.collection = self.db.articles
        self.newsapi = NewsApiClient(api_key=api_key_news)
        logger.info(f"Loading sentence transformer model: {model_path}...")
        self.model = SentenceTransformer(model_path) # 384 dimensions

        # Ensure indexes for full-text search
        logger.info("Creating MongoDB indexes if they don't exist...")
        self.collection.create_index([("title", "text"), ("content", "text")])
        # Create vector search index for content_embedding (requires MongoDB Atlas Search setup)
        # For programmatic creation, you'd typically use `create_search_index` which is an Atlas feature.
        # Example (conceptual, requires Atlas specific setup):
        # self.collection.create_search_index(
        #     name="default", # Or a specific name
        #     definition={
        #         "mappings": {
        #             "dynamic": True,
        #             "fields": {
        #                 "content_embedding": {
        #                     "type": "knnVector",
        #                     "dimensions": 384,
        #                     "similarity": "cosine"
        #                 },
        #                 "title_embedding": {
        #                     "type": "knnVector",
        #                     "dimensions": 384,
        #                     "similarity": "cosine"
        #                 }
        #             }
        #         }
        #     }
        # )

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
        """Fetch top headlines from News API using newsapi-python"""
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
        """Fetch articles matching query from News API using newsapi-python"""
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
        """Extract full article content using newspaper3k"""
        try:
            article = Article(url)
            article.download()
            article.parse()
            return article.text if article.text else ""
        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            return ""

    def generate_embedding(self, text):
        """Generate vector embedding for text using sentence-transformers"""
        try:
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length]
            embedding = self.model.encode(text)
            self.stats['embeddings_generated'] += 1
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def process_article(self, article):
        """Process a single article from News API"""
        try:
            title = article.get("title", "")
            url = article.get("url", "")
            source_name = article.get("source", {}).get("name", "Unknown")
            published_at = article.get("publishedAt", "")
            description = article.get("description", "")

            if not title or not url:
                logger.warning(f"Skipping article with missing title or URL")
                return None

            article_id = hashlib.md5(url.encode()).hexdigest()

            existing = self.collection.find_one({"article_id": article_id})
            if existing:
                logger.info(f"Article already exists: {title[:50]}...")
                self.stats['duplicates_skipped'] += 1
                return None

            content = self.extract_full_content(url)

            if not content and description:
                content = description

            if len(content) < 200:
                logger.warning(f"Skipping article with insufficient content: {title[:50]}...")
                return None

            content_embedding = self.generate_embedding(content)
            title_embedding = self.generate_embedding(title)

            article_doc = {
                "article_id": article_id,
                "title": title,
                "url": url,
                "source": source_name,
                "published_at": published_at,
                "content": content,
                "description": description,
                "scraped_at": datetime.utcnow(),
                "processed": False, # Will be set to True after AI analysis
                "processing_status": "pending", # Initial status for AI analysis
                "content_hash": hashlib.md5(content.encode()).hexdigest(),
                "word_count": len(content.split()),
                "content_embedding": content_embedding,
                "title_embedding": title_embedding,
                "data_source": "news_api"
            }
            return article_doc

        except Exception as e:
            logger.error(f"Error processing article: {e}")
            self.stats['errors'] += 1
            return None

    def store_articles(self, articles):
        """Store articles in MongoDB"""
        if not articles:
            return 0

        try:
            inserted_count = 0
            # Use bulk insert for efficiency if many articles
            # Otherwise, iterate and insert one by one (as per original logic)
            for article in articles:
                if article:
                    self.collection.insert_one(article)
                    inserted_count += 1
                    logger.info(f"Stored: {article['title'][:50]}...")

            self.stats['articles_stored'] += inserted_count
            logger.info(f"Stored {inserted_count} articles in MongoDB")
            return inserted_count

        except Exception as e:
            logger.error(f"Error storing articles: {e}")
            self.stats['errors'] += 1
            return 0

    def run_scraper(self):
        """Run the complete fetching process"""
        logger.info("Starting TruthGuard News API fetching task...")

        all_articles = []

        for category in CATEGORIES:
            logger.info(f"Fetching top headlines for category: {category}")
            articles = self.fetch_top_headlines(category=category)

            processed_articles = []
            with ThreadPoolExecutor(max_workers=5) as executor: # Use a thread pool for processing
                future_to_article = {executor.submit(self.process_article, article): article for article in articles}
                for future in as_completed(future_to_article):
                    processed = future.result()
                    if processed:
                        processed_articles.append(processed)
                        self.stats['articles_found'] += 1

            logger.info(f"Processed {len(processed_articles)} articles for category {category}")
            self.store_articles(processed_articles) # Store articles in batches
            self.stats['categories_processed'] += 1
            time.sleep(1) # Rate limiting for News API

        for topic in TOPICS:
            logger.info(f"Fetching articles for topic: {topic}")
            articles = self.fetch_everything(query=topic)

            processed_articles = []
            with ThreadPoolExecutor(max_workers=5) as executor: # Use a thread pool for processing
                future_to_article = {executor.submit(self.process_article, article): article for article in articles}
                for future in as_completed(future_to_article):
                    processed = future.result()
                    if processed:
                        processed_articles.append(processed)
                        self.stats['articles_found'] += 1

            logger.info(f"Processed {len(processed_articles)} articles for topic {topic}")
            self.store_articles(processed_articles) # Store articles in batches
            self.stats['topics_processed'] += 1
            time.sleep(1) # Rate limiting for News API

        logger.info(f"Fetching complete. Stats: {self.stats}")
        return self.stats