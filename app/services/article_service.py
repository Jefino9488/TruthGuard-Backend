# flask_backend/app/services/article_service.py

import logging
from bson.objectid import ObjectId
from pymongo import ASCENDING, DESCENDING

logger = logging.getLogger(__name__)

class ArticleService:
    def __init__(self, db_client):
        self.db = db_client
        self.articles_collection = self.db.articles

    def get_all_articles(self, page=1, limit=10, sort_by="published_at", sort_order="desc"):
        """
        Retrieves a paginated list of articles from the database.
        """
        skip = (page - 1) * limit
        sort_direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING

        try:
            articles_cursor = self.articles_collection.find({}) \
                .sort(sort_by, sort_direction) \
                .skip(skip) \
                .limit(limit)
            articles = []
            for article in articles_cursor:
                article['_id'] = str(article['_id']) # Convert ObjectId to string for JSON serialization
                articles.append(article)

            total_articles = self.articles_collection.count_documents({})

            logger.info(f"Retrieved {len(articles)} articles (page {page}, limit {limit})")
            return {
                "articles": articles,
                "total_results": total_articles,
                "page": page,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error fetching all articles: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}

    def get_article_by_id(self, article_id):
        """
        Retrieves a single article by its MongoDB ObjectId.
        """
        try:
            article = self.articles_collection.find_one({"_id": ObjectId(article_id)})
            if article:
                article['_id'] = str(article['_id'])
                logger.info(f"Retrieved article with ID: {article_id}")
                return article
            else:
                logger.warning(f"Article with ID {article_id} not found.")
                return None
        except Exception as e:
            logger.error(f"Error fetching article by ID {article_id}: {e}")
            return None

    def search_articles(self, query, page=1, limit=10, sort_by="published_at", sort_order="desc"):
        """
        Searches articles using MongoDB's text index and potentially vector search.
        For true vector search, you'd implement $vectorSearch (requires Atlas).
        For now, this uses text search.
        """
        skip = (page - 1) * limit
        sort_direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING

        try:
            # Using $text operator for full-text search
            # Ensure you have a text index on 'title' and 'content' fields
            # self.collection.create_index([("title", "text"), ("content", "text")])

            pipeline = [
                {
                    '$match': {
                        '$text': {
                            '$search': query
                        }
                    }
                },
                {
                    '$project': {
                        'score': { '$meta': 'textScore' },
                        'article_id': 1,
                        'title': 1,
                        'url': 1,
                        'source': 1,
                        'published_at': 1,
                        'description': 1,
                        'scraped_at': 1,
                        'processing_status': 1,
                        'bias_score': 1,
                        'misinformation_risk': 1,
                        'sentiment': 1,
                        'credibility_score': 1,
                        'ai_analysis': 1 # Include full analysis for detail
                    }
                },
                { '$sort': { 'score': { '$meta': 'textScore' }, sort_by: sort_direction } }, # Sort by text score first, then other criteria
                { '$skip': skip },
                { '$limit': limit }
            ]

            articles = list(self.articles_collection.aggregate(pipeline))

            # Convert ObjectId to string for JSON serialization
            for article in articles:
                article['_id'] = str(article['_id'])

            # Count total matches for pagination
            total_results_pipeline = [
                {
                    '$match': {
                        '$text': {
                            '$search': query
                        }
                    }
                },
                { '$count': 'total_results' }
            ]
            total_results_cursor = self.articles_collection.aggregate(total_results_pipeline)
            total_results = next(total_results_cursor, {}).get('total_results', 0)

            logger.info(f"Searched for '{query}', found {total_results} results. Retrieved {len(articles)} (page {page}, limit {limit})")
            return {
                "articles": articles,
                "total_results": total_results,
                "page": page,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error searching articles with query '{query}': {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}

    # You could add methods here for specific filtering or analysis result queries
    def get_articles_by_bias_score(self, min_score=0.7, page=1, limit=10, sort_order="desc"):
        """
        Retrieves articles with a bias_score above a certain threshold.
        """
        skip = (page - 1) * limit
        sort_direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING

        try:
            query = {"bias_score": {"$gte": min_score}}
            articles_cursor = self.articles_collection.find(query) \
                .sort("bias_score", sort_direction) \
                .skip(skip) \
                .limit(limit)
            articles = []
            for article in articles_cursor:
                article['_id'] = str(article['_id'])
                articles.append(article)

            total_articles = self.articles_collection.count_documents(query)

            logger.info(f"Retrieved {len(articles)} high-bias articles.")
            return {
                "articles": articles,
                "total_results": total_articles,
                "page": page,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error fetching articles by bias score: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}

    def get_articles_by_misinformation_risk(self, min_risk=0.6, page=1, limit=10, sort_order="desc"):
        """
        Retrieves articles with a misinformation_risk above a certain threshold.
        """
        skip = (page - 1) * limit
        sort_direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING

        try:
            query = {"misinformation_risk": {"$gte": min_risk}}
            articles_cursor = self.articles_collection.find(query) \
                .sort("misinformation_risk", sort_direction) \
                .skip(skip) \
                .limit(limit)
            articles = []
            for article in articles_cursor:
                article['_id'] = str(article['_id'])
                articles.append(article)

            total_articles = self.articles_collection.count_documents(query)

            logger.info(f"Retrieved {len(articles)} high-misinformation risk articles.")
            return {
                "articles": articles,
                "total_results": total_articles,
                "page": page,
                "limit": limit
            }
        except Exception as e:
            logger.error(f"Error fetching articles by misinformation risk: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}