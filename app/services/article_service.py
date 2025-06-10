import logging
from bson.objectid import ObjectId
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

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
                article['_id'] = str(article['_id'])  # Convert ObjectId to string for JSON serialization
                articles.append(article)

            total_articles = self.articles_collection.count_documents({})

            logger.info(f"Retrieved {len(articles)} articles (page {page}, limit {limit})")
            return {
                "articles": articles,
                "total_results": total_articles,
                "page": page,
                "limit": limit
            }
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching all articles: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}
        except Exception as e:
            logger.error(f"Error fetching all articles: {e}", exc_info=True)
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
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching article by ID {article_id}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error fetching article by ID {article_id}: {e}", exc_info=True)
            return None

    def search_articles(self, query, page=1, limit=10, sort_by="published_at", sort_order="desc"):
        """
        Searches articles using MongoDB's text index.
        """
        skip = (page - 1) * limit
        sort_direction = ASCENDING if sort_order.lower() == "asc" else DESCENDING

        try:
            # Ensure you have a text index on 'title' and 'content' fields
            # self.articles_collection.create_index([("title", "text"), ("content", "text")])

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
                        'score': {'$meta': 'textScore'},
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
                        'ai_analysis': 1
                    }
                },
                {'$sort': {'score': {'$meta': 'textScore'}, sort_by: sort_direction}},
                {'$skip': skip},
                {'$limit': limit}
            ]

            articles = list(self.articles_collection.aggregate(pipeline))

            for article in articles:
                article['_id'] = str(article['_id'])

            total_results_pipeline = [
                {
                    '$match': {
                        '$text': {
                            '$search': query
                        }
                    }
                },
                {'$count': 'total_results'}
            ]
            total_results_cursor = self.articles_collection.aggregate(total_results_pipeline)
            total_results = next(total_results_cursor, {}).get('total_results', 0)

            logger.info(
                f"Searched for '{query}', found {total_results} results. Retrieved {len(articles)} (page {page}, limit {limit})")
            return {
                "articles": articles,
                "total_results": total_results,
                "page": page,
                "limit": limit
            }
        except PyMongoError as e:
            logger.error(f"MongoDB error searching articles with query '{query}': {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}
        except Exception as e:
            logger.error(f"Error searching articles with query '{query}': {e}", exc_info=True)
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}

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
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching articles by bias score: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}
        except Exception as e:
            logger.error(f"Error fetching articles by bias score: {e}", exc_info=True)
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
        except PyMongoError as e:
            logger.error(f"MongoDB error fetching articles by misinformation risk: {e}")
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}
        except Exception as e:
            logger.error(f"Error fetching articles by misinformation risk: {e}", exc_info=True)
            return {"articles": [], "total_results": 0, "page": page, "limit": limit, "error": str(e)}

    def get_dashboard_analytics(self):
        """
        Retrieves aggregated analytics data for the dashboard components.
        Combines multiple aggregations into one query for efficiency.
        """
        try:
            pipeline = [
                {
                    '$facet': {
                        'totalStats': [
                            {
                                '$group': {
                                    '_id': None,
                                    'totalArticles': {'$sum': 1},
                                    'avgBias': {'$avg': '$bias_score'},
                                    'avgCredibility': {'$avg': '$credibility_score'},
                                    'avgMisinfoRisk': {'$avg': '$misinformation_risk'},
                                    'uniqueSources': {'$addToSet': '$source'},
                                    'uniqueTopics': {'$addToSet': '$category'}
                                    # Assuming 'category' is your topic field
                                }
                            }
                        ],
                        'biasDistribution': [
                            {
                                '$bucket': {
                                    'groupBy': '$bias_score',
                                    'boundaries': [0, 0.2, 0.4, 0.6, 0.8, 1.01],  # 1.01 to include 1.0
                                    'default': 'unknown',
                                    'output': {
                                        'count': {'$sum': 1}
                                    }
                                }
                            },
                            {
                                '$project': {
                                    '_id': {  # Map bucket boundaries to representative values or labels
                                        '$switch': {
                                            'branches': [
                                                {'case': {'$eq': ['$_id', 0]}, 'then': 0},
                                                {'case': {'$eq': ['$_id', 0.2]}, 'then': 0.2},
                                                {'case': {'$eq': ['$_id', 0.4]}, 'then': 0.4},
                                                {'case': {'$eq': ['$_id', 0.6]}, 'then': 0.6},
                                                {'case': {'$eq': ['$_id', 0.8]}, 'then': 0.8}
                                            ],
                                            'default': 'unknown'
                                        }
                                    },
                                    'count': 1
                                }
                            },
                            {'$sort': {'_id': 1}}
                        ],
                        'sourceComparison': [
                            {
                                '$group': {
                                    '_id': '$source',
                                    'articleCount': {'$sum': 1},
                                    'averageBias': {'$avg': '$bias_score'},
                                    'averageMisinformationRisk': {'$avg': '$misinformation_risk'},
                                    'averageCredibility': {'$avg': '$credibility_score'}
                                }
                            },
                            {'$sort': {'articleCount': -1}}
                        ]
                    }
                }
            ]

            result = list(self.articles_collection.aggregate(pipeline))

            if result:
                return result[0]  # The $facet stage always returns an array with one document
            return {}
        except PyMongoError as e:
            logger.error(f"MongoDB error getting dashboard analytics: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting dashboard analytics: {e}", exc_info=True)
            return {"error": str(e)}

    def perform_vector_search(self, query_embedding: list[float], limit: int = 10):
        """
        Performs a MongoDB Atlas Vector Search on articles.
        Requires a pre-configured Atlas Vector Search Index (e.g., 'vector_index')
        on 'content_embedding' or 'title_embedding' fields.
        """
        try:
            pipeline = [
                {
                    '$vectorSearch': {
                        'index': 'vector_index',  # IMPORTANT: Replace with your actual vector index name
                        'path': 'content_embedding',  # Field containing the vector embeddings
                        'queryVector': query_embedding,
                        'numCandidates': 100,  # Number of nearest neighbors to consider
                        'limit': limit
                    }
                },
                {
                    '$project': {
                        '_id': {'$toString': '$_id'},  # Convert ObjectId to string
                        'title': 1,
                        'url': 1,
                        'source': 1,
                        'published_at': 1,
                        'bias_score': 1,
                        'misinformation_risk': 1,
                        'credibility_score': 1,
                        'ai_analysis': 1,  # Include full analysis
                        'vectorSearchScore': {'$meta': 'vectorSearchScore'}  # Similarity score
                    }
                }
            ]
            articles = list(self.articles_collection.aggregate(pipeline))
            logger.info(f"Performed vector search, found {len(articles)} results.")
            return {"articles": articles}
        except PyMongoError as e:
            logger.error(f"MongoDB Vector Search error: {e}")
            # Provide more specific guidance if it's an index error
            if "index not found" in str(e).lower() or "vector search" in str(e).lower():
                return {
                    "error": f"MongoDB Atlas Vector Search failed. Ensure a vector index named 'vector_index' (or your custom name) is configured on your 'articles' collection on Atlas. Details: {e}"}
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error performing vector search: {e}", exc_info=True)
            return {"error": str(e)}
