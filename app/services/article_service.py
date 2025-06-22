import logging
from bson.objectid import ObjectId
from pymongo import ASCENDING, DESCENDING
from pymongo.errors import PyMongoError

logger = logging.getLogger(__name__)


class ArticleService:
    def __init__(self, db_client):
        self.db = db_client
        # Use get_collection instead of attribute access
        self.articles_collection = self.db.get_collection('articles')

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
            # Convert numpy array to regular Python list if needed
            if hasattr(query_embedding, 'tolist'):
                query_embedding = query_embedding.tolist()

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
        except Exception as e:
            logger.error(f"Error performing vector search: {e}")
            # Provide more specific guidance if it's an index error
            if "index not found" in str(e).lower() or "vector search" in str(e).lower():
                return {
                    "error": f"MongoDB Atlas Vector Search failed. Ensure a vector index named 'vector_index' (or your custom name) is configured on your 'articles' collection on Atlas. Details: {e}"}
            return {"error": str(e)}

    def get_media_landscape_data(self):
        """
        Retrieves data for the media landscape visualization (bubble chart).
        Returns information about media sources with their bias, reliability, and reach.
        """
        try:
            pipeline = [
                {
                    '$group': {
                        '_id': '$source',
                        'articleCount': {'$sum': 1},
                        'bias': {'$avg': '$bias_score'},
                        'reliability': {'$avg': '$credibility_score'},
                        'misinformationRisk': {'$avg': '$misinformation_risk'}
                    }
                },
                {
                    '$project': {
                        'name': '$_id',
                        'bias': 1,
                        'reliability': 1,
                        'reach': {
                            '$multiply': [
                                '$articleCount',
                                {'$divide': [1, {'$max': [1, {'$sqrt': '$articleCount'}]}]}
                            ]
                        },
                        'category': {
                            '$cond': {
                                'if': {'$regexMatch': {'input': {'$toLower': '$_id'}, 'regex': 'tv|cnn|fox|msnbc|bbc'}},
                                'then': 'TV',
                                'else': {
                                    '$cond': {
                                        'if': {'$regexMatch': {'input': {'$toLower': '$_id'}, 'regex': 'times|post|journal|guardian|economist'}},
                                        'then': 'Print',
                                        'else': {
                                            '$cond': {
                                                'if': {'$regexMatch': {'input': {'$toLower': '$_id'}, 'regex': 'reuters|ap|associated press'}},
                                                'then': 'Wire',
                                                'else': {
                                                    '$cond': {
                                                        'if': {'$regexMatch': {'input': {'$toLower': '$_id'}, 'regex': 'npr|radio|bbc'}},
                                                        'then': 'Radio',
                                                        'else': 'Online'
                                                    }
                                                }
                                            }
                                        }
                                    }
                                }
                            }
                        },
                        'articleCount': 1,
                        'misinformationRisk': 1
                    }
                },
                {'$sort': {'articleCount': -1}},
                {'$limit': 30}  # Limit to top 30 sources by article count
            ]

            result = list(self.articles_collection.aggregate(pipeline))
            logger.info(f"Retrieved media landscape data for {len(result)} sources.")
            return {"sources": result}
        except PyMongoError as e:
            logger.error(f"MongoDB error getting media landscape data: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting media landscape data: {e}", exc_info=True)
            return {"error": str(e)}

    def get_topic_clusters_data(self):
        """
        Retrieves data for the topic clusters visualization (force-directed graph).
        Returns information about topics and their relationships.
        """
        try:
            # First, get the most common topics/categories
            topics_pipeline = [
                {
                    '$group': {
                        '_id': '$category',
                        'count': {'$sum': 1},
                        'avgBias': {'$avg': '$bias_score'},
                        'avgMisinformationRisk': {'$avg': '$misinformation_risk'}
                    }
                },
                {'$match': {'_id': {'$ne': None}}},  # Filter out null categories
                {'$sort': {'count': -1}},
                {'$limit': 20}  # Top 20 topics
            ]

            topics = list(self.articles_collection.aggregate(topics_pipeline))

            # Create nodes from topics
            nodes = []
            for i, topic in enumerate(topics):
                if topic['_id'] is not None:
                    # Assign group based on bias level
                    group = 1
                    if 'avgBias' in topic and topic['avgBias'] is not None:
                        if topic['avgBias'] < 0.2:
                            group = 1
                        elif topic['avgBias'] < 0.4:
                            group = 2
                        elif topic['avgBias'] < 0.6:
                            group = 3
                        elif topic['avgBias'] < 0.8:
                            group = 4
                        else:
                            group = 5

                    nodes.append({
                        'id': topic['_id'],
                        'group': group,
                        'value': max(10, min(30, topic['count'] / 2))  # Scale node size between 10-30
                    })

            # Create links between topics based on co-occurrence
            links = []
            topic_ids = [topic['_id'] for topic in topics if topic['_id'] is not None]

            # For each pair of topics, find articles that mention both
            for i in range(len(topic_ids)):
                for j in range(i+1, len(topic_ids)):
                    topic1 = topic_ids[i]
                    topic2 = topic_ids[j]

                    # Simple heuristic for link strength based on topic similarity
                    # In a real implementation, you would analyze actual co-occurrence in articles
                    similarity_score = 0

                    # Find articles with related topics or keywords
                    related_pipeline = [
                        {
                            '$match': {
                                '$or': [
                                    {'category': topic1, 'ai_analysis.keywords': {'$regex': topic2, '$options': 'i'}},
                                    {'category': topic2, 'ai_analysis.keywords': {'$regex': topic1, '$options': 'i'}}
                                ]
                            }
                        },
                        {'$count': 'related_count'}
                    ]

                    related_result = list(self.articles_collection.aggregate(related_pipeline))
                    if related_result and 'related_count' in related_result[0]:
                        similarity_score = min(10, max(1, related_result[0]['related_count'] / 2))
                    else:
                        # If no direct relationship found, use a small default value
                        # based on general topic relatedness
                        if (topic1 in ['Politics', 'Economy', 'Policy'] and 
                            topic2 in ['Politics', 'Economy', 'Policy']):
                            similarity_score = 5
                        elif (topic1 in ['Climate', 'Environment', 'Energy'] and 
                              topic2 in ['Climate', 'Environment', 'Energy']):
                            similarity_score = 6
                        elif (topic1 in ['Technology', 'AI', 'Social Media'] and 
                              topic2 in ['Technology', 'AI', 'Social Media']):
                            similarity_score = 7
                        elif (topic1 in ['Healthcare', 'Pandemic'] and 
                              topic2 in ['Healthcare', 'Pandemic']):
                            similarity_score = 8
                        else:
                            similarity_score = 2

                    if similarity_score > 0:
                        links.append({
                            'source': topic1,
                            'target': topic2,
                            'value': similarity_score
                        })

            return {
                "nodes": nodes,
                "links": links
            }
        except PyMongoError as e:
            logger.error(f"MongoDB error getting topic clusters data: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting topic clusters data: {e}", exc_info=True)
            return {"error": str(e)}

    def get_narrative_flow_data(self):
        """
        Retrieves data for the narrative flow visualization (Sankey diagram).
        Returns information about how narratives evolve across different stages.
        """
        try:
            # Get the most recent articles for analysis
            recent_articles_pipeline = [
                {'$sort': {'published_at': -1}},
                {'$limit': 100},
                {
                    '$project': {
                        'title': 1,
                        'source': 1,
                        'category': 1,
                        'bias_score': 1,
                        'misinformation_risk': 1,
                        'ai_analysis': 1
                    }
                }
            ]

            recent_articles = list(self.articles_collection.aggregate(recent_articles_pipeline))

            # Define narrative stages
            stages = [
                "Initial Report", 
                "Economic Impact", 
                "Political Response", 
                "Public Reaction", 
                "Expert Analysis",
                "Policy Proposal",
                "Market Response",
                "Social Media",
                "Opposition View",
                "International Perspective",
                "Historical Context",
                "Future Implications"
            ]

            # Create nodes for each stage
            nodes = []
            for i, stage in enumerate(stages):
                group = (i % 6) + 1  # Assign group (1-6) based on stage index
                nodes.append({
                    "id": stage,
                    "group": group
                })

            # Create links between stages based on article analysis
            links = []

            # Define some common flows between narrative stages
            common_flows = [
                ("Initial Report", "Economic Impact", 5),
                ("Initial Report", "Political Response", 8),
                ("Initial Report", "Public Reaction", 6),
                ("Economic Impact", "Market Response", 7),
                ("Economic Impact", "Expert Analysis", 4),
                ("Political Response", "Policy Proposal", 6),
                ("Political Response", "Opposition View", 5),
                ("Public Reaction", "Social Media", 9),
                ("Expert Analysis", "Future Implications", 5),
                ("Expert Analysis", "Historical Context", 4),
                ("Policy Proposal", "Future Implications", 3),
                ("Policy Proposal", "International Perspective", 2),
                ("Opposition View", "Public Reaction", 4),
                ("Social Media", "Opposition View", 3),
                ("Market Response", "Future Implications", 4)
            ]

            # Add common flows to links
            for source, target, value in common_flows:
                links.append({
                    "source": source,
                    "target": target,
                    "value": value
                })

            # Analyze articles to enhance the narrative flow
            # This is a simplified approach; a real implementation would use more sophisticated NLP
            for article in recent_articles:
                if 'ai_analysis' in article and article['ai_analysis']:
                    # Use AI analysis to determine narrative stages
                    analysis = article['ai_analysis']

                    # Example: If article has high economic focus, strengthen economic flows
                    if 'economic' in str(analysis).lower() or article.get('category') == 'Economy':
                        for i, (source, target, _) in enumerate(common_flows):
                            if source == "Economic Impact" or target == "Economic Impact" or \
                               source == "Market Response" or target == "Market Response":
                                links[i]["value"] += 1

                    # Example: If article has high political focus, strengthen political flows
                    if 'political' in str(analysis).lower() or article.get('category') == 'Politics':
                        for i, (source, target, _) in enumerate(common_flows):
                            if source == "Political Response" or target == "Political Response" or \
                               source == "Policy Proposal" or target == "Policy Proposal" or \
                               source == "Opposition View" or target == "Opposition View":
                                links[i]["value"] += 1

            return {
                "nodes": nodes,
                "links": links
            }
        except PyMongoError as e:
            logger.error(f"MongoDB error getting narrative flow data: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Error getting narrative flow data: {e}", exc_info=True)
            return {"error": str(e)}
