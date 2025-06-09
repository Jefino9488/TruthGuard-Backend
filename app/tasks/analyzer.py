# flask_backend/app/tasks/analyzer.py

import os
import json
import logging
import sys
import random
from datetime import datetime, timezone
import pymongo
from google import genai
from google.genai import types  # Import types for Pydantic schema and GenerationConfig
from google.genai import errors  # Corrected import for API errors
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Pydantic models for structured JSON response (re-declared here for self-containment)
class FactCheck(BaseModel):
    claim: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str

class BiasAnalysis(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    political_leaning: str
    bias_indicators: list[str]
    language_bias: float = Field(ge=0.0, le=1.0)
    source_bias: float = Field(ge=0.0, le=1.0)
    framing_bias: float = Field(ge=0.0, le=1.0)

class MisinformationAnalysis(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    fact_checks: list[FactCheck]
    red_flags: list[str]

class SentimentAnalysis(BaseModel):
    overall_sentiment: float = Field(ge=-1.0, le=1.0)
    emotional_tone: str
    key_phrases: list[str]

class CredibilityAssessment(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    source_reliability: float = Field(ge=0.0, le=1.0)

class AnalysisResponse(BaseModel):
    bias_analysis: BiasAnalysis
    misinformation_analysis: MisinformationAnalysis
    sentiment_analysis: SentimentAnalysis
    credibility_assessment: CredibilityAssessment
    confidence: float = Field(ge=0.0, le=1.0)

class GeminiAnalyzerTask:
    def __init__(self, db_client, google_api_key, model_path='all-MiniLM-L6-v2'):
        self.db = db_client
        self.collection = self.db.articles

        # Configure Gemini client using genai.Client
        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not provided to GeminiAnalyzerTask")
        self.client = genai.Client(api_key=google_api_key) #
        self.model_name = 'gemini-2.0-flash-001' #

        logger.info(f"Loading sentence transformer model: {model_path}...")
        self.embedding_model = SentenceTransformer(model_path) #

        self.stats = {
            'articles_analyzed': 0,
            'high_bias_detected': 0,
            'misinformation_flagged': 0,
            'embeddings_generated': 0,
            'processing_errors': 0
        }

    def generate_embedding(self, text):
        """Generate vector embedding for text using sentence-transformers"""
        try:
            max_length = 10000
            if len(text) > max_length:
                text = text[:max_length]
            embedding = self.embedding_model.encode(text)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}") #
            return None

    def analyze_article_comprehensive(self, article, max_retries=3):
        """Comprehensive analysis using Gemini AI with retry logic"""
        for attempt in range(max_retries):
            try:
                prompt = f"""
                You are TruthGuard AI, an expert media bias and misinformation detection system.
                Analyze this news article comprehensively and return a JSON object strictly conforming to the AnalysisResponse Pydantic model.
                Ensure all fields are present and valid, especially for float ranges (0.0 to 1.0 or -1.0 to 1.0) and list types.
                Title: {article['title']}
                Source: {article['source']}
                Content: {article['content'][:8000]}
                """

                # Use client.models.generate_content and types.GenerateContentConfig
                response = self.client.models.generate_content( #
                    model=self.model_name, #
                    contents=prompt, #
                    config=types.GenerateContentConfig( #
                        response_mime_type='application/json', #
                        response_schema=AnalysisResponse.model_json_schema(), #
                        temperature=0.3, #
                        max_output_tokens=2000 #
                    )
                )

                try:
                    # Access the text attribute from the response object
                    analysis_data = json.loads(response.text) #
                    # Validate with Pydantic model
                    analysis = AnalysisResponse(**analysis_data) #
                    analysis_dict = analysis.model_dump() # Convert back to dict for MongoDB

                    update_fields = {
                        'ai_analysis': analysis_dict,
                        'bias_score': analysis_dict['bias_analysis']['overall_score'], #
                        'misinformation_risk': analysis_dict['misinformation_analysis']['risk_score'], #
                        'sentiment': analysis_dict['sentiment_analysis']['overall_sentiment'], #
                        'credibility_score': analysis_dict['credibility_assessment']['overall_score'], #
                        'processing_status': 'analyzed', #
                        'analyzed_at': datetime.now(timezone.utc), #
                        'analysis_model': self.model_name #
                    }

                    # Generate and update embeddings if not already present or needs regeneration
                    if 'content_embedding' not in article or article['content_embedding'] is None: #
                        content_embedding = self.generate_embedding(article['content']) #
                        if content_embedding:
                            update_fields['content_embedding'] = content_embedding #
                            self.stats['embeddings_generated'] += 1 #

                    if 'title_embedding' not in article or article['title_embedding'] is None: #
                        title_embedding = self.generate_embedding(article['title']) #
                        if title_embedding:
                            update_fields['title_embedding'] = title_embedding #
                            self.stats['embeddings_generated'] += 1 #

                    analysis_text = f"{analysis_dict['bias_analysis']['political_leaning']} {' '.join(analysis_dict['bias_analysis']['bias_indicators'])} {' '.join(analysis_dict['misinformation_analysis']['red_flags'])} {analysis_dict['sentiment_analysis']['emotional_tone']}" #
                    analysis_embedding = self.generate_embedding(analysis_text) #
                    if analysis_embedding:
                        update_fields['analysis_embedding'] = analysis_embedding #
                        self.stats['embeddings_generated'] += 1 #

                    self.collection.update_one(
                        {'_id': article['_id']},
                        {'$set': update_fields}
                    )

                    self.stats['articles_analyzed'] += 1 #
                    if analysis_dict['bias_analysis']['overall_score'] > 0.7:
                        self.stats['high_bias_detected'] += 1 #
                    if analysis_dict['misinformation_analysis']['risk_score'] > 0.6:
                        self.stats['misinformation_flagged'] += 1 #

                    logger.info(f"Analyzed: {article['title'][:50]}... ID: {article['_id']}") #
                    return analysis_dict

                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    logger.error(f"Failed to parse or validate Gemini response for article {article['_id']}: {e}. Raw response: {response.text}") #
                    self.stats['processing_errors'] += 1
                    # Attempt a fallback analysis if parsing/validation fails
                    return self.generate_fallback_analysis(article)

            except errors.APIError as e: #
                if e.status_code in [429, 503]: #
                    wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying article {article['_id']} after {wait_time:.2f}s due to {e.status_code} error: {e.message}") #
                    time.sleep(wait_time)
                    if attempt == max_retries - 1: #
                        logger.error(f"Max retries reached for article {article['_id']}: {e.status_code} - {e.message}") #
                        self.stats['processing_errors'] += 1
                        return self.generate_fallback_analysis(article)
                else: #
                    logger.error(f"Gemini API error for article {article['_id']}: {e.status_code} - {e.message}") #
                    self.stats['processing_errors'] += 1
                    return self.generate_fallback_analysis(article)
            except Exception as e: #
                logger.error(f"Unexpected error analyzing article {article['_id']}: {e}", exc_info=True)
                self.stats['processing_errors'] += 1
                return self.generate_fallback_analysis(article)
        return None

    def analyze_raw_content(self, title: str, content: str, max_retries=3):
        """
        Analyzes raw text content (not from MongoDB) using Gemini AI.
        Returns the analysis dictionary without attempting to update the database.
        """
        for attempt in range(max_retries):
            try:
                prompt = f"""
                You are TruthGuard AI, an expert media bias and misinformation detection system.
                Analyze this news content comprehensively and return a JSON object strictly conforming to the AnalysisResponse Pydantic model.
                Ensure all fields are present and valid, especially for float ranges (0.0 to 1.0 or -1.0 to 1.0) and list types.
                Title: {title}
                Content: {content[:8000]}
                """

                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                        response_schema=AnalysisResponse.model_json_schema(),
                        temperature=0.3,
                        max_output_tokens=2000
                    )
                )

                try:
                    analysis_data = json.loads(response.text)
                    analysis = AnalysisResponse(**analysis_data)
                    return analysis.model_dump()

                except (json.JSONDecodeError, ValueError, AttributeError) as e:
                    logger.error(f"Failed to parse or validate Gemini response for raw content: {e}. Raw response: {response.text}")
                    # Fallback to simple heuristic for raw content if Gemini fails
                    return self._generate_fallback_raw_analysis(title, content)

            except errors.APIError as e:
                if e.status_code in [429, 503]:
                    wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying raw content analysis after {wait_time:.2f}s due to {e.status_code} error: {e.message}")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for raw content analysis: {e.status_code} - {e.message}")
                        return self._generate_fallback_raw_analysis(title, content)
                else:
                    logger.error(f"Gemini API error for raw content analysis: {e.status_code} - {e.message}")
                    return self._generate_fallback_raw_analysis(title, content)
            except Exception as e:
                logger.error(f"Unexpected error analyzing raw content: {e}", exc_info=True)
                return self._generate_fallback_raw_analysis(title, content)
        return None

    def _generate_fallback_raw_analysis(self, title: str, content: str):
        """Generates a simple fallback analysis for raw content."""
        logger.info(f"Generating fallback analysis for raw content (Title: {title[:50]}...)")
        full_text = (title + " " + content).lower()
        bias_keywords = {
            'left': ['progressive', 'liberal', 'social justice', 'inequality', 'democrat'],
            'right': ['conservative', 'traditional', 'free market', 'law and order', 'republican']
        }

        left_score = sum(1 for word in bias_keywords['left'] if word in full_text)
        right_score = sum(1 for word in bias_keywords['right'] if word in full_text)

        bias_score = 0.0
        political_leaning = 'center'
        if left_score > right_score:
            bias_score = min(left_score / 5, 1.0)
            political_leaning = 'left-leaning'
        elif right_score > left_score:
            bias_score = min(right_score / 5, 1.0)
            political_leaning = 'right-leaning'

        return {
            'bias_analysis': {
                'overall_score': bias_score,
                'political_leaning': political_leaning,
                'bias_indicators': ["keyword_detection"] if bias_score > 0 else [],
                'language_bias': bias_score,
                'source_bias': 0.3,
                'framing_bias': bias_score * 0.8
            },
            'misinformation_analysis': {
                'risk_score': 0.3,
                'fact_checks': [],
                'red_flags': ["fallback_analysis_used_raw"]
            },
            'sentiment_analysis': {
                'overall_sentiment': 0.0,
                'emotional_tone': 'neutral',
                'key_phrases': []
            },
            'credibility_assessment': {
                'overall_score': 0.5,
                'evidence_quality': 0.4,
                'source_reliability': 0.5
            },
            'confidence': 0.2
        }

    def generate_fallback_analysis(self, article):
        """Generate fallback analysis when Gemini fails or Pydantic validation fails"""
        logger.info(f"Generating fallback analysis for article {article['_id']}") #
        content = article['content'].lower() #
        bias_keywords = {
            'left': ['progressive', 'liberal', 'social justice', 'inequality', 'democrat'],
            'right': ['conservative', 'traditional', 'free market', 'law and order', 'republican']
        }

        left_score = sum(1 for word in bias_keywords['left'] if word in content) #
        right_score = sum(1 for word in bias_keywords['right'] if word in content) #

        # Simple heuristic for bias score
        bias_score = 0.0 #
        political_leaning = 'center' #
        if left_score > right_score:
            bias_score = min(left_score / 5, 1.0) # Max 1.0 for simplified scoring
            political_leaning = 'left-leaning' #
        elif right_score > left_score:
            bias_score = min(right_score / 5, 1.0) #
            political_leaning = 'right-leaning' #

        analysis = {
            'bias_analysis': {
                'overall_score': bias_score, #
                'political_leaning': political_leaning, #
                'bias_indicators': ["keyword_detection"] if bias_score > 0 else [], #
                'language_bias': bias_score, #
                'source_bias': 0.3, #
                'framing_bias': bias_score * 0.8 #
            },
            'misinformation_analysis': {
                'risk_score': 0.3, #
                'fact_checks': [], #
                'red_flags': ["fallback_analysis_used"] #
            },
            'sentiment_analysis': {
                'overall_sentiment': 0.0, #
                'emotional_tone': 'neutral', #
                'key_phrases': [] #
            },
            'credibility_assessment': {
                'overall_score': 0.5, # Lower confidence for fallback
                'evidence_quality': 0.4, #
                'source_reliability': 0.5 #
            },
            'confidence': 0.2 # Low confidence for fallback
        }

        update_fields = {
            'ai_analysis': analysis,
            'bias_score': analysis['bias_analysis']['overall_score'], #
            'misinformation_risk': analysis['misinformation_analysis']['risk_score'], #
            'sentiment': analysis['sentiment_analysis']['overall_sentiment'], #
            'credibility_score': analysis['credibility_assessment']['overall_score'], #
            'processing_status': 'analyzed_fallback', #
            'analyzed_at': datetime.now(timezone.utc), #
            'analysis_model': 'fallback' #
        }

        # Still attempt to generate embeddings even for fallback
        if 'content_embedding' not in article or article['content_embedding'] is None: #
            content_embedding = self.generate_embedding(article['content']) #
            if content_embedding:
                update_fields['content_embedding'] = content_embedding #
                self.stats['embeddings_generated'] += 1 #

        if 'title_embedding' not in article or article['title_embedding'] is None: #
            title_embedding = self.generate_embedding(article['title']) #
            if title_embedding:
                update_fields['title_embedding'] = title_embedding #
                self.stats['embeddings_generated'] += 1 #

        analysis_text = f"{analysis['bias_analysis']['political_leaning']} {' '.join(analysis['bias_analysis']['bias_indicators'])} {' '.join(analysis['misinformation_analysis']['red_flags'])} {analysis['sentiment_analysis']['emotional_tone']}" #
        analysis_embedding = self.generate_embedding(analysis_text) #
        if analysis_embedding:
            update_fields['analysis_embedding'] = analysis_embedding #
            self.stats['embeddings_generated'] += 1 #

        try:
            self.collection.update_one(
                {'_id': article['_id']},
                {'$set': update_fields}
            ) #
            logger.info(f"Updated article {article['_id']} with fallback analysis.") #
        except Exception as e:
            logger.error(f"Error updating article {article['_id']} with fallback analysis: {e}") #

        return analysis

    def run_analyzer(self, batch_size=50):
        """Run analysis on unprocessed articles"""
        logger.info(f"Starting Gemini AI batch analysis task with batch size {batch_size}...") #

        unprocessed = list(self.collection.find({
            'processing_status': {'$in': ['pending', 'analyzed_fallback', None]} # Re-analyze fallbacks potentially
        }).limit(batch_size))

        if not unprocessed:
            logger.info("No unprocessed articles found to analyze.") #
            return self.stats

        logger.info(f"Found {len(unprocessed)} articles to analyze") #

        # Use ThreadPoolExecutor for concurrent API calls
        # Be mindful of Gemini API rate limits. max_workers should be set cautiously. #
        # A low number like 1-2 might be safer for free tiers or lower usage. #
        # The 'time.sleep(5)' in the loop also helps. #
        with ThreadPoolExecutor(max_workers=2) as executor: # Adjusted for safer API rate limiting
            future_to_article = {
                executor.submit(self.analyze_article_comprehensive, article): article
                for article in unprocessed
            }

            for future in as_completed(future_to_article):
                article = future_to_article[future] #
                try:
                    analysis = future.result() #
                    # Add a delay between processing each article's result to mitigate rate limiting
                    time.sleep(5) #
                except Exception as e:
                    logger.error(f"Analysis failed for {article['_id']}: {e}") #

        logger.info(f"Analysis complete. Stats: {self.stats}") #
        return self.stats