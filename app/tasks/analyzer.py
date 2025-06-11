import json
import logging
import sys
import random
from datetime import datetime, timezone
import pymongo
from google import genai
from google.genai import types
from google.genai import errors
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from sentence_transformers import SentenceTransformer
import numpy as np
from pydantic import BaseModel, Field, ValidationError, ConfigDict

logger = logging.getLogger(__name__)

# Pydantic models for structured JSON response
class FactCheck(BaseModel):
    claim: str
    verdict: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str
    sources: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

class BiasAnalysis(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    political_leaning: str
    bias_indicators: list[str]
    language_bias: float = Field(ge=0.0, le=1.0)
    source_bias: float = Field(ge=0.0, le=1.0)
    framing_bias: float = Field(ge=0.0, le=1.0)
    selection_bias: float = Field(ge=0.0, le=1.0)
    confirmation_bias: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="forbid")

class MisinformationAnalysis(BaseModel):
    risk_score: float = Field(ge=0.0, le=1.0)
    fact_checks: list[FactCheck] = Field(default_factory=list)
    red_flags: list[str] = Field(default_factory=list)
    logical_fallacies: list[str] = Field(default_factory=list)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="forbid")

class SentimentAnalysis(BaseModel):
    overall_sentiment: float = Field(ge=-1.0, le=1.0)
    emotional_tone: str
    key_phrases: list[str] = Field(default_factory=list)
    emotional_manipulation: float = Field(ge=0.0, le=1.0)
    subjectivity_score: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="forbid")

class CredibilityAssessment(BaseModel):
    overall_score: float = Field(ge=0.0, le=1.0)
    evidence_quality: float = Field(ge=0.0, le=1.0)
    source_reliability: float = Field(ge=0.0, le=1.0)
    logical_consistency: float = Field(ge=0.0, le=1.0)
    transparency: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="forbid")

class ActorPortrayal(BaseModel):
    main_actor: str | None = None
    tone: str | None = None
    role: str | None = None

class NarrativeAnalysis(BaseModel):
    primary_frame: str
    secondary_frames: list[str] = Field(default_factory=list)
    narrative_patterns: list[str] = Field(default_factory=list)
    actor_portrayal: ActorPortrayal = Field(default_factory=ActorPortrayal)
    perspective_diversity: float = Field(ge=0.0, le=1.0)
    model_config = ConfigDict(extra="forbid")

class TechnicalAnalysis(BaseModel):
    readability_score: float = Field(ge=0.0, le=1.0)
    complexity_level: str
    word_count: int
    key_topics: list[str] = Field(default_factory=list)
    named_entities: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

class Recommendations(BaseModel):
    verification_needed: list[str] = Field(default_factory=list)
    alternative_sources: list[str] = Field(default_factory=list)
    critical_questions: list[str] = Field(default_factory=list)
    bias_mitigation: list[str] = Field(default_factory=list)
    model_config = ConfigDict(extra="forbid")

class AnalysisResponse(BaseModel):
    bias_analysis: BiasAnalysis
    misinformation_analysis: MisinformationAnalysis
    sentiment_analysis: SentimentAnalysis
    credibility_assessment: CredibilityAssessment
    narrative_analysis: NarrativeAnalysis
    technical_analysis: TechnicalAnalysis
    recommendations: Recommendations
    confidence: float = Field(ge=0.0, le=1.0)
    model_version: str = Field(default="gemini-2.0-flash-001")
    model_config = ConfigDict(extra="forbid")

def clean_json_schema(schema: dict) -> dict:
    """Recursively remove 'additionalProperties' from a JSON schema."""
    if not isinstance(schema, dict):
        return schema

    cleaned = {}
    for key, value in schema.items():
        if key == "additionalProperties":
            continue  # Skip this key entirely
        elif isinstance(value, dict):
            cleaned[key] = clean_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [clean_json_schema(item) for item in value]
        else:
            cleaned[key] = value
    return cleaned



class GeminiAnalyzerTask:
    def __init__(self, db_client, google_api_key, model_path='all-MiniLM-L6-v2'):
        self.db = db_client
        self.collection = self.db.articles

        if not google_api_key:
            raise ValueError("GOOGLE_API_KEY is not provided to GeminiAnalyzerTask")
        self.client = genai.Client(api_key=google_api_key)
        self.model_name = 'gemini-2.0-flash-001' # Using a specific model for consistency

        logger.info(f"Loading sentence transformer model: {model_path}...")
        self.embedding_model = SentenceTransformer(model_path)

        self.stats = {
            'articles_analyzed': 0,
            'high_bias_detected': 0,
            'misinformation_flagged': 0,
            'embeddings_generated': 0,
            'processing_errors': 0
        }

    def generate_embedding(self, text):
        """Generate vector embedding for text using sentence-transformers."""
        try:
            # Maximum input length for all-MiniLM-L6-v2 is 256 tokens.
            # Truncate to ensure it fits, or use a model with larger context.
            # Using 500 characters as a rough safe limit, can be adjusted.
            max_char_length = 500
            if len(text) > max_char_length:
                text = text[:max_char_length]
            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def _prepare_gemini_prompt(self, article_title: str, article_content: str):
        """Prepares the prompt for Gemini API call."""
        # Truncate content to avoid exceeding Gemini's context window
        content_to_send = article_content[:7000] # Adjust as needed for Gemini model context
        title_to_send = article_title[:500]

        return f"""
        You are TruthGuard AI, an expert media bias and misinformation detection system.
        Analyze the provided news article content comprehensively and return a JSON object strictly conforming to the AnalysisResponse Pydantic model.
        Ensure all nested fields are present and valid, especially for float ranges (0.0 to 1.0 or -1.0 to 1.0) and list types.
        If a specific sub-field's data is not clearly deducible, provide a neutral or default value within its valid range (e.g., 0.5 for scores, "neutral" for tones).
        Always provide at least one item for lists if the context allows, even if it's a general observation like "lack of specific bias indicators".
        Ensure the 'model_version' field in the response matches '{self.model_name}'.

        Article Title: {title_to_send}
        Article Content: {content_to_send}

        Your JSON response must adhere to the following schema:
        {json.dumps(clean_json_schema(AnalysisResponse.model_json_schema()))}
        """

    def analyze_article_comprehensive(self, article, max_retries=3):
        """Comprehensive analysis using Gemini AI with retry logic."""
        prompt = self._prepare_gemini_prompt(article['title'], article['content'])
        raw_schema = AnalysisResponse.model_json_schema()
        cleaned_schema = clean_json_schema(raw_schema)

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                        response_schema=cleaned_schema,
                        temperature=0.3,
                        max_output_tokens=3000
                    )
                )

                try:
                    analysis_data = json.loads(response.text)
                    # Validate with Pydantic model
                    analysis = AnalysisResponse(**analysis_data)
                    analysis_dict = analysis.model_dump()

                    # Always ensure model_version is correctly set
                    analysis_dict['model_version'] = self.model_name

                    update_fields = {
                        'ai_analysis': analysis_dict,
                        'bias_score': analysis_dict['bias_analysis']['overall_score'],
                        'misinformation_risk': analysis_dict['misinformation_analysis']['risk_score'],
                        'sentiment': analysis_dict['sentiment_analysis']['overall_sentiment'],
                        'credibility_score': analysis_dict['credibility_assessment']['overall_score'],
                        'processing_status': 'analyzed',
                        'analyzed_at': datetime.now(timezone.utc),
                        'analysis_model': self.model_name
                    }

                    # Generate and update embeddings
                    content_embedding = self.generate_embedding(article['content'])
                    if content_embedding:
                        update_fields['content_embedding'] = content_embedding
                        self.stats['embeddings_generated'] += 1

                    title_embedding = self.generate_embedding(article['title'])
                    if title_embedding:
                        update_fields['title_embedding'] = title_embedding
                        self.stats['embeddings_generated'] += 1

                    # Combine key analysis parts for a general analysis embedding
                    analysis_text_summary = (
                        f"{analysis_dict['bias_analysis']['political_leaning']} "
                        f"{' '.join(analysis_dict['bias_analysis'].get('bias_indicators', []))} "
                        f"{' '.join(analysis_dict['misinformation_analysis'].get('red_flags', []))} "
                        f"{analysis_dict['sentiment_analysis'].get('emotional_tone', '')}"
                    )
                    analysis_embedding = self.generate_embedding(analysis_text_summary)
                    if analysis_embedding:
                        update_fields['analysis_embedding'] = analysis_embedding
                        self.stats['embeddings_generated'] += 1

                    self.collection.update_one(
                        {'_id': article['_id']},
                        {'$set': update_fields}
                    )

                    self.stats['articles_analyzed'] += 1
                    if analysis_dict['bias_analysis']['overall_score'] > 0.7:
                        self.stats['high_bias_detected'] += 1
                    if analysis_dict['misinformation_analysis']['risk_score'] > 0.6:
                        self.stats['misinformation_flagged'] += 1

                    logger.info(f"Analyzed: {article['title'][:50]}... ID: {article['_id']}")
                    return analysis_dict

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Failed to parse or validate Gemini response for article {article['_id']}: {e}. Raw response: {response.text}")
                    self.stats['processing_errors'] += 1
                    return self.generate_fallback_analysis(article)
                except AttributeError as e:
                    logger.error(f"Attribute error in Gemini response for article {article['_id']}: {e}. Response text structure might be unexpected: {response.text}")
                    self.stats['processing_errors'] += 1
                    return self.generate_fallback_analysis(article)

            except errors.APIError as e:
                if hasattr(e, 'response') and e.response.status_code in [429, 503]:
                    wait_time = 5 * (2 ** attempt) + random.uniform(0, 1)
                    logger.warning(f"Retrying article {article['_id']} after {wait_time:.2f}s due to {e.status_code} error: {e.message}")
                    time.sleep(wait_time)
                    if attempt == max_retries - 1:
                        logger.error(f"Max retries reached for article {article['_id']}: {e.status_code} - {e.message}")
                        self.stats['processing_errors'] += 1
                        return self.generate_fallback_analysis(article)
                else:
                    logger.error(f"Gemini API error for article {article['_id']}: {e.status_code} - {e.message}")
                    self.stats['processing_errors'] += 1
                    return self.generate_fallback_analysis(article)
            except Exception as e:
                logger.error(f"Unexpected error analyzing article {article['_id']}: {e}", exc_info=True)
                self.stats['processing_errors'] += 1
                return self.generate_fallback_analysis(article)
        return None

    def analyze_raw_content(self, title: str, content: str, max_retries=3):
        """Analyzes raw text content (not from MongoDB) using Gemini AI."""
        prompt = self._prepare_gemini_prompt(title, content)
        raw_schema = AnalysisResponse.model_json_schema()
        cleaned_schema = clean_json_schema(raw_schema)

        for attempt in range(max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        response_mime_type='application/json',
                        response_schema=cleaned_schema,  # Use cleaned schema
                        temperature=0.3,
                        max_output_tokens=3000
                    )
                )
                try:
                    analysis_data = json.loads(response.text)
                    analysis = AnalysisResponse(**analysis_data)
                    analysis_dict = analysis.model_dump()

                    # Always ensure model_version is correctly set
                    analysis_dict['model_version'] = self.model_name

                    # Generate embeddings for the raw content and analysis summary
                    analysis_dict['content_embedding'] = self.generate_embedding(content)
                    analysis_dict['title_embedding'] = self.generate_embedding(title)
                    analysis_text_summary = (
                        f"{analysis_dict['bias_analysis'].get('political_leaning', '')} "
                        f"{' '.join(analysis_dict['bias_analysis'].get('bias_indicators', []))} "
                        f"{' '.join(analysis_dict['misinformation_analysis'].get('red_flags', []))} "
                        f"{analysis_dict['sentiment_analysis'].get('emotional_tone', '')}"
                    )
                    analysis_dict['analysis_embedding'] = self.generate_embedding(analysis_text_summary)

                    return analysis_dict

                except (json.JSONDecodeError, ValidationError) as e:
                    logger.error(f"Failed to parse or validate Gemini response for raw content: {e}. Raw response: {response.text}")
                    return self._generate_fallback_raw_analysis(title, content)
                except AttributeError as e:
                    logger.error(f"Attribute error in Gemini response for raw content: {e}. Response text structure might be unexpected: {response.text}")
                    return self._generate_fallback_raw_analysis(title, content)

            except errors.APIError as e:
                if hasattr(e, 'response') and e.response.status_code in [429, 503]:
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
        """Generates a simple fallback analysis for raw content, including dummy embeddings."""
        logger.info(f"Generating fallback analysis for raw content (Title: {title[:50]}...)")
        full_text = (title + " " + content).lower()
        bias_keywords = {
            'left': ['progressive', 'liberal', 'social justice', 'inequality', 'democrat'],
            'right': ['conservative', 'traditional', 'free market', 'law and order', 'republican']
        }

        left_score_count = sum(1 for word in bias_keywords['left'] if word in full_text)
        right_score_count = sum(1 for word in bias_keywords['right'] if word in full_text)

        bias_score_val = 0.5
        political_leaning_val = 'center'
        if left_score_count > right_score_count:
            bias_score_val = min(left_score_count * 0.1 + 0.3, 1.0)
            political_leaning_val = 'left-leaning'
        elif right_score_count > left_score_count:
            bias_score_val = min(right_score_count * 0.1 + 0.3, 1.0)
            political_leaning_val = 'right-leaning'

        # Dummy embeddings for fallback, ensure they are lists of numbers
        dummy_embedding = np.random.rand(self.embedding_model.get_sentence_embedding_dimension()).tolist()

        return {
            'bias_analysis': {
                'overall_score': bias_score_val,
                'political_leaning': political_leaning_val,
                'bias_indicators': ["keyword_detection", "fallback_heuristic"],
                'language_bias': round(random.uniform(0.3, 0.7), 2),
                'source_bias': round(random.uniform(0.3, 0.7), 2),
                'framing_bias': round(random.uniform(0.3, 0.7), 2),
                'selection_bias': round(random.uniform(0.3, 0.7), 2),
                'confirmation_bias': round(random.uniform(0.3, 0.7), 2)
            },
            'misinformation_analysis': {
                'risk_score': round(random.uniform(0.2, 0.5), 2),
                'fact_checks': [],
                'red_flags': ["unverified_content", "fallback_analysis"],
                'logical_fallacies': [],
                'evidence_quality': round(random.uniform(0.4, 0.7), 2)
            },
            'sentiment_analysis': {
                'overall_sentiment': round(random.uniform(-0.3, 0.3), 2),
                'emotional_tone': 'neutral',
                'key_phrases': [],
                'emotional_manipulation': round(random.uniform(0.1, 0.4), 2),
                'subjectivity_score': round(random.uniform(0.3, 0.6), 2)
            },
            'credibility_assessment': {
                'overall_score': round(random.uniform(0.4, 0.6), 2),
                'evidence_quality': round(random.uniform(0.4, 0.6), 2),
                'source_reliability': round(random.uniform(0.4, 0.6), 2),
                'logical_consistency': round(random.uniform(0.4, 0.6), 2),
                'transparency': round(random.uniform(0.4, 0.6), 2)
            },
            'narrative_analysis': {
                'primary_frame': 'informational',
                'secondary_frames': [],
                'narrative_patterns': [],
                'actor_portrayal': {},
                'perspective_diversity': round(random.uniform(0.3, 0.7), 2)
            },
            'technical_analysis': {
                'readability_score': round(random.uniform(0.5, 0.9), 2),
                'complexity_level': 'moderate',
                'word_count': len(content.split()),
                'key_topics': ['general'],
                'named_entities': []
            },
            'recommendations': {
                'verification_needed': ["manual_review_recommended"],
                'alternative_sources': [],
                'critical_questions': [],
                'bias_mitigation': []
            },
            'confidence': 0.2,
            'model_version': 'fallback-local-ai-v1.0',
            'content_embedding': dummy_embedding,
            'title_embedding': dummy_embedding,
            'analysis_embedding': dummy_embedding
        }

    def generate_fallback_analysis(self, article):
        """Generate fallback analysis for a stored article when Gemini fails or Pydantic validation fails."""
        logger.info(f"Generating fallback analysis for article {article['_id']}")
        content = article['content'].lower()

        # Simple heuristic for bias score
        bias_score_val = random.uniform(0.4, 0.6)
        political_leaning_val = random.choice(['center', 'left-leaning', 'right-leaning'])

        # Dummy embeddings for fallback, ensure they are lists of numbers
        dummy_embedding = np.random.rand(self.embedding_model.get_sentence_embedding_dimension()).tolist()

        analysis_dict = {
            'bias_analysis': {
                'overall_score': bias_score_val,
                'political_leaning': political_leaning_val,
                'bias_indicators': ["fallback_heuristic"],
                'language_bias': round(random.uniform(0.3, 0.7), 2),
                'source_bias': round(random.uniform(0.3, 0.7), 2),
                'framing_bias': round(random.uniform(0.3, 0.7), 2),
                'selection_bias': round(random.uniform(0.3, 0.7), 2),
                'confirmation_bias': round(random.uniform(0.3, 0.7), 2)
            },
            'misinformation_analysis': {
                'risk_score': round(random.uniform(0.2, 0.5), 2),
                'fact_checks': [],
                'red_flags': ["fallback_analysis"],
                'logical_fallacies': [],
                'evidence_quality': round(random.uniform(0.4, 0.7), 2)
            },
            'sentiment_analysis': {
                'overall_sentiment': round(random.uniform(-0.3, 0.3), 2),
                'emotional_tone': 'neutral',
                'key_phrases': [],
                'emotional_manipulation': round(random.uniform(0.1, 0.4), 2),
                'subjectivity_score': round(random.uniform(0.3, 0.6), 2)
            },
            'credibility_assessment': {
                'overall_score': round(random.uniform(0.4, 0.6), 2),
                'evidence_quality': round(random.uniform(0.4, 0.6), 2),
                'source_reliability': round(random.uniform(0.4, 0.6), 2),
                'logical_consistency': round(random.uniform(0.4, 0.6), 2),
                'transparency': round(random.uniform(0.4, 0.6), 2)
            },
            'narrative_analysis': {
                'primary_frame': 'informational',
                'secondary_frames': [],
                'narrative_patterns': [],
                'actor_portrayal': {},
                'perspective_diversity': round(random.uniform(0.3, 0.7), 2)
            },
            'technical_analysis': {
                'readability_score': round(random.uniform(0.5, 0.9), 2),
                'complexity_level': 'moderate',
                'word_count': len(content.split()),
                'key_topics': ['general'],
                'named_entities': []
            },
            'recommendations': {
                'verification_needed': ["manual_review_recommended"],
                'alternative_sources': [],
                'critical_questions': [],
                'bias_mitigation': []
            },
            'confidence': 0.2,
            'model_version': 'fallback-local-ai-v1.0'
        }

        update_fields = {
            'ai_analysis': analysis_dict,
            'bias_score': analysis_dict['bias_analysis']['overall_score'],
            'misinformation_risk': analysis_dict['misinformation_analysis']['risk_score'],
            'sentiment': analysis_dict['sentiment_analysis']['overall_sentiment'],
            'credibility_score': analysis_dict['credibility_assessment']['overall_score'],
            'processing_status': 'analyzed_fallback',
            'analyzed_at': datetime.now(timezone.utc),
            'analysis_model': 'fallback',
            'content_embedding': dummy_embedding,
            'title_embedding': dummy_embedding,
            'analysis_embedding': dummy_embedding
        }

        try:
            self.collection.update_one(
                {'_id': article['_id']},
                {'$set': update_fields}
            )
            logger.info(f"Updated article {article['_id']} with fallback analysis.")
        except Exception as e:
            logger.error(f"Error updating article {article['_id']} with fallback analysis: {e}")

        return analysis_dict

    def run_analyzer(self, batch_size=50):
        """Run analysis on unprocessed articles."""
        logger.info(f"Starting Gemini AI batch analysis task with batch size {batch_size}...")

        unprocessed = list(self.collection.find({
            'processing_status': {'$in': ['pending', 'analyzed_fallback', None]}
        })) # Removed .limit(batch_size)

        if not unprocessed:
            logger.info("No unprocessed articles found to analyze.")
            return self.stats

        logger.info(f"Found {len(unprocessed)} articles to analyze")

        # Use ThreadPoolExecutor for concurrent API calls
        with ThreadPoolExecutor(max_workers=2) as executor:
            future_to_article = {
                executor.submit(self.analyze_article_comprehensive, article): article
                for article in unprocessed
            }

            for future in as_completed(future_to_article):
                article = future_to_article[future]
                try:
                    _ = future.result()
                    time.sleep(5)
                except Exception as e:
                    logger.error(f"Analysis failed for {article['_id']}: {e}")

        logger.info(f"Analysis complete. Stats: {self.stats}")
        return self.stats