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
    def __init__(self, db_client, api_key, model_path='all-MiniLM-L6-v2'):
        if not db_client:
            raise ValueError("Database client cannot be None")

        self.db = db_client
        try:
            # Test the connection and get the collection
            self.db.command('ping')
            self.collection = self.db.articles
        except Exception as e:
            logger.error(f"Failed to initialize database connection: {e}")
            raise

        # Initialize Google AI
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')

        # Initialize sentence transformer for embeddings
        logger.info(f"Loading sentence transformer model: {model_path}...")
        self.embedding_model = SentenceTransformer(model_path)

    def analyze_raw_content(self, title: str, content: str) -> dict:
        """Analyze the raw content without storing in the database."""
        try:
            # Prepare the prompt
            analysis_prompt = f"""
            Please analyze the following article:
            Title: {title}
            Content: {content}

            Provide a comprehensive analysis in the following JSON structure:
            {AnalysisResponse.schema_json(indent=2)}

            Ensure all scores are between 0 and 1, and the response strictly follows the schema.
            """

            # Generate response from Gemini
            response = self.model.generate_content(analysis_prompt)

            if not response.text:
                raise ValueError("Empty response from Gemini")

            # Parse the response
            try:
                analysis_dict = json.loads(response.text)
                analysis = AnalysisResponse(**analysis_dict)

                # Generate embeddings
                title_embedding = self.generate_embedding(title)
                content_embedding = self.generate_embedding(content)

                # Combine the analysis with embeddings
                result = {
                    "analysis": analysis.model_dump(),
                    "embeddings": {
                        "title": title_embedding.tolist() if title_embedding is not None else None,
                        "content": content_embedding.tolist() if content_embedding is not None else None
                    }
                }

                return result
            except ValidationError as e:
                logger.error(f"Failed to validate Gemini response: {e}")
                raise ValueError(f"Invalid analysis format: {e}")
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Gemini response: {e}")
                raise ValueError(f"Invalid JSON response: {e}")

        except Exception as e:
            logger.error(f"Error in analyze_raw_content: {e}", exc_info=True)
            return None

    def generate_embedding(self, text: str) -> np.ndarray:
        """Generate vector embedding for text using sentence-transformers."""
        try:
            if not text:
                return None

            # Maximum input length for all-MiniLM-L6-v2 is 256 tokens
            # Truncate to ensure it fits
            max_char_length = 500  # Using 500 characters as a rough safe limit
            if len(text) > max_char_length:
                text = text[:max_char_length]

            embedding = self.embedding_model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return None

    def run_analyzer(self, batch_size: int = 10):
        """Process a batch of pending articles."""
        try:
            # Find articles pending analysis
            pending_articles = self.collection.find(
                {"processing_status": "pending"},
                limit=batch_size
            ).sort("published_at", pymongo.DESCENDING)

            processed_count = 0
            for article in pending_articles:
                try:
                    analysis_result = self.analyze_raw_content(
                        title=article.get("title", ""),
                        content=article.get("content", "")
                    )

                    if analysis_result:
                        # Update the article with analysis results
                        self.collection.update_one(
                            {"_id": article["_id"]},
                            {
                                "$set": {
                                    "analysis": analysis_result["analysis"],
                                    "title_embedding": analysis_result["embeddings"]["title"],
                                    "content_embedding": analysis_result["embeddings"]["content"],
                                    "processing_status": "completed",
                                    "analyzed_at": datetime.now(timezone.utc)
                                }
                            }
                        )
                        processed_count += 1
                    else:
                        # Mark as failed if analysis returns None
                        self.collection.update_one(
                            {"_id": article["_id"]},
                            {
                                "$set": {
                                    "processing_status": "failed",
                                    "error": "Analysis failed to produce results"
                                }
                            }
                        )

                except Exception as e:
                    logger.error(f"Error processing article {article.get('_id')}: {e}")
                    # Mark individual article as failed
                    self.collection.update_one(
                        {"_id": article["_id"]},
                        {
                            "$set": {
                                "processing_status": "failed",
                                "error": str(e)
                            }
                        }
                    )

            return processed_count

        except Exception as e:
            logger.error(f"Error in run_analyzer: {e}", exc_info=True)
            return 0
