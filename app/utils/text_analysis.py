from typing import Dict, Any
import re
from textblob import TextBlob
import nltk
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
import asyncio

async def analyze_text_bias(text: str) -> Dict[str, Any]:
    """
    Analyzes text for potential bias using NLP techniques.
    """
    try:
        # Initialize NLTK data if not already downloaded
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
            nltk.download('stopwords')

        blob = TextBlob(text)

        # Analyze sentiment polarity (-1 to 1) and subjectivity (0 to 1)
        sentiment = blob.sentiment

        # Look for loaded language and emotional words
        emotional_words = _detect_emotional_language(text)

        # Check for extreme language patterns
        has_extreme_language = _check_extreme_language(text)

        # Calculate overall bias score
        bias_score = (abs(sentiment.polarity) + sentiment.subjectivity +
                     (0.3 if emotional_words else 0) +
                     (0.3 if has_extreme_language else 0)) / 2.6

        bias_type = _determine_bias_type(sentiment.polarity, emotional_words)

        return {
            "bias_detected": bias_score > 0.6,
            "bias_score": bias_score,
            "bias_type": bias_type,
            "confidence": bias_score * 100,
            "sentiment": {
                "polarity": sentiment.polarity,
                "subjectivity": sentiment.subjectivity
            },
            "analysis_details": {
                "emotional_language_detected": bool(emotional_words),
                "extreme_language_detected": has_extreme_language
            }
        }

    except Exception as e:
        return {
            "bias_detected": False,
            "error": str(e),
            "confidence": 0
        }

async def analyze_credibility(text: str) -> float:
    """
    Analyzes text credibility based on various factors.
    Returns a score from 0 to 10.
    """
    try:
        # Initialize factors
        score = 7.0  # Start with neutral-positive score

        # Check for citation patterns
        citations = _find_citations(text)
        score += len(citations) * 0.5  # Bonus for citations

        # Check for balanced language
        blob = TextBlob(text)
        if abs(blob.sentiment.polarity) > 0.7:
            score -= 1.0  # Penalty for extremely biased language

        # Check for fact-based language
        if _contains_fact_based_language(text):
            score += 1.0

        # Ensure score stays within bounds
        return max(0.0, min(10.0, score))

    except Exception:
        return 5.0  # Return neutral score on error

def _detect_emotional_language(text: str) -> bool:
    emotional_patterns = [
        r'\b(absolutely|definitely|obviously|clearly|undoubtedly)\b',
        r'[!]{2,}',
        r'\b(amazing|terrible|horrible|awesome|perfect|worst|best)\b',
    ]
    return any(re.search(pattern, text.lower()) for pattern in emotional_patterns)

def _check_extreme_language(text: str) -> bool:
    extreme_patterns = [
        r'\b(always|never|everyone|nobody|everything|nothing)\b',
        r'\b(must|need|have to|impossible|ridiculous)\b',
        r'\b(outrageous|scandalous|catastrophic)\b'
    ]
    return any(re.search(pattern, text.lower()) for pattern in extreme_patterns)

def _determine_bias_type(polarity: float, has_emotional_words: bool) -> str:
    if abs(polarity) < 0.2:
        return "Neutral"
    elif polarity > 0:
        return "Positive bias" if has_emotional_words else "Slight positive lean"
    else:
        return "Negative bias" if has_emotional_words else "Slight negative lean"

def _find_citations(text: str) -> list:
    citation_patterns = [
        r'(?:according to|cited by|source:)\s+[^.!?\n]+',
        r'\(\d{4}\)',
        r'\[[^\]]+\]'
    ]
    citations = []
    for pattern in citation_patterns:
        citations.extend(re.findall(pattern, text, re.IGNORECASE))
    return citations

def _contains_fact_based_language(text: str) -> bool:
    fact_patterns = [
        r'\b(?:research|study|survey|data|statistics|evidence|report)\b',
        r'\b(?:according to|based on|demonstrates|shows|indicates)\b',
        r'\b(?:measured|calculated|analyzed|investigated)\b'
    ]
    return any(re.search(pattern, text.lower()) for pattern in fact_patterns)
