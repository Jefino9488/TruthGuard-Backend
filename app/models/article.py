
class Article(BaseModel):
    title: str
    url: str
    source: str
    author: Optional[str]
    published_at: datetime
    content: str
    summary: str
    bias_score: float
    credibility_score: float
    keywords: List[str]
    processed_at: datetime
    analysis_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Example News Article",
                "url": "https://example.com/news/1",
                "source": "Example News",
                "author": "John Doe",
                "published_at": "2025-06-11T10:00:00Z",
                "content": "Full article content...",
                "summary": "Brief summary of the article",
                "bias_score": 0.75,
                "credibility_score": 0.85,
                "keywords": ["politics", "technology", "science"],
                "processed_at": "2025-06-11T10:30:00Z"
            }
        }
