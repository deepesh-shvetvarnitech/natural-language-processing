from pydantic import BaseModel, Field

class ReviewRequest(BaseModel):
    review: str

class KeywordRequest(BaseModel):
    review: str
    top_n: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Kitne keywords return karne hain"
    )

class SimilarRequest(BaseModel):
    review: str
    top_k: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Kitne similar reviews return karne hain"
    )

class AnalyzeRequest(BaseModel):
    review: str
    top_keywords: int = Field(
        default=5,
        ge=1,
        le=20,
        description="Kitne keywords return karne hain"
    )
    similar_reviews: int = Field(
        default=3,
        ge=1,
        le=20,
        description="Kitne similar reviews return karne hain"
    )
