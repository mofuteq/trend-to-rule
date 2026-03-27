from typing import Literal
from pydantic import BaseModel


# Response Model
class SearchQuery(BaseModel):
    canonical_query: str
    emerging_query: str


class UserGoal(BaseModel):
    user_goal: str
    reason: str


class UserNeeds(BaseModel):
    user_goal: str
    candidate_queries: SearchQuery
    reason: str


class ArticleAttribute(BaseModel):
    vertical: Literal["mens", "womens", "unisex", "unknown"]
    language: str


class Claim(BaseModel):
    claim: str
    claim_type: Literal["observation", "interpretation", "norm", "signal"]
    source_id: str


class StructuredClaims(BaseModel):
    canonical_claims: list[Claim]
    emerging_claims: list[Claim]


class StructuredDraft(BaseModel):
    theme: str
    canonical: list[str]
    emerging: list[str]
    conflicts: list[str]
    gaps: list[str]
    common_rule: list[str]


class ExampleQuerySpec(BaseModel):
    context: str | None = None
    audience: str | None = None
    color: str | None = None
    material: str | None = None
    silhouette: str | None = None
    item: str
    vibe: str | None = None
