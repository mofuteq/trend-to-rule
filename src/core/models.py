from typing import Literal
from pydantic import BaseModel


VERTICAL = Literal["mens", "womens", "unisex", "unknown"]


# Response Model
class SearchQuery(BaseModel):
    canonical_query: str
    emerging_query: str


class RequestGoal(BaseModel):
    request_goal: str
    is_in_scope: bool


class RequestAnalysis(BaseModel):
    request_goal: str
    candidate_queries: SearchQuery
    vertical: VERTICAL
    is_in_scope: bool


class ArticleAttribute(BaseModel):
    vertical: VERTICAL
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
