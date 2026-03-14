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
    canditate_queries: SearchQuery
    reason: str


class ArticleAttribute(BaseModel):
    vertical: Literal["mens", "womens", "unisex", "unknown"]
    language: str


class StructuredDraft(BaseModel):
    theme: str
    canonical: list[str]
    emerging: list[str]
    conflicts: list[str]
    gaps: list[str]
    common_rule: list[str]
