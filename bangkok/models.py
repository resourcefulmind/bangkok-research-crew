from pydantic import BaseModel


# What the LLM produces — just scores and rationale
class RankedPaperSummary(BaseModel):
    rank: int
    title: str
    composite_score: float
    novelty_score: float
    impact_score: float
    practicality_score: float
    rationale: str


class RankingResult(BaseModel):
    papers: list[RankedPaperSummary]


# What the template needs — full data (built in Python by merging)
class RankedPaper(BaseModel):
    rank: int
    title: str
    authors: str
    arxiv_url: str
    pdf_url: str
    categories: str
    abstract: str
    composite_score: float
    novelty_score: float
    impact_score: float
    practicality_score: float
    rationale: str
