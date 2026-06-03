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


def merge_rankings_with_search(
    ranked_papers: list[RankedPaperSummary],
    search_papers: list[dict],
) -> list[RankedPaper]:
    """Combine LLM ranking scores with original search metadata."""
    merged = []
    for ranked in ranked_papers:
        # Find matching paper from search results by title
        match = None
        for paper in search_papers:
            if paper["title"].lower().strip() == ranked.title.lower().strip():
                match = paper
                break

        # Fuzzy fallback — check if title is contained
        if not match:
            for paper in search_papers:
                if (
                    ranked.title.lower()[:50] in paper["title"].lower()
                    or paper["title"].lower()[:50] in ranked.title.lower()
                ):
                    match = paper
                    break

        merged.append(
            RankedPaper(
                rank=ranked.rank,
                title=ranked.title,
                authors=match["authors"] if match else "Unknown",
                arxiv_url=match["arxiv_url"] if match else "#",
                pdf_url=match["pdf_url"] if match else "#",
                categories=match["categories"] if match else "",
                abstract=match["abstract"] if match else "",
                composite_score=ranked.composite_score,
                novelty_score=ranked.novelty_score,
                impact_score=ranked.impact_score,
                practicality_score=ranked.practicality_score,
                rationale=ranked.rationale,
            )
        )

    return merged
