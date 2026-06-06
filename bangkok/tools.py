from typing import Callable, ClassVar, Optional

import arxiv
from crewai.tools import BaseTool


def format_papers_for_eval(papers: list[dict]) -> str:
    """Format the search results as a text block for the evaluator prompts.

    This is the data the novelty/impact/practicality evaluators read. Pulled
    out of the tool so the pipeline can build it directly from the paper list
    (no LLM round-trip needed to relay the search results).
    """
    blocks = [f"Found {len(papers)} papers.\n"]
    for i, p in enumerate(papers, 1):
        blocks.append(
            f"Paper {i}:\n"
            f"  Title: {p['title']}\n"
            f"  Authors: {p['authors']}\n"
            f"  Abstract: {p['abstract'][:500]}\n"
            f"  ArXiv URL: {p['arxiv_url']}\n"
            f"  PDF URL: {p['pdf_url']}\n"
            f"  Categories: {p['categories']}\n"
        )
    return "\n".join(blocks)


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Searches ArXiv for research papers published on a specific date "
        "in given categories. Input should be a date string in YYYY-MM-DD "
        "format followed by a comma-separated list of categories. "
        "Example: '2026-04-01, cs.AI, cs.LG, cs.CL'"
    )

    # Store results so the pipeline can access them after the crew runs.
    # ClassVar (not a Pydantic field) so it's always a real class attribute —
    # accessible even if the search fails before any assignment happens.
    last_results: ClassVar[list] = []

    # Optional status reporter the web pipeline injects so the tool can surface
    # live progress (querying / found) to the dashboard. None under the CLI.
    progress: ClassVar[Optional[Callable[[str], None]]] = None

    @classmethod
    def report(cls, message: str) -> None:
        if cls.progress:
            cls.progress(message)

    def _run(self, query: str) -> str:
        # Parse the input — expects "date, cat1, cat2, ..."
        parts = [p.strip() for p in query.split(",")]
        date_str = parts[0]  # e.g. "2026-04-01"
        categories = parts[1:] if len(parts) > 1 else ["cs.AI"]

        # Build date range from the date string
        # Remove dashes: "2026-04-01" -> "20260401"
        date_clean = date_str.replace("-", "")
        # Next day for the range end
        date_int = int(date_clean)
        next_date = str(date_int + 1)

        # Build one combined query
        cat_query = " OR ".join([f"cat:{cat}" for cat in categories])
        full_query = (
            f"({cat_query}) AND "
            f"submittedDate:[{date_clean} TO {next_date}]"
        )

        client = arxiv.Client(
            page_size=25,
            delay_seconds=5.0,
            num_retries=0,   # fail fast: retrying after a 429 is what escalates to a multi-day block
        )
        # arXiv asks callers to identify themselves; the default requests UA reads as a scraper.
        client._session.headers.update({
            "User-Agent": "bangkok-research-crew/1.0 (+https://github.com/resourcefulmind/bangkok-research-crew)"
        })

        search = arxiv.Search(
            query=full_query,
            max_results=25,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        self.report(f"Querying arXiv for {date_str}…")

        papers = []
        try:
            for result in client.results(search):
                paper = {
                    "title": result.title,
                    "authors": ", ".join([a.name for a in result.authors]),
                    "abstract": result.summary,
                    "arxiv_url": result.entry_id,
                    "pdf_url": str(result.pdf_url),
                    "categories": ", ".join(result.categories),
                    "published": str(result.published),
                }
                papers.append(paper)
        except Exception as e:
            if "429" in str(e):
                raise RuntimeError(
                    "arXiv is rate-limiting us (HTTP 429). Stop and wait before "
                    "retrying; repeated calls only deepen the block."
                ) from e
            raise

        # Save for later use in main.py
        ArxivSearchTool.last_results = papers
        self.report(f"Found {len(papers)} papers")

        if not papers:
            return f"No papers found for query: {full_query}"

        return format_papers_for_eval(papers)
