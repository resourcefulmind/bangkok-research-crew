import arxiv
from crewai.tools import BaseTool


class ArxivSearchTool(BaseTool):
    name: str = "arxiv_search"
    description: str = (
        "Searches ArXiv for research papers published on a specific date "
        "in given categories. Input should be a date string in YYYY-MM-DD "
        "format followed by a comma-separated list of categories. "
        "Example: '2026-04-01, cs.AI, cs.LG, cs.CL'"
    )

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
            page_size=50,
            delay_seconds=5.0,
            num_retries=5,
        )

        search = arxiv.Search(
            query=full_query,
            max_results=50,
            sort_by=arxiv.SortCriterion.SubmittedDate,
            sort_order=arxiv.SortOrder.Descending,
        )

        papers = []
        for result in client.results(search):
            paper = {
                "title": result.title,
                "authors": ", ".join([a.name for a in result.authors]),
                "abstract": result.summary,
                "arxiv_url": result.entry_id,
                "pdf_url": result.pdf_url,
                "categories": ", ".join(result.categories),
                "published": str(result.published),
            }
            papers.append(paper)

        if not papers:
            return f"No papers found for query: {full_query}"

        output = f"Found {len(papers)} papers for {date_str} in categories: {', '.join(categories)}\n\n"
        for i, p in enumerate(papers, 1):
            output += (
                f"Paper {i}:\n"
                f"  Title: {p['title']}\n"
                f"  Authors: {p['authors']}\n"
                f"  Abstract: {p['abstract'][:500]}\n"
                f"  ArXiv URL: {p['arxiv_url']}\n"
                f"  PDF URL: {p['pdf_url']}\n"
                f"  Categories: {p['categories']}\n"
                f"  Published: {p['published']}\n\n"
            )

        return output
