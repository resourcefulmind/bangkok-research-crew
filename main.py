import argparse
from dotenv import load_dotenv
from crewai import Crew, Process
from tasks import (
    search_task,
    novelty_task,
    impact_task,
    practical_task,
    ranking_task,
)
from models import RankedPaper
from tools import ArxivSearchTool
from render import render_report

load_dotenv()

crew = Crew(
    agents=[
        search_task.agent,
        novelty_task.agent,
        impact_task.agent,
        practical_task.agent,
        ranking_task.agent,
    ],
    tasks=[
        search_task,
        novelty_task,
        impact_task,
        practical_task,
        ranking_task,
    ],
    process=Process.sequential,
    verbose=True,
)


def merge_rankings_with_search(ranking_result, search_papers):
    """Combine LLM ranking scores with original search metadata."""
    merged = []
    for ranked in ranking_result.papers:
        # Find matching paper from search results by title
        match = None
        for paper in search_papers:
            if paper["title"].lower().strip() == ranked.title.lower().strip():
                match = paper
                break

        # Fuzzy fallback — check if title is contained
        if not match:
            for paper in search_papers:
                if (ranked.title.lower()[:50] in paper["title"].lower()
                        or paper["title"].lower()[:50] in ranked.title.lower()):
                    match = paper
                    break

        merged.append(RankedPaper(
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
        ))

    return merged


def main():
    parser = argparse.ArgumentParser(
        description="Search ArXiv for AI research papers, rank the top 10 and generate a report"
    )
    parser.add_argument(
        "--date",
        type=str,
        required=True,
        help="The date to search for papers in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--categories",
        type=str,
        help="ArXiv categories to search (default: cs.AI cs.LG cs.CL cs.CV stat.ML)",
        nargs="+",
        default=["cs.AI", "cs.LG", "cs.CL", "cs.CV", "stat.ML"],
    )
    args = parser.parse_args()

    categories_str = ", ".join(args.categories)

    print(f"\nSearching ArXiv for papers on {args.date}")
    print(f"Categories: {categories_str}\n")

    # Run the crew
    result = crew.kickoff(
        inputs={
            "date": args.date,
            "categories": categories_str,
        }
    )

    # Merge ranking scores with original search metadata
    search_papers = ArxivSearchTool.last_results
    ranked_papers = merge_rankings_with_search(result.pydantic, search_papers)

    # Render the report
    render_report(
        papers=ranked_papers,
        total_papers_searched=len(search_papers),
        search_date=args.date,
        categories_searched=categories_str,
        output_path="output/report.html",
    )
    print("\nDone! Report saved to output/report.html")


if __name__ == "__main__":
    main()
