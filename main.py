import argparse
from dotenv import load_dotenv
from crewai import Crew, Process
from bangkok.tasks import (
    make_search_task,
    make_novelty_task,
    make_impact_task,
    make_practical_task,
    make_ranking_task,
)
from bangkok.models import merge_rankings_with_search
from bangkok.tools import ArxivSearchTool
from bangkok.render import render_report

load_dotenv()


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

    # Build tasks using factory functions — fresh instances per run
    search_task = make_search_task()
    novelty_task = make_novelty_task(search_task)
    impact_task = make_impact_task(search_task)
    practical_task = make_practical_task(search_task)
    ranking_task = make_ranking_task(novelty_task, impact_task, practical_task)

    # Build the crew
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

    # Run the crew
    result = crew.kickoff(
        inputs={
            "date": args.date,
            "categories": categories_str,
        }
    )

    # Merge ranking scores with original search metadata
    search_papers = ArxivSearchTool.last_results
    ranked_papers = merge_rankings_with_search(result.pydantic.papers, search_papers)

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
