import argparse
from dotenv import load_dotenv
from crewai import Crew, Process
from bangkok.tasks import (
    make_novelty_task,
    make_impact_task,
    make_practical_task,
    make_ranking_task,
)
from bangkok.models import merge_rankings_with_search
from bangkok.tools import ArxivSearchTool, format_papers_for_eval
from bangkok.render import render_report

load_dotenv()


def main():
    parser = argparse.ArgumentParser(
        description="Search ArXiv for AI research papers, rank the most important ones, and generate a report"
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
    parser.add_argument(
        "--top-n",
        type=int,
        choices=[5, 10],
        default=10,
        help="How many papers to rank (5 or 10, default 10)",
    )
    args = parser.parse_args()

    categories_str = ", ".join(args.categories)

    print(f"\nSearching ArXiv for papers on {args.date}")
    print(f"Categories: {categories_str}\n")

    # Search is deterministic data-fetching — call the tool directly, no LLM.
    ArxivSearchTool()._run(f"{args.date}, {categories_str}")
    search_papers = ArxivSearchTool.last_results
    if not search_papers:
        print(
            "No papers found. ArXiv may be unavailable, or there were no "
            "submissions that day — try a recent weekday."
        )
        return
    print(f"Found {len(search_papers)} papers.\n")

    papers_text = format_papers_for_eval(search_papers)

    # Build the evaluation tasks (fresh instances per run)
    novelty_task = make_novelty_task(papers_text)
    impact_task = make_impact_task(papers_text)
    practical_task = make_practical_task(papers_text)
    ranking_task = make_ranking_task(novelty_task, impact_task, practical_task, top_n=args.top_n)

    crew = Crew(
        agents=[
            novelty_task.agent,
            impact_task.agent,
            practical_task.agent,
            ranking_task.agent,
        ],
        tasks=[novelty_task, impact_task, practical_task, ranking_task],
        process=Process.sequential,
        verbose=True,
    )

    # No inputs: papers are embedded in the task descriptions, and we don't
    # want CrewAI re-interpolating braces that can appear in abstracts.
    result = crew.kickoff()

    # Merge ranking scores with original search metadata
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
