import os 
import argparse 
from dotenv import load_dotenv 
from crewai import Crew, Process 
from tasks import (
    search_task,
    novelty_task,
    impact_task,
    practical_task,
    ranking_task,
    output_task,
)

load_dotenv() 

crew = Crew(
    agents=[
        search_task.agent, 
        novelty_task.agent,
        impact_task.agent, 
        practical_task.agent, 
        ranking_task.agent, 
        output_task.agent, 
    ], 
    tasks=[
        search_task, 
        novelty_task, 
        impact_task, 
        practical_task, 
        ranking_task, 
        output_task, 
    ], 
    process=Process.sequential, 
    verbose=True, 
)

# Run Function
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
            help="Comma-separated list of categories to search for. ArXiv categories to search (default: cs.AI cs.LG cs.CL cs.CV stat.ML)", 
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
        print("\nDone! Report saved to output/report.html")

if __name__ == "__main__":
    main()