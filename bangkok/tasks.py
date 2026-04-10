from crewai import Task
from .agents import (
    search_agent,
    novelty_evaluator,
    impact_evaluator,
    practical_evaluator,
    ranking_synthesizer,
)
from .models import RankingResult


# Task 1 - Search Arxiv for papers
def make_search_task(feedback: str = "") -> Task:
    description = (
        "Search ArXiv for research papers published on {date}. "
        "Call the arxiv_search tool EXACTLY ONCE with this input: "
        "'{date}, {categories}' — the tool handles combining categories "
        "internally. Do NOT make separate calls per category. "
        "Return whatever the tool finds. Do not evaluate or filter papers."
    )
    if feedback:
        description += f"\n\nUser feedback from previous attempt: {feedback}"

    return Task(
        description=description,
        expected_output=(
            "A complete list of all papers found, where each paper includes: "
            "title, authors, abstract, ArXiv URL, PDF link, and categories. "
            "Also include the total number of papers found."
        ),
        agent=search_agent,
        human_input=False,  # Temporarily disabled for testing
    )


# Task 2 - Evaluate Novelty of Papers
def make_novelty_task(search_task: Task) -> Task:
    description = (
        "Evaluate each paper from the search results on its NOVELTY only. "
        "For each paper, assess: Has this specific approach been done before? "
        "If similar work exists, how is this fundamentally different? Does it "
        "challenge an existing assumption in the field? Could other researchers "
        "build on this in surprising ways? Assign a novelty score from 1 to 10 "
        "where 1 is purely incremental and 10 is a groundbreaking new idea. "
        "Provide 1-2 sentences of reasoning for each score."
    )

    return Task(
        description=description,
        expected_output=(
            "A list of all papers with their novelty scores (1-10) and short "
            "reasoning for each score. Group papers into tiers: Breakthrough "
            "(8-10), Significant (5-7), Incremental (1-4)."
        ),
        agent=novelty_evaluator,
        context=[search_task],
    )


# Task 3 - Evaluate Impact of Papers
def make_impact_task(search_task: Task) -> Task:
    description = (
        "Evaluate each paper from the search results on its IMPACT and "
        "RIGOR only. For each paper, assess these four dimensions: SCALE — "
        "how many datasets, tasks, or domains were evaluated? RIGOR — is "
        "there mention of ablation studies, error bars, or statistical "
        "significance? REPRODUCIBILITY — is code or data availability "
        "mentioned? Are implementation details clear? BENCHMARKS — does this "
        "paper set a new standard or introduce a new evaluation method? "
        "Assign an impact score from 1 to 10 where 1 is minimal validation "
        "and 10 is comprehensive, field-defining work. Provide 1-2 sentences "
        "of reasoning for each score."
    )

    return Task(
        description=description,
        expected_output=(
            "A list of all papers with their impact scores (1-10) and short "
            "reasoning for each score. Group papers into tiers: Field-defining "
            "(8-10), Solid validation (5-7), Limited validation (1-4)."
        ),
        agent=impact_evaluator,
        context=[search_task],
    )


# Task 4 - Evaluate Practical Relevance of Papers
def make_practical_task(search_task: Task) -> Task:
    description = (
        "Evaluate each paper from the search results on its PRACTICAL "
        "APPLICABILITY only. For each paper, assess these five dimensions: "
        "PROBLEM RELEVANCE — does this solve a real problem practitioners "
        "care about? DEPLOYMENT READINESS — could someone implement this in "
        "a real system? EFFICIENCY — does it reduce compute, memory, or "
        "time requirements? ECOSYSTEM FIT — does it work with existing tools "
        "and frameworks? GENERALIZATION — can it be applied beyond the "
        "specific task shown in the paper? Assign a practicality score from "
        "1 to 10 where 1 is purely academic and 10 is immediately deployable. "
        "Provide 1-2 sentences of reasoning for each score."
    )

    return Task(
        description=description,
        expected_output=(
            "A list of all papers with their practicality scores (1-10) and "
            "short reasoning for each score. Group papers into tiers: Immediately "
            "useful (8-10), Moderately applicable (5-7), Primarily academic (1-4)."
        ),
        agent=practical_evaluator,
        context=[search_task],
    )


# Task 5 - Rank the Papers
def make_ranking_task(
    novelty_task: Task,
    impact_task: Task,
    practical_task: Task,
    feedback: str = "",
) -> Task:
    description = (
        "Using the novelty, impact, and practicality evaluations from the "
        "three specialist evaluators, produce a final ranking of the top 10 "
        "most important papers. For each paper, calculate a composite score "
        "by weighing novelty, impact, and practicality roughly equally. When "
        "evaluators disagree — for example, high novelty but low practicality "
        "— explain the trade-off and justify your ranking decision. "
        "For each paper in the top 10, provide: the final rank (1-10), "
        "the exact paper title, composite score out of 10, the individual "
        "novelty score, impact score, and practicality score, and a 2-3 "
        "sentence rationale explaining why it earned its position."
    )
    if feedback:
        description += f"\n\nUser feedback from previous attempt: {feedback}"

    return Task(
        description=description,
        expected_output=(
            "A ranked list of the top 10 papers. Each entry includes: rank, "
            "title, composite score (out of 10), individual novelty/impact/"
            "practicality scores, and a clear ranking rationale."
        ),
        agent=ranking_synthesizer,
        context=[novelty_task, impact_task, practical_task],
        output_pydantic=RankingResult,
        human_input=False,  # Temporarily disabled for testing
    )
