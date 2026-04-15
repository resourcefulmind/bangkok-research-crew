import queue
import logging
import threading
from crewai import Crew, Process

from .tasks import (
    make_search_task,
    make_novelty_task,
    make_impact_task,
    make_practical_task,
    make_ranking_task,
)
from .models import merge_rankings_with_search
from .tools import ArxivSearchTool
from .render import render_report_string

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - [%(threadName)s] - %(name)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


class PipelineAborted(Exception):
    """Raised when the user aborts the pipeline at a feedback checkpoint."""
    pass


def emit_event(q: queue.Queue, event_type: str, **data) -> None:
    """Push an event onto the queue for the SSE endpoint to pick up."""
    event = {"type": event_type, **data}
    try:
        q.put_nowait(event)
    except queue.Full:
        logger.warning(f"Queue is full, dropping event: {event_type}")
    logger.info(f"Event emitted: {event_type} | {data}")


# ── Stage Functions ──────────────────────────────────────────────────


def _run_search(run_id, date, categories, event_queue):
    """Stage A: Search ArXiv for papers."""
    logger.info(f"[{run_id}] Stage A: Starting search")
    emit_event(event_queue, "agent_start", agent="Search Agent", stage="search")

    search_task = make_search_task()
    search_crew = Crew(
        agents=[search_task.agent],
        tasks=[search_task],
        process=Process.sequential,
        verbose=True,
    )
    search_crew.kickoff(inputs={"date": date, "categories": categories})

    search_papers = ArxivSearchTool.last_results
    paper_count = len(search_papers)
    logger.info(f"[{run_id}] Stage A: Found {paper_count} papers")
    emit_event(
        event_queue, "agent_complete",
        agent="Search Agent",
        summary=f"Found {paper_count} papers for {date} in {categories}",
    )

    return search_task, search_papers


def _run_evaluation(run_id, search_task, event_queue):
    """Stage B: Evaluate papers on novelty, impact, practicality + rank."""
    logger.info(f"[{run_id}] Stage B: Starting evaluation")
    emit_event(event_queue, "agent_start", agent="Evaluation Pipeline", stage="evaluation")

    novelty_task = make_novelty_task(search_task)
    impact_task = make_impact_task(search_task)
    practical_task = make_practical_task(search_task)
    ranking_task = make_ranking_task(novelty_task, impact_task, practical_task)

    eval_crew = Crew(
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
    eval_result = eval_crew.kickoff()

    logger.info(f"[{run_id}] Stage B: Evaluation complete")
    emit_event(
        event_queue, "agent_complete",
        agent="Evaluation Pipeline",
        summary="All evaluations and ranking complete",
    )

    return eval_result, novelty_task, impact_task, practical_task


def _run_feedback_checkpoint(
    run_id, eval_result, novelty_task, impact_task, practical_task,
    event_queue, feedback_event, feedback_data,
):
    """Feedback checkpoint: show ranking, wait for approve/adjust/abort."""
    ranking_result = eval_result

    while True:
        top_papers = []
        if ranking_result.pydantic:
            for p in ranking_result.pydantic.papers:
                top_papers.append({
                    "rank": p.rank,
                    "title": p.title,
                    "composite_score": p.composite_score,
                    "rationale": p.rationale,
                })

        logger.info(f"[{run_id}] Checkpoint: Waiting for user feedback")
        emit_event(
            event_queue, "feedback_requested",
            checkpoint="ranking",
            top_papers=top_papers,
        )

        feedback_event.wait()
        feedback_event.clear()

        action = feedback_data.get("action", "approve")
        message = feedback_data.get("message", "")
        logger.info(f"[{run_id}] Checkpoint: User chose '{action}'")

        if action == "approve":
            break

        elif action == "abort":
            raise PipelineAborted("User aborted at ranking checkpoint")

        elif action == "adjust":
            logger.info(f"[{run_id}] Re-running ranking with feedback: {message}")
            emit_event(
                event_queue, "agent_start",
                agent="Ranking Synthesizer",
                stage="re-ranking",
            )

            ranking_task = make_ranking_task(
                novelty_task, impact_task, practical_task,
                feedback=message,
            )
            ranking_crew = Crew(
                agents=[ranking_task.agent],
                tasks=[ranking_task],
                process=Process.sequential,
                verbose=True,
            )
            ranking_result = ranking_crew.kickoff()

            emit_event(
                event_queue, "agent_complete",
                agent="Ranking Synthesizer",
                summary="Re-ranking complete with user feedback",
            )

    return ranking_result


def _run_merge_and_render(
    run_id, ranking_result, search_papers, date, categories,
    event_queue, run_state,
):
    """Stage C: Merge ranking scores with search metadata, render HTML."""
    logger.info(f"[{run_id}] Stage C: Merging and rendering")
    emit_event(event_queue, "agent_start", agent="Report Generator", stage="rendering")

    merged = merge_rankings_with_search(
        ranking_result.pydantic.papers, search_papers
    )

    html = render_report_string(
        papers=merged,
        total_papers_searched=len(search_papers),
        search_date=date,
        categories_searched=categories,
    )

    run_state["report_html"] = html
    run_state["status"] = "complete"

    logger.info(f"[{run_id}] Stage C: Report rendered ({len(html)} chars)")
    emit_event(
        event_queue, "complete",
        report_url=f"/report/{run_id}",
    )


# ── Main Orchestrator ────────────────────────────────────────────────


def run_pipeline(
    run_id: str,
    date: str,
    categories: str,
    event_queue: queue.Queue,
    feedback_event: threading.Event,
    feedback_data: dict,
    run_state: dict,
) -> None:
    """
    Run the full paper search + evaluation + ranking pipeline.

    Called in a background thread by app.py. Communicates with Flask via:
    - event_queue: push progress events (pipeline -> Flask -> browser)
    - feedback_event: block until user responds at checkpoint (Flask -> pipeline)
    - feedback_data: dict where Flask stores the user's feedback response
    - run_state: shared dict where we store the final report HTML
    """
    try:
        search_task, search_papers = _run_search(
            run_id, date, categories, event_queue
        )
        eval_result, novelty_task, impact_task, practical_task = _run_evaluation(
            run_id, search_task, event_queue
        )
        ranking_result = _run_feedback_checkpoint(
            run_id, eval_result, novelty_task, impact_task, practical_task,
            event_queue, feedback_event, feedback_data,
        )
        _run_merge_and_render(
            run_id, ranking_result, search_papers, date, categories,
            event_queue, run_state,
        )

    except PipelineAborted as e:
        logger.info(f"[{run_id}] Pipeline aborted: {e}")
        run_state["status"] = "aborted"
        emit_event(event_queue, "aborted", reason=str(e))

    except Exception as e:
        logger.error(f"[{run_id}] Pipeline error: {e}", exc_info=True)
        run_state["status"] = "error"
        emit_event(event_queue, "error", message=str(e))
