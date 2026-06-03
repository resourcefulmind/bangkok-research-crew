import queue
import logging
import threading
from crewai import Crew, Process

from .tasks import (
    make_novelty_task,
    make_impact_task,
    make_practical_task,
    make_ranking_task,
)
from .models import merge_rankings_with_search
from .tools import ArxivSearchTool, format_papers_for_eval
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


class _ArxivRetryReporter(logging.Handler):
    """Bridges the arxiv library's backoff log records into a dashboard status,
    so a slow or unavailable arXiv shows progress instead of a silent spinner.
    The library logs 'Sleeping: N seconds' only when backing off before a retry.
    """

    def __init__(self, event_queue):
        super().__init__()
        self.event_queue = event_queue

    def emit(self, record):
        if "Sleeping" in record.getMessage():
            emit_event(
                self.event_queue, "agent_progress",
                agent="Search Agent",
                message="arXiv is responding slowly — retrying…",
            )


# ── Stage Functions ──────────────────────────────────────────────────


def _run_search(run_id, date, categories, event_queue):
    """Stage A: Search ArXiv for papers."""
    logger.info(f"[{run_id}] Stage A: Starting search")
    emit_event(event_queue, "agent_start", agent="Search Agent", stage="search")

    # Let the search tool report live status ("Querying arXiv…", "Found N"),
    # and bridge arXiv's retry/backoff logging so a slow arXiv shows up too.
    ArxivSearchTool.progress = lambda msg: emit_event(
        event_queue, "agent_progress", agent="Search Agent", message=msg
    )
    arxiv_logger = logging.getLogger("arxiv")
    retry_reporter = _ArxivRetryReporter(event_queue)
    arxiv_logger.addHandler(retry_reporter)

    try:
        # Search is deterministic data-fetching — call the tool directly, no
        # CrewAI agent/LLM. The tool stores its results on .last_results.
        ArxivSearchTool()._run(f"{date}, {categories}")
        search_papers = ArxivSearchTool.last_results
    finally:
        # Always detach the per-run hooks, even if the search raises.
        ArxivSearchTool.progress = None
        arxiv_logger.removeHandler(retry_reporter)

    paper_count = len(search_papers)

    if not search_papers:
        raise RuntimeError(
            f"No papers returned for {date}. ArXiv may be unavailable, or there "
            f"were no submissions in {categories} that day — try a recent weekday."
        )

    logger.info(f"[{run_id}] Stage A: Found {paper_count} papers")
    emit_event(
        event_queue, "agent_complete",
        agent="Search Agent",
        summary=f"Found {paper_count} papers for {date} in {categories}",
    )

    papers_text = format_papers_for_eval(search_papers)
    return papers_text, search_papers


def _run_evaluation(run_id, papers_text, event_queue):
    """Stage B: Evaluate papers on novelty, impact, practicality + rank.

    Emits a per-evaluator event sequence so the dashboard shows each evaluator
    light up in turn — instead of one opaque "Evaluation Pipeline" spinner
    through the longest stage. Uses CrewAI's per-task `callback` (documented
    API), not the internal event bus.
    """
    logger.info(f"[{run_id}] Stage B: Starting evaluation")

    novelty_task = make_novelty_task(papers_text)
    impact_task = make_impact_task(papers_text)
    practical_task = make_practical_task(papers_text)
    ranking_task = make_ranking_task(novelty_task, impact_task, practical_task)

    # The four evaluators run sequentially. Each task's callback fires on
    # completion, so we mark it done and start the next one — giving the
    # frontend a live card-by-card animation using the existing event types.
    # (The dynamic stepper builds a card per agent name, so no frontend change.)
    sequence = [
        (novelty_task, "Novelty Evaluator", "Impact Evaluator"),
        (impact_task, "Impact Evaluator", "Practicality Evaluator"),
        (practical_task, "Practicality Evaluator", "Ranking Synthesizer"),
        (ranking_task, "Ranking Synthesizer", None),
    ]
    for task, name, next_name in sequence:
        def make_callback(name=name, next_name=next_name):
            def on_complete(_output):
                logger.info(f"[{run_id}] Stage B: {name} complete")
                emit_event(event_queue, "agent_complete", agent=name, summary=f"{name} done")
                if next_name:
                    emit_event(event_queue, "agent_start", agent=next_name, stage="evaluation")
            return on_complete
        task.callback = make_callback()

    # Light up the first evaluator before the crew starts.
    emit_event(event_queue, "agent_start", agent="Novelty Evaluator", stage="evaluation")

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
        papers_text, search_papers = _run_search(
            run_id, date, categories, event_queue
        )
        eval_result, novelty_task, impact_task, practical_task = _run_evaluation(
            run_id, papers_text, event_queue
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
