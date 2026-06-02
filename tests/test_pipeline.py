import queue 
import threading 
from unittest.mock import patch, MagicMock 
from bangkok.pipeline import run_pipeline
from bangkok.models import RankedPaperSummary, RankingResult, RankedPaper
from bangkok.render import render_report_string

def _make_fake_ranking_result(): 
    """Build a fake CrewOutput that looks like what the ranking crew returns. """
    ranking = RankingResult(
        papers=[
            RankedPaperSummary(
                rank=1, 
                title="Test Paper 1", 
                composite_score=9.0, 
                novelty_score=9.0, 
                impact_score=8.5, 
                practicality_score=9.5, 
                rationale="A groundbreaking paper but on the side, we all know that this is a fake paper for testing purposes", 
            ), 
        ]
    )
    fake_output = MagicMock()
    fake_output.pydantic = ranking 
    return fake_output 


def _make_fake_search_papers(): 
    """Build fake search papers that match the fake ranking."""
    return [
        {
            "title": "Test Paper 1", 
            "authors": "John Doe, Jane Smith", 
            "abstract": "This is a test abstract", 
            "arxiv_url": "https://arxiv.org/abs/2001.00001", 
            "pdf_url": "https://arxiv.org/pdf/2001.00001", 
            "categories": "cs.AI, cs.LG", 
            "published": "2026-04-15", 
        }
    ]

def _run_pipeline_with_feedback(action, message=""): 
    """Helper: run the pipeline with mocked LLM and auto-submit feedback."""
    event_queue = queue.Queue()
    feedback_event = threading.Event()
    feedback_data = {}
    run_state = {
        "status": "running", 
        "report_html": None, 
    }

    fake_result = _make_fake_ranking_result()
    fake_papers = _make_fake_search_papers()

    # Events consumed by auto_feedback are saved here so they're not lost
    consumed_events = []

    def auto_feedback():
        """Wait for feedback_requested event, then submit feedback."""
        while True:
            try:
                event = event_queue.get(timeout=5)
                consumed_events.append(event)
                if event["type"] == "feedback_requested":
                    feedback_data["action"] = action
                    feedback_data["message"] = message
                    feedback_event.set()
                    break
            except queue.Empty:
                break

    feedback_thread = threading.Thread(target=auto_feedback)
    feedback_thread.start()

    with patch("bangkok.pipeline.Crew") as MockCrew, \
        patch("bangkok.pipeline.ArxivSearchTool") as MockTool:

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = fake_result
        MockCrew.return_value = mock_crew

        MockTool.last_results = fake_papers

        run_pipeline(
            run_id="test-123", 
            date="2026-04-15", 
            categories="cs.AI, cs.LG", 
            event_queue=event_queue, 
            feedback_event=feedback_event, 
            feedback_data=feedback_data, 
            run_state=run_state, 
        )

    feedback_thread.join(timeout=5)

    # Combine events: ones consumed by auto_feedback + ones still in queue
    events = list(consumed_events)
    while not event_queue.empty():
        events.append(event_queue.get_nowait())

    return events, run_state

def test_approve_produces_complete_event(): 
    """When user approves, pipeline should finish with a complete event."""
    events, run_state = _run_pipeline_with_feedback("approve")

    event_types = [e["type"] for e in events]
    assert "complete" in event_types, f"Expected 'complete' in events, got: {event_types}"
    assert run_state["status"] == "complete"
    assert run_state["report_html"] is not None

def test_abort_produces_aborted_event(): 
    """When user aborts, pipeline should stop with an aborted event."""
    events, run_state = _run_pipeline_with_feedback("abort")

    event_types = [e["type"] for e in events]
    assert "aborted" in event_types, f"Expected 'aborted' in events, got: {event_types}"
    assert run_state["status"] == "aborted"

def test_event_sequence_on_approve(): 
    """Events should arrive in the correct stage order."""
    events, _ = _run_pipeline_with_feedback("approve")

    event_types = [e["type"] for e in events]

    # Should see: search start/complete. eval start/complete, 
    # feedback_requested, render start, complete
    assert event_types[0] == "agent_start", f"first event should be agent_start, got: {event_types[0]}"
    assert "feedback_requested" in event_types, "feedback_requested should be in events" 
    assert event_types[-1] == "complete", f"last event should be complete, got: {event_types[-1]}"

    # feedback_requested should come before complete
    fb_index = event_types.index("feedback_requested")
    complete_index = event_types.index("complete")
    assert fb_index < complete_index, "feedback_requested should come before complete"


def test_malicious_paper_title_is_escaped():
    """A paper title containing HTML must be escaped in the report, not rendered as live markup (XSS guard)."""
    malicious = RankedPaper(
        rank=1,
        title="<script>alert('xss')</script>",
        authors="Jane Doe",
        arxiv_url="https://arxiv.org/abs/2001.00001",
        pdf_url="https://arxiv.org/pdf/2001.00001",
        categories="cs.AI",
        abstract="A normal abstract.",
        composite_score=9.0,
        novelty_score=9.0,
        impact_score=9.0,
        practicality_score=9.0,
        rationale="Testing escaping.",
    )

    html = render_report_string(
        papers=[malicious],
        total_papers_searched=1,
        search_date="2026-04-15",
        categories_searched="cs.AI",
    )

    # The raw tag must NOT survive into the output...
    assert "<script>alert('xss')</script>" not in html
    # ...it must be escaped instead
    assert "&lt;script&gt;" in html


def test_empty_search_results_produces_error_event():
    """If ArXiv returns no papers, the pipeline emits a clear error event and
    never reaches the feedback checkpoint — not an AttributeError on last_results."""
    event_queue = queue.Queue()
    feedback_event = threading.Event()
    feedback_data = {}
    run_state = {"status": "running", "report_html": None}

    with patch("bangkok.pipeline.Crew") as MockCrew, \
         patch("bangkok.pipeline.ArxivSearchTool") as MockTool:

        mock_crew = MagicMock()
        mock_crew.kickoff.return_value = MagicMock()
        MockCrew.return_value = mock_crew

        MockTool.last_results = []   # search found nothing (e.g. arXiv 503 / empty date)

        run_pipeline(
            run_id="test-empty",
            date="2026-05-31",
            categories="cs.AI",
            event_queue=event_queue,
            feedback_event=feedback_event,
            feedback_data=feedback_data,
            run_state=run_state,
        )

    events = []
    while not event_queue.empty():
        events.append(event_queue.get_nowait())
    event_types = [e["type"] for e in events]

    assert "error" in event_types, f"Expected 'error' in events, got: {event_types}"
    assert "feedback_requested" not in event_types, "should fail before the checkpoint"
    assert run_state["status"] == "error"

    error_event = next(e for e in events if e["type"] == "error")
    assert "No papers" in error_event["message"]