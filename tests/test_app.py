import queue
import threading
import pytest 
from unittest.mock import patch 

from app import app as flask_app, run_states

@pytest.fixture
def client():
    """Flask test client. Sends HTTP requests without running a real server"""
    flask_app.config["TESTING"] = True
    with flask_app.test_client() as c:
        yield c
    run_states.clear()

@patch("app.run_pipeline")
def test_post_run_returns_id_and_202(mock_run_pipeline, client):
    """POST /run should return a run_id and 202 status Accepted"""
    response = client.post("/run", json={"date": "2026-04-15", "categories": "cs.AI, cs.LG"})
    
    assert response.status_code == 202
    data = response.get_json()
    assert "run_id" in data 
    assert data["status"] == "started" 

def test_stream_bad_run_id_returns_404(client):
    """GET /stream/bad-id should return 404 when run_id is not found"""
    response = client.get("/stream/not-a-real-uuid")
    
    assert response.status_code == 404
    data = response.get_json()
    assert "error" in data

@patch("app.run_pipeline")
def test_run_missing_date_returns_400(mock_run_pipeline, client): 
    """POST /run without a date should be rejected with 400 Bad Request before any run is spawned"""
    response = client.post("/run", json={"categories": "cs.AI, cs.LG"})

    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    assert "date" in data["error"].lower()

def test_feedback_bad_action_returns_400(client):
    """POST /feedback with an unknown action should be rejected with 400 Bad Request, not silently approved."""
    run_id = "seeded-run-id" 
    run_states[run_id] = { 
        "queue": queue.Queue(), 
        "feedback_event": threading.Event(), 
        "feedback_data": {}, 
        "run_state": {
            "status": "running", 
            "report_html": None, 
        }, 
    }
    response = client.post(f"/feedback/{run_id}", json={"action": "banana"})
    
    assert response.status_code == 400
    data = response.get_json()
    assert "error" in data
    # critical: a bad action must NOT have ublocked the waiting pipeline
    assert not run_states[run_id]["feedback_event"].is_set()

def test_feedback_valid_action_stores_data_and_sets_event(client):
    """POST /feedback with a valid action should store it and unblock the waiting pipeline."""
    run_id = "seeded-run-id"
    feedback_data = {}
    feedback_event = threading.Event()
    run_states[run_id] = {
        "queue": queue.Queue(),
        "feedback_event": feedback_event,
        "feedback_data": feedback_data,
        "run_state": {"status": "running", "report_html": None},
    }

    response = client.post(
        f"/feedback/{run_id}",
        json={"action": "adjust", "message": "prefer LLM papers"},
    )

    assert response.status_code == 200
    assert feedback_data["action"] == "adjust"
    assert feedback_data["message"] == "prefer LLM papers"
    assert feedback_event.is_set()

def test_report_returns_202_until_ready_then_html(client):
    """GET /report returns 202 while rendering, then 200 + HTML once the pipeline stores it."""
    run_id = "seeded-run-id"
    run_state = {"status": "running", "report_html": None}
    run_states[run_id] = {
        "queue": queue.Queue(),
        "feedback_event": threading.Event(),
        "feedback_data": {},
        "run_state": run_state,
    }

    # Report not rendered yet
    pending = client.get(f"/report/{run_id}")
    assert pending.status_code == 202

    # Pipeline finishes and stores the HTML
    run_state["report_html"] = "<html><body>Top 10 papers</body></html>"

    ready = client.get(f"/report/{run_id}")
    assert ready.status_code == 200
    assert ready.mimetype == "text/html"
    assert "Top 10 papers" in ready.get_data(as_text=True)