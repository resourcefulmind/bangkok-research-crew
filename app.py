import json
import uuid
import queue
import threading 

from flask import Flask, Response, request, jsonify
from bangkok.pipeline import run_pipeline

app = Flask(__name__)

# store state for each pipeline run with a run_id key and every route should look it up to find the right run
run_states = {}

@app.route("/")
def index(): 
    return "<h1>Bangkok Research Crew</h1><p>This is a multi-agent system that searches ArXiv for AI research papers published on a given date, ranks the top 10 by importance, and outputs a clean HTML report.</p>Dashboard coming in step 5<p>"

@app.route("/run", methods=["POST"])
def start_run():
    data = request.get_json()
    date = data.get("date")
    categories = data.get("categories", "cs.AI, cs.LG, cs.CL, cs.CV, stat.ML")

    run_id = str(uuid.uuid4())
    
    #create shared objects for specific run
    event_queue = queue.Queue()
    feedback_event = threading.Event()
    feedback_data = {}
    run_state = {
        "status": "running", 
        "report_html": None, 
    }

    #store so other routes can find them by run_id
    run_states[run_id] = {
        "queue": event_queue, 
        "feedback_event": feedback_event, 
        "feedback_data": feedback_data, 
        "run_state": run_state, 
    }

    #spawn pipeline in background thread 
    thread = threading.Thread(
        target=run_pipeline, 
        args=(run_id, date, categories, event_queue, feedback_event, feedback_data, run_state), 
        daemon=True, 
    )
    thread.start()

    return jsonify({
        "run_id": run_id, 
        "status": "started", 
        "message": "Pipeline started in background. Check progress at /progress/<run_id>", 
    }), 202

# SSE endpoint for progress updates
@app.route("/stream/<run_id>")
def stream(run_id): 
    run = run_states.get(run_id)
    if not run:
        return jsonify({
            "error": "Run not found", 
        }), 404
    event_queue = run["queue"] 

    def generate():
        while True:
            try:
                event = event_queue.get(timeout=30)
            except queue.Empty:
                # send a comment to keep the connection alive
                yield ": keepalive\n\n"
                continue
            yield f"data: {json.dumps(event)}\n\n"

            # stop streaming when pipeline is complete or aborted
            if event["type"] in ["complete", "aborted", "error"]:
                break
    
    return Response(
        generate(), 
        mimetype="text/event-stream", 
        headers={
            "Cache-Control": "no-cache",
        }
    )

@app.route("/feedback/<run_id>", methods=["POST"]) 
def feedback(run_id): 
    run = run_states.get(run_id)
    if not run: 
        return jsonify({
            "error": "Run not found", 
        }), 404
    data = request.get_json()

    #write feedback to shared state
    run["feedback_data"]["action"] = data.get("action", "approve")
    run["feedback_data"]["message"] = data.get("message", "")

    #signal pipeline to continue/wake up from feedback checkpoint
    run["feedback_event"].set()
    return jsonify({
        "status": "feedback received", 
        "message": "Thank you for your feedback!", 
    }), 200

# route to serve the HTML report
@app.route("/report/<run_id>")
def report(run_id): 
    run = run_states.get(run_id)
    if not run:
        return jsonify({"error": "Run not found"}), 404
    
    html = run["run_state"].get("report_html")
    if not html:
        return jsonify({"error": "Report not ready yet"}), 202

    return Response(
        html, 
        mimetype="text/html", 
    )

if __name__ == "__main__": 
    app.run(debug=True, port=5000, threaded=True)