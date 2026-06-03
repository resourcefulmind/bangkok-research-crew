import json
import uuid
import queue
import threading 

from flask import Flask, Response, request, jsonify, render_template
from bangkok.pipeline import run_pipeline

app = Flask(__name__)

# store state for each pipeline run with a run_id key and every route should look it up to find the right run
run_states = {}

@app.route("/")
def index():
    return render_template("dashboard.html")

@app.route("/run", methods=["POST"])
def start_run():
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    date = data.get("date")
    if not date:
        return jsonify({"error": "Missing required field: date"}), 400

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
        "message": "Pipeline started in background. Check progress at /stream/<run_id>",
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
    data = request.get_json(silent=True)
    if not data:
        return jsonify({"error": "Request body must be JSON"}), 400

    action = data.get("action")
    if action not in {"approve", "adjust", "abort"}:
        return jsonify({"error": "Invalid action. Must be one of: approve, adjust, abort"}), 400

    #write feedback to shared state
    run["feedback_data"]["action"] = action
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

