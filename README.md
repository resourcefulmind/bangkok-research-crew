# Bangkok Research Crew

Point it at a date. It pulls that day's AI papers off arXiv, has a small crew of Claude agents argue about which ten matter most, shows you the ranking, and once you sign off, hands you a single HTML file you can open anywhere.

You can run it two ways: a web dashboard that streams the agents working in real time, or a one-shot command line.

![Bangkok Research Crew](https://res.cloudinary.com/resourcefulmind-inc/image/upload/v1780481803/dashboard_ha3gln.png)

## What's in it

```text
Bangkok Research Crew
├── Two ways to run
│   ├── Web dashboard      live progress over SSE, feedback in the browser
│   └── CLI                one-shot run, writes an HTML file
├── Pipeline
│   ├── Search             direct arXiv query, no LLM
│   ├── Evaluate           three agents: novelty, impact, practicality
│   └── Rank               a fourth agent synthesizes the top N (you pick 5 or 10)
├── Human-in-the-loop      one checkpoint, after ranking
│   ├── Approve            render the report
│   ├── Adjust             re-rank with a note from you
│   └── Abort              stop at the next safe point
└── Report                 one self-contained HTML file, dark-mode aware
```

## Run it

**You'll need:** Python 3.11+, an Anthropic API key with a little credit, and (recommended) a virtualenv.

```bash
cp .env.example .env          # then set ANTHROPIC_API_KEY
pip install -r requirements.txt
```

Web dashboard:

```bash
python app.py                 # open http://127.0.0.1:5000
```

Or one-shot from the terminal:

```bash
python main.py --date 2026-05-28
```

The CLI writes its report to `output/report.html`. The dashboard serves it in the browser.

> On macOS, the AirPlay Receiver also sits on port 5000. If the page won't load, turn AirPlay Receiver off (System Settings → General → AirDrop & Handoff), or change the port in `app.py`.

## How it works

A run moves through three stages.

**Search.** A plain arXiv query for the date and categories you chose. There's no model in this step. It's fetching data, and you don't need an LLM to fetch data. (An earlier version wrapped the search in an agent that "summarized" the results before handing them on, so we were paying for a model call whose output we then ignored. We cut it.)

**Evaluate.** Three agents each score every paper on one axis: novelty, impact, practicality. A fourth folds those into a ranked top ten. Splitting the judgment keeps each agent's job narrow, which keeps its scores legible. When something ranks high on novelty but low on practicality, you can see exactly why.

**Review.** The run stops and waits for you. Approve, and it renders the report. Adjust, and it re-ranks with a note from you. Abort, and it ends. There's one checkpoint, and it sits here, after ranking, because that's the first moment you have something worth judging. (Your other real input is up front, on the start screen: the date, which categories to search, and whether you want the top 5 or top 10. Everything between is mechanical.)

The dashboard shows all of this as it happens using **server-sent events**, a one-way stream from the server to the page. The browser only needs to listen; it never pushes mid-run, so a full WebSocket is more machinery than the job needs. The pipeline runs on a background thread and pushes progress onto a queue, and the page reads from it. At the review step, the thread simply blocks until your answer comes back.

## Limitations (by design)

This is a single-user tool you run on your own machine, and a worked example for an article. So, on purpose: no accounts, no multi-user handling, no reconnect if the stream drops, and no canceling a run mid-flight (abort takes effect at the next checkpoint, not instantly; there's no way to interrupt a model call that's already in progress). A run spends a small amount of Anthropic credit, roughly a couple of evaluation passes.

## Status

The dashboard works, and earlier runs produced full reports end to end. The most recent change (removing the LLM from the search step) is verified in pieces: the search call, the evaluation, and the rendering each check out, but it hasn't yet had one clean run against live arXiv, which kept rate-limiting us during testing.

## Tech

CrewAI runs the agents, Claude does the judging, the `arxiv` package handles fetching, Flask and server-sent events drive the live page, and Jinja2 renders the report.

## Tests

```bash
pip install -r requirements-dev.txt
python -m pytest tests/
```

The tests mock the model, so they're fast and free. They cover the pipeline's event flow, the Flask routes and their input validation, and a check that a malicious paper title can't inject script into the report.

## License

MIT. See [LICENSE](LICENSE). Set your own name in the copyright line, and swap the license if you'd prefer something else.
