# Bangkok Research Crew

A multi-agent system that searches ArXiv for AI research papers published on a given date, ranks the top 10 by importance, and outputs a clean HTML report.

## How It Works

1. **Search** — Queries ArXiv for all AI papers on the specified date
2. **Review** — Pauses for your feedback on the search results
3. **Rank** — Uses Claude to rank papers by importance, novelty, and impact
4. **Review** — Pauses again for your feedback on the rankings
5. **Report** — Generates a self-contained HTML file with the top 10 papers

Human feedback is collected between steps so you can monitor progress and adjust on the fly.

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
python main.py --date 2026-03-31
```

The output HTML report will be saved to the `output/` directory.

## Tech Stack

- Python 3.11+
- [arxiv](https://pypi.org/project/arxiv/) — ArXiv API access
