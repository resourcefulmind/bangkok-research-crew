from datetime import datetime
from jinja2 import Environment, FileSystemLoader
from models import RankedPaper


def render_report(
    papers: list[RankedPaper],
    total_papers_searched: int,
    search_date: str,
    categories_searched: str,
    output_path: str,
) -> None:
    env = Environment(loader=FileSystemLoader("templates"))
    template = env.get_template("report.html")

    html = template.render(
        papers=papers,
        total_papers_searched=total_papers_searched,
        search_date=search_date,
        categories_searched=categories_searched,
        generated_at=datetime.now().strftime("%Y-%m-%d %H:%M"),
    )

    with open(output_path, "w") as f:
        f.write(html)
