from __future__ import annotations

from typing import List, Optional, Dict, Tuple

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from orchestrator import (
    register_default_agents,
    agentic_once_with_metadata,
    TELEMETRY_EVENTS,
)

console = Console()


def compute_simple_metrics(answer, citations, context_snippets) -> Dict[str, float]:
    num_snippets = len(context_snippets)
    num_docs = len({cs.doc_id for cs in context_snippets}) if context_snippets else 0

    scores = [cs.score for cs in context_snippets if cs.score is not None]
    avg_score = sum(scores) / len(scores) if scores else 0.0
    max_score = max(scores) if scores else 0.0

    def _clamp01(x: float) -> float:
        try:
            return max(0.0, min(1.0, float(x)))
        except Exception:
            return 0.0

    avg_score = _clamp01(avg_score)
    max_score = _clamp01(max_score)

    num_citations = len(citations)
    cited_docs = {c.doc_id for c in citations} if citations else set()
    num_cited_docs = len(cited_docs)

    text = answer.text or ""
    answer_chars = len(text)
    answer_words = len(text.split()) if text else 0

    return {
        "num_snippets": num_snippets,
        "num_docs": num_docs,
        "avg_score": avg_score,
        "max_score": max_score,
        "num_citations": num_citations,
        "num_cited_docs": num_cited_docs,
        "answer_chars": answer_chars,
        "answer_words": answer_words,
    }


def print_header(config_path: Optional[str], num_queries: int) -> None:
    text_lines = [
        "[bold]Configuration[/bold]",
        f"• Config path : [cyan]{config_path or '<default>'}[/cyan]",
        f"• Num queries : [cyan]{num_queries}[/cyan]",
    ]
    panel = Panel(
        "\n".join(text_lines),
        title="[bold cyan]Agentic RAG Smoke Test[/bold cyan]",
        border_style="cyan",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_query_block(i: int, query: str) -> None:
    body = "\n".join(
        [
            "[bold]User Query[/bold]",
            "",
            query,
        ]
    )
    panel = Panel(
        body,
        title=f"[bold magenta]Query {i}[/bold magenta]",
        border_style="magenta",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_answer_block(answer) -> None:
    text = answer.text or ""
    body = "\n".join(
        [
            "[bold]Answer[/bold]",
            "",
            text,
        ]
    )
    panel = Panel(
        body,
        title="[bold green]Final Answer[/bold green]",
        border_style="green",
        box=box.ROUNDED,
    )
    console.print(panel)
    console.print()


def print_context_table(context_snippets: List) -> None:
    if not context_snippets:
        console.print(
            Panel(
                "No context snippets retrieved.",
                title="[bold yellow]Context[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    table = Table(
        title="[bold cyan]Context Snippets (top 20)[/bold cyan]",
        box=box.SIMPLE,
        show_lines=True,
        expand=True,
    )
    table.add_column("#", justify="right", width=4)
    table.add_column("Score", justify="right", width=8)
    table.add_column("Title / File", width=40)
    table.add_column("Page", justify="right", width=6)
    table.add_column(
        "Snippet (translated where applicable)",
        overflow="fold",
        width=80,
    )

    for idx, cs in enumerate(context_snippets[:20], start=1):
        score_str = f"{cs.score:.4f}" if isinstance(cs.score, (int, float)) else "0.0000"
        title = cs.doc_title or "(unknown)"
        page = str(cs.page) if cs.page is not None else "?"

        snippet = cs.translated_text or cs.source_text or ""
        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) > 320:
            snippet = snippet[:317].rstrip() + "…"

        table.add_row(
            str(idx),
            score_str,
            title,
            page,
            snippet,
        )

    console.print(table)
    console.print()


def print_sources_table(citations, context_snippets) -> None:
    if not citations:
        console.print(
            Panel(
                "No citations produced.",
                title="[bold yellow]Sources[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    title_idx: Dict[Tuple[str, str], str] = {}
    level_idx: Dict[Tuple[str, str], str] = {}
    page_idx: Dict[Tuple[str, str], int] = {}

    for cs in context_snippets:
        key = (cs.doc_id, cs.chunk_id)
        if cs.doc_title:
            title_idx[key] = cs.doc_title
        if getattr(cs, "level", None) is not None:
            level_idx[key] = cs.level
        if cs.page is not None:
            page_idx[key] = cs.page

    table = Table(
        title="[bold cyan]Sources[/bold cyan]",
        box=box.SIMPLE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Ref", justify="right", width=6)
    table.add_column("Doc ID (short)", width=22)
    table.add_column("Title / File", width=64)
    table.add_column("Page", justify="right", width=6)
    table.add_column("Level", justify="center", width=10)

    for i, c in enumerate(citations, start=1):
        key = (c.doc_id, c.chunk_id)

        doc_id_full = c.doc_id
        doc_id_short = doc_id_full if len(doc_id_full) <= 18 else doc_id_full[:15] + "…"

        title = title_idx.get(key)
        if title is None:
            for cs in context_snippets:
                if cs.doc_id == c.doc_id and cs.doc_title:
                    title = cs.doc_title
                    break
        if title is None:
            title = "(unknown title)"

        page_val = getattr(c, "page", None)
        if page_val is None:
            page_val = page_idx.get(key)
        page = str(page_val) if page_val is not None else "?"

        level = level_idx.get(key)
        if level is None:
            for cs in context_snippets:
                if cs.doc_id == c.doc_id and getattr(cs, "level", None):
                    level = cs.level
                    break
        if level is None:
            level = "unknown"

        ref = f"[S{i}]"
        table.add_row(
            ref,
            doc_id_short,
            title,
            page,
            level,
        )

    console.print(table)
    console.print()


def print_metrics_table(metrics: Dict[str, float]) -> None:
    table = Table(
        title="[bold cyan]Agentic RAG – Evaluation Metrics[/bold cyan]",
        box=box.SIMPLE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Category", width=16)
    table.add_column("Metric", width=32)
    table.add_column("Value", justify="right", width=14)

    table.add_row("Retrieval", "Documents with snippets", str(metrics["num_docs"]))
    table.add_row("Retrieval", "Total snippets (context)", str(metrics["num_snippets"]))
    table.add_row("Retrieval", "Average snippet score (0–1)", f"{metrics['avg_score']:.4f}")
    table.add_row("Retrieval", "Max snippet score (0–1)", f"{metrics['max_score']:.4f}")

    table.add_row("Citations", "Number of citations", str(metrics["num_citations"]))
    table.add_row("Citations", "Unique cited documents", str(metrics["num_cited_docs"]))

    table.add_row("Answer", "Answer length (chars)", str(metrics["answer_chars"]))
    table.add_row("Answer", "Answer length (words)", str(metrics["answer_words"]))

    console.print(table)
    console.print()


def print_telemetry_table(events) -> None:
    if not events:
        console.print(
            Panel(
                "No telemetry events captured.",
                title="[bold yellow]Telemetry[/bold yellow]",
                border_style="yellow",
                box=box.ROUNDED,
            )
        )
        console.print()
        return

    table = Table(
        title="[bold cyan]Telemetry[/bold cyan]",
        box=box.SIMPLE,
        show_lines=True,
        expand=True,
    )
    table.add_column("Agent", width=20)
    table.add_column("Event", width=28)
    table.add_column("Phase", width=12)
    table.add_column("Elapsed (ms)", justify="right", width=14)
    table.add_column("Model", width=20)

    for e in events:
        agent = e.agent or "?"
        event_type = e.event_type or "?"
        phase = getattr(e.phase, "name", str(e.phase))
        elapsed_ms = (
            f"{e.timing.elapsed_ms:.2f}"
            if e.timing and e.timing.elapsed_ms is not None
            else "0.00"
        )
        model = e.model or "?"

        table.add_row(
            agent,
            event_type,
            phase,
            elapsed_ms,
            model,
        )

    console.print(table)
    console.print()


def smoke_test_agentic(
    queries: Optional[List[str]] = None,
    config_path: str | None = None,
) -> None:
    if queries is None:
        queries = [
            "What is hierarchical RAG?",
            "What is the Model Context Protocol (MCP)?",
        ]

    print_header(config_path, len(queries))

    register_default_agents(config_path=config_path)

    for i, q in enumerate(queries, start=1):
        console.rule(f"[bold magenta]Query {i}[/bold magenta]")
        print_query_block(i, q)

        TELEMETRY_EVENTS.clear()

        try:
            result = agentic_once_with_metadata(q)
        except Exception as e:
            err_panel = Panel(
                f"{type(e).__name__}: {e}",
                title="[bold red]ERROR in agentic_once_with_metadata[/bold red]",
                border_style="red",
                box=box.ROUNDED,
            )
            console.print(err_panel)
            console.print()
            continue

        answer = result["answer"]
        citations = result["citations"]
        context_snippets = result["context_snippets"]

        print_answer_block(answer)
        print_context_table(context_snippets)
        print_sources_table(citations, context_snippets)

        metrics = compute_simple_metrics(answer, citations, context_snippets)
        print_metrics_table(metrics)

        print_telemetry_table(TELEMETRY_EVENTS)

        console.rule("[bold green]End of Query[/bold green]")
        console.print()


if __name__ == "__main__":
    smoke_test_agentic()
