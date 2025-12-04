from __future__ import annotations

import argparse
from typing import List, Any, Dict

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

from orchestrator import (
    register_default_agents,
    agentic_once_with_metadata,
)

console = Console()


# ---------------------------------------------------------------------------
# Pretty-printers for each agent
# ---------------------------------------------------------------------------

def _panel(title: str, body_lines: List[str]) -> Panel:
    return Panel(
        "\n".join(body_lines) if body_lines else "[dim]No data[/dim]",
        title=title,
        border_style="cyan",
        box=box.ROUNDED,
    )


def print_router(router_out: Any) -> None:
    if router_out is None:
        console.print(_panel("[bold magenta]RouterAgent[/bold magenta]", ["<no output>"]))
        return
    rp = getattr(router_out, "router_profile", router_out)
    lines = [
        f"[bold]Route[/bold] : {getattr(rp, 'route', '<unknown>')}",
        f"• use_qe     : {getattr(rp, 'use_qe', None)}",
        f"• use_prf    : {getattr(rp, 'use_prf', None)}",
        f"• use_rerank : {getattr(rp, 'use_rerank', None)}",
        f"• complexity : {getattr(rp, 'complexity_hint', None)}",
    ]
    console.print(_panel("[bold magenta]RouterAgent[/bold magenta]", lines))
    console.print()


def print_decomposition(decomp_out: Any) -> None:
    if decomp_out is None:
        console.print(_panel("[bold magenta]DecompositionAgent[/bold magenta]", ["<no output>"]))
        console.print()
        return

    dec = getattr(decomp_out, "decomposition", decomp_out)
    subqs = getattr(dec, "subqueries", []) or []
    cmps = getattr(dec, "comparison_pairs", []) or []

    lines = [
        f"Is multi-part : {getattr(dec, 'is_multi_part', False)}",
        f"Num subqueries: {len(subqs)}",
        f"Num pairs     : {len(cmps)}",
        "",
        "[bold]Subqueries[/bold]",
    ]
    for sq in subqs:
        sid = getattr(sq, "id", "")
        txt = getattr(sq, "text", "")
        lines.append(f"  - {sid}: {txt}")

    if cmps:
        lines.append("")
        lines.append("[bold]Comparison pairs[/bold]")
        for cp in cmps:
            left = getattr(cp, "left", "")
            right = getattr(cp, "right", "")
            lines.append(f"  - {left}  vs  {right}")

    console.print(_panel("[bold magenta]DecompositionAgent[/bold magenta]", lines))
    console.print()


def print_plan(plan: Any) -> None:
    if plan is None:
        console.print(_panel("[bold magenta]PlannerAgent[/bold magenta]", ["<no plan>"]))
        console.print()
        return

    iterations = getattr(plan, "iterations", None)
    lines = [
        f"Retrieval mode : {getattr(plan, 'retrieval_mode', None)}",
        f"use_qe         : {getattr(plan, 'use_qe', None)}",
        f"use_prf        : {getattr(plan, 'use_prf', None)}",
        f"use_rerank     : {getattr(plan, 'use_rerank', None)}",
        f"top_k          : {getattr(plan, 'top_k', None)}",
        f"rerank_top_k   : {getattr(plan, 'rerank_top_k', None)}",
        f"language       : {getattr(plan, 'language', None)}",
        f"backend        : {getattr(plan, 'backend', None)}",
        f"num_logical_parts: {getattr(plan, 'num_logical_parts', 1)}",
    ]
    if iterations is not None:
        lines.append("")
        lines.append("[bold]Iterations[/bold]")
        lines.append(f"  max_iters    : {getattr(iterations, 'max_iters', None)}")
        lines.append(f"  max_rewrites : {getattr(iterations, 'max_rewrites', None)}")

    console.print(_panel("[bold magenta]PlannerAgent[/bold magenta]", lines))
    console.print()


def print_guardrail(guard_out: Any) -> None:
    if guard_out is None:
        console.print(_panel("[bold magenta]GuardrailAgent[/bold magenta]", ["<no output>"]))
        console.print()
        return

    lines = [
        f"Status   : {getattr(guard_out, 'status', None)}",
        "",
    ]
    msgs = getattr(guard_out, "messages", []) or []
    if msgs:
        lines.append("[bold]Messages[/bold]")
        for m in msgs:
            lines.append(f"  - {m}")

    console.print(_panel("[bold magenta]GuardrailAgent[/bold magenta]", lines))
    console.print()


def print_prf(prf_terms: Any) -> None:
    terms = prf_terms or []
    lines = [f"Num PRF terms: {len(terms)}"]
    for t in terms[:20]:
        lines.append(f"  - {t}")
    console.print(_panel("[bold magenta]PRFAgent[/bold magenta]", lines))
    console.print()


def print_qe(qe_expansions: Any) -> None:
    exps = qe_expansions or []
    lines = [f"Num QE expansions: {len(exps)}"]
    for e in exps[:10]:
        lines.append(f"  - {e}")
    console.print(_panel("[bold magenta]QEAgent[/bold magenta]", lines))
    console.print()


def print_retrieval(results: Any) -> None:
    rows = results or []
    table = Table(
        title="RetrieverAgent – top results",
        box=box.SIMPLE_HEAVY,
        show_lines=False,
    )
    table.add_column("Rank", justify="right", style="cyan", no_wrap=True)
    table.add_column("Score", justify="right")
    table.add_column("Doc ID", overflow="fold")
    table.add_column("Title/Page", overflow="fold")

    for idx, r in enumerate(rows[:20], start=1):
        score = getattr(r, "score", None)
        doc_id = getattr(r, "doc_id", "")
        meta = getattr(r, "meta", {}) or {}
        title = meta.get("title", "")
        page = meta.get("page", "")
        tp = f"{title} (p.{page})" if page not in (None, "") else title
        table.add_row(str(idx), f"{score:.3f}" if score is not None else "-", str(doc_id), tp)

    console.print(table)
    console.print()


def print_answer(answer: Any) -> None:
    if answer is None:
        console.print(_panel("[bold magenta]GeneratorAgent[/bold magenta]", ["<no answer>"]))
        console.print()
        return

    text = getattr(answer, "text", "") or ""
    lines = [
        "[bold]Final answer[/bold]",
        "",
        text,
    ]
    console.print(_panel("[bold magenta]GeneratorAgent[/bold magenta]", lines))
    console.print()


def print_critic(critic_out: Any) -> None:
    if critic_out is None:
        console.print(_panel("[bold magenta]CriticAgent[/bold magenta]", ["<no critic feedback>"]))
        console.print()
        return

    lines = [
        f"Label   : {getattr(critic_out, 'label', None)}",
        "",
        "[bold]Notes[/bold]",
    ]
    for note in getattr(critic_out, "notes", []) or []:
        lines.append(f"  - {note}")

    console.print(_panel("[bold magenta]CriticAgent[/bold magenta]", lines))
    console.print()


def print_policy(policy_out: Any) -> None:
    if policy_out is None:
        console.print(_panel("[bold magenta]PolicyAgent[/bold magenta]", ["<no policy decision>"]))
        console.print()
        return

    lines = [
        f"Decision : {getattr(policy_out, 'decision', None)}",
        "",
    ]
    for note in getattr(policy_out, "notes", []) or []:
        lines.append(f"  - {note}")

    console.print(_panel("[bold magenta]PolicyAgent[/bold magenta]", lines))
    console.print()


def print_postprocess(pp_out: Any) -> None:
    if pp_out is None:
        console.print(_panel("[bold magenta]PostProcessorAgent[/bold magenta]", ["<no output>"]))
        console.print()
        return

    # We just show the final rendered text and a couple of metadata fields.
    text = getattr(pp_out, "final_text", "") or ""
    lines = [
        "[bold]Postprocessed answer[/bold]",
        "",
        text,
    ]
    console.print(_panel("[bold magenta]PostProcessorAgent[/bold magenta]", lines))
    console.print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Inspect the behavior of each Agentic RAG agent individually.",
    )
    parser.add_argument(
        "--config",
        "-c",
        default="config.fast.yaml",
        help="Path to config.fast.yaml (default: config.fast.yaml)",
    )
    parser.add_argument(
        "--query",
        "-q",
        action="append",
        required=True,
        help="User query to run. Can be specified multiple times.",
    )

    args = parser.parse_args()

    register_default_agents(config_path=args.config)

    for idx, query in enumerate(args.query, start=1):
        console.rule(f"[bold green]Query {idx}[/bold green]")
        console.print(f"[bold]User query:[/bold] {query}\n")

        meta = agentic_once_with_metadata(query)

        print_router(meta.get("router"))
        print_decomposition(meta.get("decomposition"))
        print_plan(meta.get("plan"))
        print_guardrail(meta.get("guardrail"))
        print_prf(meta.get("prf_terms"))
        print_qe(meta.get("qe_expansions"))
        print_retrieval(meta.get("retrieval_results"))
        print_answer(meta.get("answer"))
        print_critic(meta.get("critic"))
        print_policy(meta.get("policy"))
        print_postprocess(meta.get("postprocess"))

        console.rule("[bold green]End of query[/bold green]")
        console.print()


if __name__ == "__main__":
    main()
