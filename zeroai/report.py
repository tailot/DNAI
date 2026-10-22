"""Module 6: experiment reporting."""

from __future__ import annotations

import csv
import json
import os
from typing import Dict, List, Optional

from zeroai.genome import genome_to_dict


def write_report(result, run_dir: str, generator=None) -> Dict[str, str]:
    """Persist a run as JSON + CSV + a human-readable summary.  Returns the paths."""
    os.makedirs(run_dir, exist_ok=True)
    paths: Dict[str, str] = {}

    best_path = os.path.join(run_dir, "best_genome.json")
    payload = {
        "mode": result.mode,
        "best_fitness": result.best_fitness,
        "best_genome": genome_to_dict(result.best_genome),
        "elapsed_seconds": result.elapsed_seconds,
        "candidates": len(result.candidates),
        "failures": sum(1 for c in result.candidates if c.error),
        "domains": [
            {"name": d.name, "weight": d.weight, "description": d.description} for d in result.domains
        ],
        "blocks": [
            {"index": b.index, "name": b.name, "tensors": len(b.tensor_names)} for b in result.blocks
        ],
    }
    if result.optimization is not None:
        payload["optimization"] = {
            "optimizer": result.optimization.optimizer,
            "evaluations": result.optimization.evaluations,
            "best_fitness": result.optimization.best_fitness,
            "history": list(result.optimization.history),
        }
    with open(best_path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    paths["best_genome"] = best_path

    if generator is not None:
        yaml_path = os.path.join(run_dir, "best_mergekit.yaml")
        generator.write_yaml(result.best_genome, yaml_path, candidate_id="best")
        paths["mergekit_yaml"] = yaml_path

    csv_path = os.path.join(run_dir, "candidates.csv")
    _write_csv(result, csv_path)
    paths["candidates_csv"] = csv_path

    summary_path = os.path.join(run_dir, "summary.md")
    with open(summary_path, "w", encoding="utf-8") as handle:
        handle.write(render_summary(result))
    paths["summary"] = summary_path

    full_path = os.path.join(run_dir, "summary.json")
    with open(full_path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2, sort_keys=True)
    paths["summary_json"] = full_path
    paths["run_dir"] = run_dir
    return paths


def _metric_names(result) -> List[str]:
    names: List[str] = []
    for record in result.candidates:
        if record.report is None:
            continue
        for metric in record.report.metrics:
            if metric.name not in names:
                names.append(metric.name)
    return names


def _write_csv(result, path: str) -> None:
    metric_names = _metric_names(result)
    n_blocks = len(result.blocks)
    header = [
        "candidate_id",
        "method",
        "fitness",
        "density",
        "dare_density",
        "dare_scale",
        "normalize",
    ]
    header += [f"weight_m{m}_b{b}" for m in range(max(1, result.best_genome.n_models)) for b in range(n_blocks)]
    header += metric_names
    header += ["merge_seconds", "eval_seconds", "error"]

    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for record in result.candidates:
            genome = record.genome
            row = [
                record.candidate_id,
                genome.method.value,
                f"{record.fitness:.6f}",
                f"{genome.density:.6f}",
                f"{genome.dare_density:.6f}",
                f"{genome.dare_scale:.6f}",
                int(bool(genome.normalize)),
            ]
            for model_row in genome.weights:
                for block_index in range(n_blocks):
                    row.append(f"{model_row[block_index]:.6f}" if block_index < len(model_row) else "")
            metrics = {m.name: m.value for m in (record.report.metrics if record.report else ())}
            for name in metric_names:
                value = metrics.get(name)
                row.append("" if value is None else f"{value:.6f}")
            row += [f"{record.merge_seconds:.3f}", f"{record.eval_seconds:.3f}", record.error or ""]
            writer.writerow(row)


def render_summary(result) -> str:
    genome = result.best_genome
    lines: List[str] = []
    lines.append("# ZeroAI run summary")
    lines.append("")
    lines.append(f"- mode: `{result.mode}`")
    lines.append(f"- evaluator domains: {', '.join(d.name for d in result.domains)}")
    lines.append(f"- blocks: {len(result.blocks)}")
    lines.append(f"- candidates evaluated: {len(result.candidates)}")
    lines.append(f"- failures: {sum(1 for c in result.candidates if c.error)}")
    lines.append(f"- elapsed: {result.elapsed_seconds:.1f}s")
    if result.optimization is not None:
        lines.append(
            f"- optimizer: `{result.optimization.optimizer}` "
            f"({result.optimization.evaluations} evaluations)"
        )
    lines.append(f"- **best fitness: {result.best_fitness:.6f}**")
    lines.append("")
    lines.append("## Best genome")
    lines.append("")
    lines.append(f"- merge method: `{genome.method.value}`")
    lines.append(f"- density (retained): {genome.density:.4f}")
    lines.append(f"- DARE drop: {genome.dare_density:.4f} | DARE scale: {genome.dare_scale:.4f}")
    lines.append(f"- normalize: {genome.normalize}")
    lines.append("")
    lines.append("### Per-block coefficients")
    lines.append("")
    header = "| model | " + " | ".join(f"block {b}" for b in range(genome.n_blocks)) + " |"
    lines.append(header)
    lines.append("|" + "---|" * (genome.n_blocks + 1))
    for model_index, row in enumerate(genome.weights):
        lines.append(
            f"| model {model_index} | " + " | ".join(f"{v:.4f}" for v in row) + " |"
        )
    lines.append("")

    top = result.top(5)
    if top:
        lines.append("## Top candidates")
        lines.append("")
        lines.append("| candidate | method | fitness | merge s | eval s |")
        lines.append("|---|---|---|---|---|")
        for record in top:
            lines.append(
                f"| {record.candidate_id} | {record.genome.method.value} | "
                f"{record.fitness:.6f} | {record.merge_seconds:.2f} | {record.eval_seconds:.2f} |"
            )
        lines.append("")

    best_metrics = {}
    for record in top[:1]:
        if record.report is not None:
            best_metrics = {m.name: m.value for m in record.report.metrics}
    if best_metrics:
        lines.append("## Best candidate metrics")
        lines.append("")
        lines.append("| metric | value |")
        lines.append("|---|---|")
        for name, value in best_metrics.items():
            lines.append(f"| {name} | {value:.6f} |")
        lines.append("")

    if result.optimization is not None and result.optimization.history:
        history = result.optimization.history
        lines.append("## Convergence")
        lines.append("")
        lines.append("| step | best fitness |")
        lines.append("|---|---|")
        step = max(1, len(history) // 12)
        for index in range(0, len(history), step):
            lines.append(f"| {index} | {history[index]:.6f} |")
        if (len(history) - 1) % step:
            lines.append(f"| {len(history) - 1} | {history[-1]:.6f} |")
        lines.append("")

    return "\n".join(lines)