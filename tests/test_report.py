"""Run reporting: best_genome.json, candidates.csv, summary.md and summary.json."""

from __future__ import annotations

import csv
import json
import os

import pytest
import yaml

from zeroai.blocks import partition_tensors
from zeroai.candidate import CandidateGenerator
from zeroai.genome import Genome, genome_from_dict
from zeroai.optimizers.base import OptimizationResult
from zeroai.pipelines import CandidateRecord, PipelineResult
from zeroai.report import render_summary, write_report
from zeroai.schemas import (
    BlockSpec,
    DType,
    DomainSpec,
    EvaluationReport,
    MergeMethod,
    MetricScore,
    ModelSpec,
)

DOMAINS = (
    DomainSpec("reasoning", 0.6, "logic"),
    DomainSpec("code", 0.4, "syntax"),
)


def make_genome(n_models=2, n_blocks=2, **kwargs):
    kwargs.setdefault("method", MergeMethod.DARE_TIES)
    kwargs.setdefault("weights", [[0.2, 0.8] for _ in range(n_models)])
    kwargs.setdefault("density", 0.75)
    kwargs.setdefault("dare_density", 0.3)
    kwargs.setdefault("dare_scale", 1.5)
    kwargs.setdefault("normalize", True)
    return Genome(**kwargs)


def make_blocks(n_blocks=2, tensors_per_block=2):
    names = [f"block_{b}.t{j}" for b in range(n_blocks) for j in range(tensors_per_block)]
    return partition_tensors(names, n_blocks)


def make_report(candidate_id, capability=0.5, purity=0.4, extra=None):
    metrics = [MetricScore("capability", capability), MetricScore("purity", purity)]
    if extra is not None:
        metrics.append(MetricScore(extra, 0.25))
    return EvaluationReport(
        fitness=capability, metrics=tuple(metrics), candidate_id=candidate_id
    )


def make_result(tmp_path, optimization=True):
    best = make_genome()
    records = (
        CandidateRecord(
            candidate_id="cand_0001",
            genome=best,
            fitness=0.62,
            report=make_report("cand_0001", 0.62, 0.4),
            merge_seconds=1.25,
            eval_seconds=2.5,
            stats={"tensors": 4.0},
        ),
        CandidateRecord(
            candidate_id="cand_0002",
            genome=make_genome(weights=[[0.9, 0.1], [0.1, 0.9]], method=MergeMethod.LINEAR),
            fitness=0.31,
            report=make_report("cand_0002", 0.31, 0.2, extra="novel_metric"),
            merge_seconds=0.5,
            eval_seconds=0.75,
        ),
        CandidateRecord(
            candidate_id="cand_0003",
            genome=make_genome(density=0.1),
            fitness=0.0,
            error="merge: nan produced",
            merge_seconds=0.1,
        ),
    )
    optimization_result = (
        OptimizationResult(
            best_genome=best,
            best_fitness=0.62,
            evaluations=3,
            optimizer="genetic",
            history=(0.2, 0.4, 0.62),
            extra={"population_size": 4.0},
        )
        if optimization
        else None
    )
    return PipelineResult(
        best_genome=best,
        best_fitness=0.62,
        candidates=records,
        optimization=optimization_result,
        run_dir=str(tmp_path / "run"),
        mode="mock",
        blocks=make_blocks(),
        domains=DOMAINS,
        elapsed_seconds=3.5,
    )


def make_generator():
    models = (ModelSpec(path="models/a", label="a"), ModelSpec(path="models/b", label="b"))
    return CandidateGenerator(
        base_model=ModelSpec(path="models/base", label="base"),
        models=models,
        blocks=make_blocks(),
        dtype=DType.FLOAT16,
    )


# ------------------------------------------------------------------ artifacts
def test_write_report_produces_every_artifact(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"), generator=make_generator())

    assert set(paths) == {
        "best_genome",
        "mergekit_yaml",
        "candidates_csv",
        "summary",
        "summary_json",
        "run_dir",
    }
    for key, path in paths.items():
        target = path if key == "run_dir" else path
        assert os.path.exists(target), f"{key} missing at {target}"


def test_write_report_without_a_generator_skips_the_yaml(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"))
    assert "mergekit_yaml" not in paths
    assert os.path.isfile(paths["best_genome"])


def test_mergekit_yaml_artifact_is_valid_and_matches_the_best_genome(tmp_path):
    result = make_result(tmp_path)
    paths = write_report(result, str(tmp_path / "run"), generator=make_generator())
    with open(paths["mergekit_yaml"], "r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    assert payload["merge_method"] == "dare_ties"
    assert payload["parameters"]["density"] == pytest.approx(0.75)
    assert [model["model"] for model in payload["models"]] == ["models/a", "models/b"]
    # per-block coefficients survive as one scalar per block, after the plan's
    # column-wise normalization (each block's model weights sum to 1)
    assert payload["models"][0]["parameters"]["weight"] == pytest.approx([0.5, 0.5])
    assert payload["models"][1]["parameters"]["weight"] == pytest.approx([0.5, 0.5])
    assert payload["models"][0]["parameters"]["density"] == pytest.approx(0.7)  # 1 - dare_density
    assert payload["models"][0]["parameters"]["rescale_norm"] == "l2"


# ------------------------------------------------------------- best_genome.json
def test_best_genome_json_round_trips_the_winning_genome(tmp_path):
    result = make_result(tmp_path)
    paths = write_report(result, str(tmp_path / "run"))
    with open(paths["best_genome"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload["mode"] == "mock"
    assert payload["best_fitness"] == pytest.approx(0.62)
    assert payload["candidates"] == 3
    assert payload["failures"] == 1
    assert payload["elapsed_seconds"] == pytest.approx(3.5)
    assert [d["name"] for d in payload["domains"]] == ["reasoning", "code"]
    assert [b["tensors"] for b in payload["blocks"]] == [2, 2]

    restored = genome_from_dict(payload["best_genome"])
    assert restored.key() == result.best_genome.key()


def test_best_genome_json_records_the_optimisation_trace(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"))
    with open(paths["best_genome"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["optimization"]["optimizer"] == "genetic"
    assert payload["optimization"]["evaluations"] == 3
    assert payload["optimization"]["history"] == pytest.approx((0.2, 0.4, 0.62))


def test_best_genome_json_omits_the_optimisation_block_for_single_shot_runs(tmp_path):
    paths = write_report(make_result(tmp_path, optimization=False), str(tmp_path / "run"))
    with open(paths["best_genome"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "optimization" not in payload


# ---------------------------------------------------------------- candidates.csv
def test_candidates_csv_has_one_row_per_candidate(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"))
    with open(paths["candidates_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert len(rows) == 3
    assert [row["candidate_id"] for row in rows] == ["cand_0001", "cand_0002", "cand_0003"]
    assert rows[0]["method"] == "dare_ties"
    assert float(rows[0]["fitness"]) == pytest.approx(0.62)
    assert rows[2]["error"] == "merge: nan produced"
    assert rows[0]["error"] == ""


def test_candidates_csv_exposes_the_genome_columns(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"))
    with open(paths["candidates_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    header = list(rows[0].keys())
    assert "weight_m0_b0" in header and "weight_m1_b1" in header
    assert float(rows[0]["density"]) == pytest.approx(0.75)
    assert float(rows[0]["dare_density"]) == pytest.approx(0.3)
    assert float(rows[0]["dare_scale"]) == pytest.approx(1.5)
    assert rows[0]["normalize"] == "1"
    assert float(rows[0]["weight_m0_b0"]) == pytest.approx(0.2)
    assert float(rows[0]["merge_seconds"]) == pytest.approx(1.25)
    assert float(rows[0]["eval_seconds"]) == pytest.approx(2.5)


def test_candidates_csv_unions_metric_columns_across_candidates(tmp_path):
    paths = write_report(make_result(tmp_path), str(tmp_path / "run"))
    with open(paths["candidates_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    header = list(rows[0].keys())
    assert "capability" in header and "purity" in header and "novel_metric" in header
    # cand_0001 has no novel_metric, cand_0003 has no metrics at all.
    assert rows[0]["novel_metric"] == ""
    assert float(rows[1]["novel_metric"]) == pytest.approx(0.25)
    assert rows[2]["capability"] == ""


# ------------------------------------------------------------------ summary.md
def test_render_summary_covers_genome_metrics_and_convergence(tmp_path):
    summary = render_summary(make_result(tmp_path))

    assert "# ZeroAI run summary" in summary
    assert "- mode: `mock`" in summary
    assert "- failures: 1" in summary
    assert "- **best fitness: 0.620000**" in summary
    assert "- merge method: `dare_ties`" in summary
    assert "### Per-block coefficients" in summary
    assert "| model 0 |" in summary and "| model 1 |" in summary
    assert "## Top candidates" in summary
    assert "| cand_0001 | dare_ties | 0.620000 |" in summary
    assert "## Best candidate metrics" in summary
    assert "## Convergence" in summary
    assert "| 2 | 0.620000 |" in summary
    # Failed candidates never appear in the leaderboard.
    assert "cand_0003" not in summary.split("## Top candidates")[1].split("## Best candidate metrics")[0]


def test_render_summary_without_an_optimizer_omits_optimisation_sections(tmp_path):
    summary = render_summary(make_result(tmp_path, optimization=False))
    assert "optimizer:" not in summary
    assert "## Convergence" not in summary
    assert "**best fitness: 0.620000**" in summary


def test_render_summary_handles_a_run_where_everything_failed(tmp_path):
    result = make_result(tmp_path)
    failed = tuple(
        CandidateRecord(r.candidate_id, r.genome, 0.0, error="merge: boom") for r in result.candidates
    )
    broken = PipelineResult(
        best_genome=result.best_genome,
        best_fitness=0.0,
        candidates=failed,
        optimization=None,
        run_dir=result.run_dir,
        mode="mock",
        blocks=result.blocks,
        domains=result.domains,
        elapsed_seconds=1.0,
    )
    summary = render_summary(broken)
    assert "- failures: 3" in summary
    assert "## Top candidates" not in summary
    assert "## Best candidate metrics" not in summary


def test_write_report_summary_md_matches_render_summary(tmp_path):
    result = make_result(tmp_path)
    paths = write_report(result, str(tmp_path / "run"))
    with open(paths["summary"], "r", encoding="utf-8") as handle:
        on_disk = handle.read()
    assert on_disk == render_summary(result)


def test_summary_json_is_the_full_result_dump(tmp_path):
    result = make_result(tmp_path)
    paths = write_report(result, str(tmp_path / "run"))
    with open(paths["summary_json"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)

    assert payload == json.loads(json.dumps(result.to_dict()))
    assert payload["failures"] == 1
    assert len(payload["candidates"]) == 3
    assert payload["candidates"][0]["metrics"]["capability"] == pytest.approx(0.62)
    assert payload["candidates"][2]["error"] == "merge: nan produced"


def test_write_report_is_idempotent_and_creates_missing_directories(tmp_path):
    run_dir = str(tmp_path / "deep" / "nested" / "run")
    first = write_report(make_result(tmp_path), run_dir)
    second = write_report(make_result(tmp_path), run_dir)
    assert first["best_genome"] == second["best_genome"]
    assert os.path.isfile(second["candidates_csv"])


def test_block_specs_are_reported_with_their_tensor_counts(tmp_path):
    blocks = [BlockSpec(index=0, name="block_0", tensor_names=("a", "b", "c"))]
    result = PipelineResult(
        best_genome=make_genome(n_blocks=1, weights=[[1.0], [0.5]]),
        best_fitness=0.1,
        candidates=(),
        optimization=None,
        run_dir=str(tmp_path / "run"),
        mode="mock",
        blocks=tuple(blocks),
        domains=DOMAINS,
        elapsed_seconds=0.0,
    )
    paths = write_report(result, str(tmp_path / "run"))
    with open(paths["best_genome"], "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["blocks"] == [{"index": 0, "name": "block_0", "tensors": 3}]
    assert payload["candidates"] == 0
    with open(paths["candidates_csv"], newline="", encoding="utf-8") as handle:
        rows = list(csv.reader(handle))
    assert len(rows) == 1  # header only
    assert "| model 0 |" in render_summary(result)