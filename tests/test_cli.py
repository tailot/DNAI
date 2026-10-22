"""CLI surface: argument parsing, genome flags, and the run / yaml / merge / doctor commands."""

from __future__ import annotations

import json
import os

import pytest
import yaml

from zeroai import cli
from zeroai.genome import Genome, genome_to_dict
from zeroai.pipelines import PipelineConfig
from zeroai.schemas import DType, MergeMethod, SearchSpace


def parse(*argv):
    return cli.build_parser().parse_args(list(argv))


# ------------------------------------------------------------------ structure
def test_a_subcommand_is_required():
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([])


def test_run_defaults_match_the_documented_workflow():
    args = parse("run", "--mode", "mock")
    assert args.command == "run"
    assert args.optimizer == "optuna"
    assert args.trials == 40
    assert args.population == 20
    assert args.generations == 8
    assert args.blocks == 4
    assert args.dtype == "float16"
    assert args.merge_backend == "native"
    assert args.write_checkpoints is False
    assert args.normalize is None
    assert args.func is cli.cmd_run


def test_each_subcommand_maps_to_its_handler():
    assert parse("yaml", "--mode", "mock").func is cli.cmd_yaml
    assert parse("merge", "--mode", "mock").func is cli.cmd_merge
    assert parse("doctor").func is cli.cmd_doctor


def test_unknown_mode_is_rejected():
    with pytest.raises(SystemExit):
        parse("run", "--mode", "quantum")


# --------------------------------------------------------------- arg mapping
def test_build_config_translates_flags_into_a_pipeline_config(tmp_path):
    args = parse(
        "run",
        "--mode",
        "mock",
        "--models",
        "experts/alpha",
        "experts/beta/",
        "--blocks",
        "3",
        "--dtype",
        "bfloat16",
        "--seed",
        "7",
        "--run-dir",
        str(tmp_path),
        "--run-name",
        "cli",
        "--optimizer",
        "genetic",
        "--population",
        "6",
        "--generations",
        "2",
        "--trials",
        "11",
        "--merge-backend",
        "mergekit",
        "--mergekit-python",
        "/venv/bin/python",
        "--write-checkpoints",
    )
    config = cli.build_config(args)
    assert isinstance(config, PipelineConfig)
    assert config.mode == "mock"
    assert config.num_blocks == 3
    assert config.dtype is DType.BFLOAT16
    assert config.seed == 7
    assert config.run_dir == str(tmp_path)
    assert config.run_name == "cli"
    assert config.optimizer == "genetic"
    assert config.population_size == 6
    assert config.generations == 2
    assert config.n_trials == 11
    assert config.merge_backend == "mergekit"
    assert config.mergekit_python == "/venv/bin/python"
    assert config.write_checkpoints is True
    assert [model.label for model in config.models] == ["alpha", "beta"]


def test_search_space_flags_restrict_the_genome_genes():
    args = parse(
        "run",
        "--methods",
        "ties",
        "dare_ties",
        "--weight-low",
        "0.1",
        "--weight-high",
        "0.9",
        "--density-low",
        "0.2",
        "--density-high",
        "0.8",
        "--dare-low",
        "0.05",
        "--dare-high",
        "0.5",
    )
    space = cli._search_space(args)
    assert space.methods == (MergeMethod.TIES, MergeMethod.DARE_TIES)
    assert (space.weight_low, space.weight_high) == (0.1, 0.9)
    assert (space.density_low, space.density_high) == (0.2, 0.8)
    assert (space.dare_density_low, space.dare_density_high) == (0.05, 0.5)


def test_a_single_method_flag_also_restricts_the_search_space():
    space = cli._search_space(parse("yaml", "--method", "taskslim"))
    assert space.methods == (MergeMethod.TASKSLIM,)


def test_an_unknown_method_name_is_a_value_error():
    with pytest.raises(ValueError):
        cli._search_space(parse("yaml", "--methods", "not_a_method"))


def test_bounds_are_validated_by_the_search_space():
    with pytest.raises(ValueError, match="weight_low"):
        cli._search_space(parse("yaml", "--weight-low", "2.0", "--weight-high", "0.5"))


# ------------------------------------------------------------ genome flags
def test_weights_shared_by_every_model(tmp_path):
    args = parse("yaml", "--blocks", "3", "--models", "a", "b", "c", "--weights", "0.2", "0.5", "0.8")
    genome = cli._explicit_genome(args, n_models=3, n_blocks=3, space=SearchSpace())
    assert [list(row) for row in genome.weights] == [[0.2, 0.5, 0.8]] * 3
    assert genome.method is MergeMethod.LINEAR  # no --method means plain linear


def test_model_major_weights_are_split_row_by_row():
    args = parse(
        "yaml",
        "--blocks",
        "2",
        "--models",
        "a",
        "b",
        "--method",
        "dare_ties",
        "--weights",
        "0.1",
        "0.9",
        "0.7",
        "0.3",
        "--density",
        "0.6",
        "--dare-density",
        "0.25",
        "--dare-scale",
        "1.4",
    )
    genome = cli._explicit_genome(args, n_models=2, n_blocks=2, space=SearchSpace())
    assert [list(row) for row in genome.weights] == [[0.1, 0.9], [0.7, 0.3]]
    assert genome.method is MergeMethod.DARE_TIES
    assert genome.density == pytest.approx(0.6)
    assert genome.dare_density == pytest.approx(0.25)
    assert genome.dare_scale == pytest.approx(1.4)
    assert genome.normalize is True  # default when neither flag is given


def test_normalize_flags_are_honoured():
    off = parse("yaml", "--blocks", "1", "--weights", "0.5", "--no-normalize")
    assert cli._explicit_genome(off, 1, 1, SearchSpace()).normalize is False
    on = parse("yaml", "--blocks", "1", "--weights", "0.5", "--normalize")
    assert cli._explicit_genome(on, 1, 1, SearchSpace()).normalize is True


def test_weights_without_a_method_stay_uniform_random():
    args = parse("yaml", "--blocks", "2")
    genome = cli._explicit_genome(args, n_models=3, n_blocks=2, space=SearchSpace())
    assert isinstance(genome, Genome)
    assert len(genome.weights) == 3 and all(len(row) == 2 for row in genome.weights)


def test_a_wrong_weight_count_fails_with_a_helpful_message():
    args = parse("yaml", "--blocks", "2", "--models", "a", "b", "--weights", "0.1", "0.2", "0.3")
    with pytest.raises(SystemExit, match="needs 2 values"):
        cli._explicit_genome(args, n_models=2, n_blocks=2, space=SearchSpace())


def test_explicit_genomes_are_clipped_to_the_search_space():
    args = parse("yaml", "--blocks", "1", "--weights", "9.0", "--dare-scale", "99.0")
    space = SearchSpace(weight_low=0.0, weight_high=1.0, dare_scale_low=1.0, dare_scale_high=2.0)
    genome = cli._explicit_genome(args, n_models=1, n_blocks=1, space=space)
    assert genome.weights[0][0] <= 1.0
    assert genome.dare_scale <= 2.0


def test_initial_genomes_come_from_weights_or_from_a_genome_file(tmp_path):
    from_flags = parse("run", "--blocks", "2", "--models", "a", "b", "--method", "ties", "--weights", "0.4", "0.6")
    initial = cli._initial_genomes(from_flags, 2, 2)
    assert len(initial) == 1 and initial[0].method is MergeMethod.TIES

    empty = parse("run", "--blocks", "2")
    assert cli._initial_genomes(empty, 2, 2) == ()

    payload = genome_to_dict(
        Genome(
            method=MergeMethod.DARE_LINEAR,
            weights=[[0.3, 0.4]],
            density=0.7,
            dare_density=0.1,
            dare_scale=1.2,
        )
    )
    genome_file = tmp_path / "genome.json"
    genome_file.write_text(json.dumps(payload), encoding="utf-8")
    from_file = parse("run", "--genome", str(genome_file))
    loaded = cli._initial_genomes(from_file)
    assert loaded[0].method is MergeMethod.DARE_LINEAR
    assert tuple(loaded[0].weights[0]) == pytest.approx((0.3, 0.4))
    assert loaded[0].density == pytest.approx(0.7)


# ------------------------------------------------------------------ commands
def test_yaml_command_emits_a_parseable_mergekit_config(capsys):
    code = cli.main(
        [
            "yaml",
            "--mode",
            "mock",
            "--models",
            "alpha",
            "beta",
            "--blocks",
            "2",
            "--method",
            "dare_ties",
            "--weights",
            "0.2",
            "0.8",
            "--no-normalize",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    document = yaml.safe_load(capsys.readouterr().out)
    assert document["merge_method"] == "dare_ties"
    assert len(document["models"]) == 2
    assert document["models"][0]["parameters"]["weight"] == pytest.approx([0.2, 0.8])


def test_yaml_command_falls_back_to_a_uniform_genome(capsys):
    assert cli.main(["yaml", "--mode", "mock", "--blocks", "3", "--log-level", "WARNING"]) == 0
    document = yaml.safe_load(capsys.readouterr().out)
    assert document["merge_method"] == "linear"
    assert len(document["models"]) == 3  # mock world default: three experts


def test_run_command_executes_the_loop_and_writes_artifacts(capsys, tmp_path):
    code = cli.main(
        [
            "run",
            "--mode",
            "mock",
            "--optimizer",
            "genetic",
            "--population",
            "5",
            "--generations",
            "2",
            "--blocks",
            "2",
            "--seed",
            "5",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "cli_run",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    out = capsys.readouterr().out
    assert "# ZeroAI run summary" in out
    assert "artifacts:" in out
    assert "best_genome:" in out

    run_dir = tmp_path / "cli_run"
    for name in ("best_genome.json", "candidates.csv", "summary.md", "summary.json", "best_mergekit.yaml"):
        assert os.path.isfile(os.path.join(str(run_dir), name))
    payload = json.loads((run_dir / "best_genome.json").read_text(encoding="utf-8"))
    assert payload["mode"] == "mock"
    assert payload["optimization"]["optimizer"] == "genetic"
    assert payload["best_fitness"] > 0.0


def test_run_command_with_optimizer_none_evaluates_a_single_genome(capsys, tmp_path):
    code = cli.main(
        [
            "run",
            "--mode",
            "mock",
            "--optimizer",
            "none",
            "--blocks",
            "2",
            "--models",
            "a",
            "b",
            "--method",
            "ties",
            "--weights",
            "0.5",
            "0.5",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "single",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    payload = json.loads(
        (tmp_path / "single" / "best_genome.json").read_text(encoding="utf-8")
    )
    assert payload["candidates"] == 1
    assert payload["best_genome"]["method"] == "ties"
    assert "optimization" not in payload


def test_run_command_accepts_a_seed_genome_file(capsys, tmp_path):
    genome = Genome(
        method=MergeMethod.DARE_TIES,
        weights=[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5]],
        density=0.5,
        dare_density=0.3,
        dare_scale=2.0,
    )
    genome_file = tmp_path / "seed.json"
    genome_file.write_text(json.dumps(genome_to_dict(genome)), encoding="utf-8")

    code = cli.main(
        [
            "run",
            "--mode",
            "mock",
            "--optimizer",
            "none",
            "--blocks",
            "2",
            "--genome",
            str(genome_file),
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "seeded",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "seeded" / "best_genome.json").read_text(encoding="utf-8"))
    assert payload["best_genome"]["method"] == "dare_ties"
    assert payload["best_genome"]["dare_scale"] == pytest.approx(2.0)


def test_merge_command_forces_a_single_native_merge_with_checkpoints(capsys, tmp_path):
    code = cli.main(
        [
            "merge",
            "--mode",
            "mock",
            "--models",
            "a",
            "b",
            "--method",
            "dare_ties",
            "--density",
            "0.6",
            "--blocks",
            "2",
            "--run-dir",
            str(tmp_path),
            "--run-name",
            "merged",
            "--log-level",
            "WARNING",
        ]
    )
    assert code == 0
    run_dir = tmp_path / "merged"
    candidates = run_dir / "candidates"
    assert candidates.is_dir()
    checkpoints = [d for d in candidates.iterdir() if (d / "model.safetensors").exists()]
    assert len(checkpoints) == 1
    assert "artifacts:" in capsys.readouterr().out


def test_model_mode_without_a_base_model_is_reported(tmp_path):
    with pytest.raises(ValueError, match="base_model"):
        cli.main(
            [
                "run",
                "--mode",
                "model",
                "--run-dir",
                str(tmp_path),
                "--log-level",
                "WARNING",
            ]
        )


def test_doctor_reports_the_environment_as_json(capsys):
    assert cli.main(["doctor"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["python"]
    assert "torch" in report and "optuna" in report and "yaml" in report
    assert isinstance(report["cuda_available"], bool)
    assert isinstance(report["mergekit_in_process"], bool)