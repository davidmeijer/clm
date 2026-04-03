import argparse
from pathlib import Path

import harvest.cli as harvest_cli
from harvest.cli import _submit_via_slurm


def test_submit_via_slurm_dry_run_builds_wrap_command(tmp_path, capsys):
    args = argparse.Namespace(
        part="skinniderlabhp",
        cpus=8,
        mem="32G",
        time="72:00:00",
        gres="gpu:1",
        job_name="harvest_lipo_test",
        dry_run=True,
        out_dir=str(tmp_path),
    )

    _submit_via_slurm(
        args,
        [
            "train",
            "--configfile",
            "./workflow/config/config_lipopeptides.yml",
            "--jobs",
            "3",
        ],
    )

    output = capsys.readouterr().out
    logs_dir = tmp_path / "logs"

    assert logs_dir.is_dir()
    assert "--wrap" in output
    assert str(logs_dir / "harvest_%x_%j.out") in output
    assert "python -m harvest.cli train" in output


def test_submit_via_slurm_train_uses_paths_output_dir_override(tmp_path, capsys):
    run_dir = tmp_path / "run"
    configfile = tmp_path / "config.yaml"
    configfile.write_text("paths:\n  output_dir: ./fallback_out\n")

    args = argparse.Namespace(
        cmd="train",
        part="skinniderlabhp",
        cpus=8,
        mem="32G",
        time="72:00:00",
        gres="gpu:1",
        job_name="harvest_lipo_test",
        dry_run=True,
        out_dir=None,
        configfile=str(configfile),
        workflow_dir=None,
        snakefile=None,
        snakemake_args=[
            "--config",
            f"paths={{output_dir: {run_dir}, dataset: /tmp/data.jsonl}}",
            "enum_factors=[10]",
            "--until",
            "train_models_RNN",
        ],
    )

    _submit_via_slurm(
        args,
        [
            "train",
            "--configfile",
            str(configfile),
            "--jobs",
            "3",
            "--snakemake-args",
            "--config",
            f"paths={{output_dir: {run_dir}, dataset: /tmp/data.jsonl}}",
            "enum_factors=[10]",
            "--until",
            "train_models_RNN",
        ],
    )

    output = capsys.readouterr().out
    logs_dir = run_dir / "logs"

    assert logs_dir.is_dir()
    assert str(logs_dir / "harvest_%x_%j.out") in output


def test_validate_cli_passes_batch_size(monkeypatch, tmp_path):
    captured = {}

    def fake_validate(**kwargs):
        captured.update(kwargs)

    monkeypatch.setattr(harvest_cli, "cmd_validate", fake_validate)

    harvest_cli.cli(
        [
            "validate",
            "--out-dir",
            str(tmp_path),
            "--model-dir",
            str(tmp_path / "model"),
            "--device",
            "cpu",
            "--test-size",
            "10",
            "--sample-size",
            "20",
            "--batch-size",
            "32",
        ]
    )

    assert captured["out_dir"] == str(tmp_path)
    assert captured["batch_size"] == 32
    assert captured["sample_size"] == 20
