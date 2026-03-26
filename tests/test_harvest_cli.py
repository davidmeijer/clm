import argparse
from pathlib import Path

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
