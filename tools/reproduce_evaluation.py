"""Run the unchanged evaluators sequentially in an exclusively new workspace."""
import argparse
import os
from pathlib import Path
import subprocess
import sys


INPUTS = (
    "src", "tests", "configs", "splits", "data", "requirements.txt",
    "requirements-lofo.txt", "requirements-local-lock.txt",
    "results/controlled_v2", "results/research_audit", "results/clean_baseline",
    "results/lofo_noise", "results/lofo_blur", "results/lofo_weather", "results/lofo_digital",
)


def prepare_workspace(research_root, output):
    root = Path(research_root).expanduser().resolve(strict=True)
    destination = Path(output).expanduser().absolute()
    # Resolve parents as well: a symlink must not redirect outputs into the inputs.
    resolved = destination.resolve()
    if resolved == root or resolved.is_relative_to(root) or root.is_relative_to(resolved):
        raise ValueError("Output must be outside the research repository and not its ancestor.")
    for relative in INPUTS:
        if not (root / relative).exists():
            raise FileNotFoundError(f"Required research input missing: {relative}")
    if destination.exists() or destination.is_symlink():
        raise FileExistsError(f"Output must not exist: {destination}")
    destination.mkdir(exist_ok=False)
    workspace = destination / "workspace"
    workspace.mkdir()
    for relative in INPUTS:
        target = workspace / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to(root / relative, target_is_directory=(root / relative).is_dir())
    return workspace, destination


def reproduce(research_root, output):
    workspace, destination = prepare_workspace(research_root, output)
    commands = (
        ("historical", [sys.executable, "-B", "-m", "src.benchmark", "full", "--device", "mps",
                        "--output", str(destination / "historical")]),
        ("controlled_v2", [sys.executable, "-B", "-m", "src.controlled_v2", "evaluate"]),
    )
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    for name, command in commands:
        print(f"Evaluating {name}; log: {destination / (name + '.log')}", flush=True)
        with (destination / f"{name}.log").open("x") as log:
            subprocess.run(command, cwd=workspace, env=environment, stdout=log,
                           stderr=subprocess.STDOUT, check=True)
    print(f"Evaluation reproduction completed in {destination}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--research-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", help="Explicitly enable full model inference")
    args = parser.parse_args()
    if not args.execute:
        parser.error("Full inference requires --execute; no files were created.")
    reproduce(args.research_root, args.output)


if __name__ == "__main__":
    main()
