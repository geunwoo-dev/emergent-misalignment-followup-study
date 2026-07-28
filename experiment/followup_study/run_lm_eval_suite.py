from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


DEFAULT_TASKS = [
    "truthfulqa_mc1",
    "truthfulqa_mc2",
    "medqa_4options",
    "gsm8k_cot",
    "mbpp",
]


def model_args(record: dict, load_in_4bit: bool) -> str:
    path = Path(record["model_path"])
    arguments = [f'pretrained={record["model_id"]}', "trust_remote_code=True"]
    if record["phase"] != "baseline":
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(f"LoRA adapter not found: {path}")
        arguments.append(f"peft={path}")
    if load_in_4bit:
        arguments.append("load_in_4bit=True")
    return ",".join(arguments)


def run_one(
    executable: str,
    record: dict,
    output_root: Path,
    tasks: list[str],
    batch_size: str,
    limit: int | None,
    load_in_4bit: bool,
    allow_code_execution: bool,
    overwrite: bool,
) -> None:
    output_dir = output_root / record["run_label"]
    done_path = output_dir / ".complete"
    if done_path.exists() and not overwrite:
        print(f"[skip] {record['run_label']}")
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        executable,
        "--model",
        "hf",
        "--model_args",
        model_args(record, load_in_4bit),
        "--tasks",
        ",".join(tasks),
        "--device",
        os.environ.get("LM_EVAL_DEVICE", "cuda:0"),
        "--batch_size",
        batch_size,
        "--apply_chat_template",
        "--output_path",
        str(output_dir),
        "--log_samples",
    ]
    if limit is not None:
        command.extend(["--limit", str(limit)])
    if "mbpp" in tasks:
        if not allow_code_execution:
            raise ValueError(
                "MBPP executes generated Python. Pass --allow-code-execution "
                "only inside an isolated RunPod container."
            )
        command.append("--confirm_run_unsafe_code")

    metadata = {
        "record": record,
        "tasks": tasks,
        "batch_size": batch_size,
        "limit": limit,
        "load_in_4bit": load_in_4bit,
        "command": command,
    }
    (output_dir / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    subprocess.run(command, check=True)
    done_path.write_text("complete\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--model-alias")
    parser.add_argument("--lm-eval-executable", default="lm_eval")
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--allow-code-execution", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    payload = json.loads(args.manifest.read_text())
    records = payload["models"]
    if args.model_alias:
        records = [
            record
            for record in records
            if record["model_alias"] == args.model_alias
        ]
    if not records:
        raise ValueError("No models matched the held-out manifest filters")
    for record in records:
        run_one(
            executable=args.lm_eval_executable,
            record=record,
            output_root=args.output_root,
            tasks=args.tasks,
            batch_size=args.batch_size,
            limit=args.limit,
            load_in_4bit=not args.no_4bit,
            allow_code_execution=args.allow_code_execution,
            overwrite=args.overwrite,
        )


if __name__ == "__main__":
    main()
