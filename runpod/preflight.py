from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def gpu_info() -> dict:
    command = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(command, text=True).strip().splitlines()
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        return {"available": False, "error": str(error)}
    gpus = []
    for line in output:
        name, memory_mb, driver = [part.strip() for part in line.split(",", 2)]
        gpus.append(
            {
                "name": name,
                "memory_mb": int(memory_mb),
                "driver_version": driver,
            }
        )
    return {"available": bool(gpus), "gpus": gpus}


def import_check(name: str) -> dict:
    try:
        module = importlib.import_module(name)
    except Exception as error:
        return {"ok": False, "error": repr(error)}
    return {"ok": True, "version": getattr(module, "__version__", None)}


def required_model_ids(spec: dict) -> list[str]:
    model_ids = {model["model_id"] for model in spec["models"]}
    model_ids.update(
        judge["model_id"]
        for judge in spec.get("judges", [])
        if judge["provider"] == "local_hf"
    )
    classifier = (
        spec.get("held_out_benchmark_suite", {})
        .get("reproducibility", {})
        .get("harmbench_classifier")
    )
    if classifier:
        model_ids.add(classifier)
    return sorted(model_ids)


def check_huggingface_access(model_ids: list[str]) -> dict[str, dict]:
    try:
        from huggingface_hub import HfApi
    except Exception as error:
        return {
            model_id: {"ok": False, "error": f"huggingface_hub unavailable: {error!r}"}
            for model_id in model_ids
        }

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    results = {}
    for model_id in model_ids:
        try:
            info = api.model_info(model_id)
        except Exception as error:
            results[model_id] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
        else:
            results[model_id] = {
                "ok": True,
                "revision": info.sha,
            }
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--spec",
        type=Path,
        default=Path("experiment/followup_study/study_spec_runpod.json"),
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = args.root.resolve()
    spec_path = args.spec if args.spec.is_absolute() else root / args.spec
    spec = json.loads(spec_path.read_text())
    generated_root = root / spec["generated_root"]
    disk = shutil.disk_usage(root)
    gpu = gpu_info()
    minimum_vram = int(spec["runtime"]["minimum_vram_gb"]) * 1024
    minimum_disk = int(spec["runtime"]["minimum_disk_gb"]) * 1024**3
    model_access = check_huggingface_access(required_model_ids(spec))

    dataset_status = {}
    for dataset in spec["datasets"]:
        dataset_root = root / spec["experiment_root"] / "dataset" / dataset["name"]
        expected = [dataset["control_level"], *dataset["levels"]]
        dataset_status[dataset["name"]] = {
            split: (dataset_root / f"{split}.jsonl").exists()
            for split in expected
        }

    report = {
        "root": str(root),
        "python": sys.version,
        "gpu": gpu,
        "disk": {
            "free_gb": round(disk.free / 1024**3, 2),
            "minimum_gb": spec["runtime"]["minimum_disk_gb"],
            "ok": disk.free >= minimum_disk,
        },
        "packages": {
            name: import_check(name)
            for name in [
                "torch",
                "transformers",
                "datasets",
                "peft",
                "bitsandbytes",
                "unsloth",
            ]
        },
        "datasets": dataset_status,
        "credentials": {
            "HF_TOKEN": bool(os.environ.get("HF_TOKEN")),
            "OPENAI_API_KEY": bool(os.environ.get("OPENAI_API_KEY")),
        },
        "huggingface_model_access": model_access,
        "generated_assets": {
            "exists": generated_root.exists(),
            "runbooks": len(list((generated_root / "runbooks").glob("*.sh"))),
            "train_configs": len(list((generated_root / "train_configs").rglob("*.json"))),
            "eval_configs": len(list((generated_root / "eval_configs").rglob("*.json"))),
        },
    }
    report["gpu"]["meets_vram_minimum"] = bool(
        gpu.get("gpus")
        and max(item["memory_mb"] for item in gpu["gpus"]) >= minimum_vram
    )
    report["datasets_ok"] = all(
        all(splits.values()) for splits in dataset_status.values()
    )
    report["packages_ok"] = all(item["ok"] for item in report["packages"].values())
    report["model_access_ok"] = all(item["ok"] for item in model_access.values())
    report["ready_for_training"] = all(
        [
            report["gpu"]["available"],
            report["gpu"]["meets_vram_minimum"],
            report["disk"]["ok"],
            report["datasets_ok"],
            report["packages_ok"],
            report["model_access_ok"],
            report["generated_assets"]["runbooks"] > 0,
        ]
    )

    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    raise SystemExit(0 if report["ready_for_training"] else 1)


if __name__ == "__main__":
    main()
