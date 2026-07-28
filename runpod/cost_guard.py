from __future__ import annotations

import argparse
import math


def maximum_seconds(budget_usd: float, price_per_gpu_hour: float, gpu_count: int) -> int:
    if budget_usd <= 0 or price_per_gpu_hour <= 0 or gpu_count <= 0:
        raise ValueError("Budget, price, and GPU count must all be positive")
    hours = budget_usd / (price_per_gpu_hour * gpu_count)
    return max(1, math.floor(hours * 3600))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--budget-usd", type=float, required=True)
    parser.add_argument("--price-per-gpu-hour", type=float, required=True)
    parser.add_argument("--gpu-count", type=int, default=1)
    parser.add_argument("--planned-hours", type=float)
    parser.add_argument("--print-max-seconds", action="store_true")
    args = parser.parse_args()

    seconds = maximum_seconds(
        args.budget_usd,
        args.price_per_gpu_hour,
        args.gpu_count,
    )
    if args.print_max_seconds:
        print(seconds)
        return
    maximum_hours = seconds / 3600
    print(f"hard budget window: {maximum_hours:.2f} hours")
    if args.planned_hours is not None:
        estimate = args.planned_hours * args.price_per_gpu_hour * args.gpu_count
        print(f"planned estimate: ${estimate:.2f}")
        if estimate > args.budget_usd:
            raise SystemExit(
                f"Planned cost ${estimate:.2f} exceeds budget ${args.budget_usd:.2f}"
            )


if __name__ == "__main__":
    main()
