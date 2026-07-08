import os
import json
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import time

import numpy as np
from itertools import product
from tqdm import tqdm

from pathlib import Path

def json_safe(obj):
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [json_safe(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def make_problem_dir(algorithm, eps, success_threshold, phi_min, phi_max):
    problem_dir = (
        Path("data")
        / algorithm
        / f"eps{eps:.0e}_success{int(success_threshold*100)}_phi{phi_min}-{phi_max}"
    )
    problem_dir.mkdir(parents=True, exist_ok=True)
    (problem_dir / "runs").mkdir(exist_ok=True)
    return problem_dir

def make_run_dir(problem_dir):
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = problem_dir / "runs" / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir

def is_better_config(new, old, success_threshold):
    if new["success_rate"] < success_threshold:
        return False

    if old is None:
        return True

    if old["success_rate"] < success_threshold:
        return True

    return new["mean_budget"] < old["mean_budget"]



def default_n_jobs(n_jobs=None):
    if n_jobs is not None:
        return n_jobs
    n = os.cpu_count() or 1
    return min(4, max(1, n // 2))


def _single_trial(find_phi_fn, phi_min, phi_max, eps, params, seed):
    rng = np.random.default_rng(seed)
    phi = rng.uniform(phi_min, phi_max)

    phi_hat, budget = find_phi_fn(
        rng,
        phi,
        phi_max,
        phi_min,
        **params,
    )

    err = abs(phi_hat - phi)
    return {
        "err": err,
        "budget": budget,
        "success": err <= eps,
    }

def _single_trial_safe(find_phi_fn, phi_min, phi_max, eps, params, seed):
    # try:
        return _single_trial(find_phi_fn, phi_min, phi_max, eps, params, seed)
    # except Exception as e:
    #     return {
    #         "err": np.nan,
    #         "budget": np.nan,
    #         "success": False,
    #         "_failed": True,
    #         "_error": str(e),
    #         "seed": seed,
    #     }



def evaluate_algorithm(
    find_phi_fn,
    *,
    params,
    R=1000,
    phi_min=None,
    phi_max=0.1,
    eps=None,
    rng=None,
    n_jobs=None,
    log_dir=None,
):
    t_start = time.perf_counter()
    
    if rng is None:
        rng = np.random.default_rng()

    if phi_min is None:
        phi_min = phi_max / 10

    n_jobs = default_n_jobs(n_jobs)
    
    print(f"Using {n_jobs} parallel workers")

    seeds = rng.integers(0, 2**32, size=R)

    results = []
    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        futures = [
            ex.submit(
                _single_trial,
                find_phi_fn,
                phi_min,
                phi_max,
                eps,
                params,
                int(seed),
            )
            for seed in seeds
        ]

        for f in tqdm(as_completed(futures), total=R, desc="Evaluating"):
            results.append(f.result())

    errs = np.array([r["err"] for r in results])
    budgets = np.array([r["budget"] for r in results])

    success_rate = np.mean(errs < eps) if eps is not None else None
    mean_budget = np.mean(budgets)

    runtime_sec = time.perf_counter() - t_start
    
    print(f"Total runtime          : {runtime_sec/60:.2f} minutes")
    print("Evaluation summary")
    print("-" * 60)
    if eps is not None:
        print(f"Success criterion      : |φ̂ − φ| ≤ {eps:.1e}")
        print(f"Success rate           : {success_rate:.3f} ({success_rate*100:.1f}%)")
    print(f"Mean absolute error    : {errs.mean():.3e}")
    print(f"Median absolute error  : {np.median(errs):.3e}")
    print(f"Mean budget            : {mean_budget:,.0f}")
    print(f"Median budget          : {np.median(budgets):,.0f}")
    print("-" * 60)

    if log_dir is not None:
        with open(os.path.join(log_dir, "evaluation_results.json"), "w") as f:
            json.dump(json_safe(results), f, indent=2)

    return results


def grid_search_algorithm(
    find_phi_fn,
    param_grid,
    *,
    R=300,
    eps=1e-3,
    phi_min=None,
    phi_max=0.1,
    success_threshold=0.95,
    lambda_budget=1e-7,
    rng=None,
    verbose=True,
    n_jobs=None,
    algorithm_name="algo"
):
    
    t_start = time.perf_counter()
    
    if rng is None:
        rng = np.random.default_rng()

    if phi_min is None:
        phi_min = phi_max / 10

    n_jobs = default_n_jobs(n_jobs)
    
    print(f"Using {n_jobs} parallel workers")

    # run_dir = make_run_dir(label=log_label)
    problem_dir = make_problem_dir(
        algorithm_name,
        eps,
        success_threshold,
        phi_min,
        phi_max,
    )

    run_dir = make_run_dir(problem_dir)


    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    grid_results = []
    pbar = tqdm(
        param_combinations,
        desc=f"Grid search | workers={n_jobs}"
    )

    with ProcessPoolExecutor(max_workers=n_jobs) as ex:
        for values in pbar:
            params = dict(zip(param_names, values))
            seeds = rng.integers(0, 2**32, size=R)

            futures = [
                ex.submit(
                    _single_trial_safe,
                    find_phi_fn,
                    phi_min,
                    phi_max,
                    eps,
                    params,
                    int(seed),
                )
                for seed in seeds
            ]

            trial_results = [f.result() for f in as_completed(futures)]

            errs = np.array([r["err"] for r in trial_results])
            budgets = np.array([r["budget"] for r in trial_results])
            successes = np.array([r["success"] for r in trial_results])

            result = {
                **params,
                "mean_budget": float(budgets.mean()),
                "err": float(errs.mean()),
                "success_rate": float(successes.mean()),
            }
            result["loss"] = result["err"] + lambda_budget * result["mean_budget"]

            grid_results.append(result)

            pbar.set_postfix(
                loss=f"{result['loss']:.2e}",
                success=f"{result['success_rate']:.2f}",
            )


    runtime_sec = time.perf_counter() - t_start

    # Save metadata
    metadata = {
        "R": R,
        "eps": eps,
        "phi_min": phi_min,
        "phi_max": phi_max,
        "success_threshold": success_threshold,
        "lambda_budget": lambda_budget,
        "n_jobs": n_jobs,
        "param_grid": param_grid,
        "runtime_sec": runtime_sec,
    }
    
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(json_safe(metadata), f, indent=2)

    # Save full results
    with open(os.path.join(run_dir, "grid_results.json"), "w") as f:
        json.dump(json_safe(grid_results), f, indent=2)


    candidates = [
        r for r in grid_results
        if r["success_rate"] >= success_threshold
    ]

    best_loss = min(grid_results, key=lambda r: r["loss"])

    if candidates:
        best_candidate = min(candidates, key=lambda r: r["mean_budget"])

    else:
        best_candidate = None

    with open(os.path.join(run_dir, "best_by_loss.json"), "w") as f:
        json.dump(json_safe(best_loss), f, indent=2)

    if best_candidate:
        with open(os.path.join(run_dir, "best_by_budget.json"), "w") as f:
            json.dump(json_safe(best_candidate), f, indent=2)
            
        # Update best overall config
        best_overall_path = os.path.join(problem_dir, "best_overall.json")
        if os.path.exists(best_overall_path):
            with open(best_overall_path, "r") as f:
                best_overall = json.load(f)
        else:
            best_overall = None

        if is_better_config(best_candidate, best_overall, success_threshold):
            best_overall = {
                **best_candidate,
                "_updated_at": datetime.now().isoformat(),
                "_source_run": os.path.basename(run_dir),
            }
            tmp = best_overall_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(json_safe(best_overall), f, indent=2)
            os.replace(tmp, best_overall_path)

            print("✓ Updated best_overall.json")

    if verbose:
        summary_lines = [
            "=== Grid search summary ===",
            f"Total runtime          : {runtime_sec/60:.2f} minutes",            
            f"Total configurations: {len(grid_results)}",
            f"Candidates ≥ {success_threshold:.2f}: {len(candidates)}",
            "",
            "Best by loss:",
            str(best_loss),
        ]

        if best_candidate:
            summary_lines += ["", "Best by budget (success-constrained):", str(best_candidate)]
        else:
            summary_lines += ["", "No configuration met success threshold."]

        print("\n".join(summary_lines))

        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))

    return grid_results


def grid_search_algorithm_single(
    find_phi_fn,
    param_grid,
    *,
    R=300,
    eps=1e-3,
    phi_min=None,
    phi_max=0.1,
    success_threshold=0.95,
    lambda_budget=1e-7,
    rng=None,
    verbose=True,
    algorithm_name="algo"
):
    t_start = time.perf_counter()

    if rng is None:
        rng = np.random.default_rng()

    if phi_min is None:
        phi_min = phi_max / 10

    problem_dir = make_problem_dir(
        algorithm_name,
        eps,
        success_threshold,
        phi_min,
        phi_max,
    )
    run_dir = make_run_dir(problem_dir)

    param_names = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))

    grid_results = []
    pbar = tqdm(param_combinations, desc="Grid search (single-threaded)")

    for values in pbar:
        params = dict(zip(param_names, values))
        seeds = rng.integers(0, 2**32, size=R)

        trial_results = [
            _single_trial_safe(find_phi_fn, phi_min, phi_max, eps, params, int(seed))
            for seed in seeds
        ]

        errs = np.array([r["err"] for r in trial_results])
        budgets = np.array([r["budget"] for r in trial_results])
        successes = np.array([r["success"] for r in trial_results])

        result = {
            **params,
            "mean_budget": float(budgets.mean()),
            "err": float(errs.mean()),
            "success_rate": float(successes.mean()),
        }
        result["loss"] = result["err"] + lambda_budget * result["mean_budget"]

        grid_results.append(result)
        pbar.set_postfix(loss=f"{result['loss']:.2e}", success=f"{result['success_rate']:.2f}")

    runtime_sec = time.perf_counter() - t_start

    metadata = {
        "R": R, "eps": eps, "phi_min": phi_min, "phi_max": phi_max,
        "success_threshold": success_threshold, "lambda_budget": lambda_budget,
        "n_jobs": 1, "param_grid": param_grid, "runtime_sec": runtime_sec,
    }
    with open(os.path.join(run_dir, "metadata.json"), "w") as f:
        json.dump(json_safe(metadata), f, indent=2)
    with open(os.path.join(run_dir, "grid_results.json"), "w") as f:
        json.dump(json_safe(grid_results), f, indent=2)

    candidates = [r for r in grid_results if r["success_rate"] >= success_threshold]
    best_loss = min(grid_results, key=lambda r: r["loss"])
    best_candidate = min(candidates, key=lambda r: r["mean_budget"]) if candidates else None

    with open(os.path.join(run_dir, "best_by_loss.json"), "w") as f:
        json.dump(json_safe(best_loss), f, indent=2)

    if best_candidate:
        with open(os.path.join(run_dir, "best_by_budget.json"), "w") as f:
            json.dump(json_safe(best_candidate), f, indent=2)

        best_overall_path = os.path.join(problem_dir, "best_overall.json")
        if os.path.exists(best_overall_path):
            with open(best_overall_path, "r") as f:
                best_overall = json.load(f)
        else:
            best_overall = None

        if is_better_config(best_candidate, best_overall, success_threshold):
            best_overall = {
                **best_candidate,
                "_updated_at": datetime.now().isoformat(),
                "_source_run": os.path.basename(run_dir),
            }
            tmp = best_overall_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(json_safe(best_overall), f, indent=2)
            os.replace(tmp, best_overall_path)
            print("✓ Updated best_overall.json")

    if verbose:
        summary_lines = [
            "=== Grid search summary ===",
            f"Total runtime          : {runtime_sec/60:.2f} minutes",
            f"Total configurations   : {len(grid_results)}",
            f"Candidates ≥ {success_threshold:.2f}: {len(candidates)}",
            "",
            "Best by loss:",
            str(best_loss),
        ]
        if best_candidate:
            summary_lines += ["", "Best by budget (success-constrained):", str(best_candidate)]
        else:
            summary_lines += ["", "No configuration met success threshold."]

        print("\n".join(summary_lines))
        with open(os.path.join(run_dir, "summary.txt"), "w") as f:
            f.write("\n".join(summary_lines))

    return grid_results



def evaluate_algorithm_single(
    find_phi_fn,
    *,
    params,
    R=1000,
    phi_min=None,
    phi_max=0.1,
    eps=None,
    rng=None,
    log_dir=None,
):
    t_start = time.perf_counter()

    if rng is None:
        rng = np.random.default_rng()

    if phi_min is None:
        phi_min = phi_max / 10

    seeds = rng.integers(0, 2**32, size=R)

    results = []
    for seed in tqdm(seeds, total=R, desc="Evaluating"):
        result = _single_trial(find_phi_fn, phi_min, phi_max, eps, params, int(seed))
        results.append(result)

    errs = np.array([r["err"] for r in results])
    budgets = np.array([r["budget"] for r in results])

    success_rate = np.mean(errs < eps) if eps is not None else None
    mean_budget = np.mean(budgets)

    runtime_sec = time.perf_counter() - t_start

    print(f"Total runtime          : {runtime_sec/60:.2f} minutes")
    print("Evaluation summary")
    print("-" * 60)
    if eps is not None:
        print(f"Success criterion      : |φ̂ − φ| ≤ {eps:.1e}")
        print(f"Success rate           : {success_rate:.3f} ({success_rate*100:.1f}%)")
    print(f"Mean absolute error    : {errs.mean():.3e}")
    print(f"Median absolute error  : {np.median(errs):.3e}")
    print(f"Mean budget            : {mean_budget:,.0f}")
    print(f"Median budget          : {np.median(budgets):,.0f}")
    print("-" * 60)

    if log_dir is not None:
        with open(os.path.join(log_dir, "evaluation_results.json"), "w") as f:
            json.dump(json_safe(results), f, indent=2)

    return results