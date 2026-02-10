from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import matplotlib.pyplot as plt


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def load_results(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_plots(results: Dict[str, Any], out_dir: Path) -> List[str]:
    """Create a few simple plots and return list of filenames."""
    _ensure_dir(out_dir)
    files = []

    # per-agent bar charts: mean tasks, mean reward, mean identity
    agents = list(results["episodes"].keys())
    mean_tasks = [np.mean([e["completed_tasks"] for e in results["episodes"][a]]) for a in agents]
    mean_reward = [np.mean([e["total_reward"] for e in results["episodes"][a]]) for a in agents]
    mean_id = [np.mean([e["identity_final"] for e in results["episodes"][a]]) for a in agents]

    # Tasks bar
    plt.figure()
    plt.bar(agents, mean_tasks)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean completed tasks")
    plt.title("Tasks per episode (mean)")
    fp = out_dir / "tasks_mean.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=180)
    plt.close()
    files.append(fp.name)

    # Reward bar
    plt.figure()
    plt.bar(agents, mean_reward)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean total reward")
    plt.title("Reward per episode (mean)")
    fp = out_dir / "reward_mean.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=180)
    plt.close()
    files.append(fp.name)

    # Identity bar
    plt.figure()
    plt.bar(agents, mean_id)
    plt.xticks(rotation=20, ha="right")
    plt.ylim(0, 1.0)
    plt.ylabel("Mean identity_final")
    plt.title("Identity score (mean)")
    fp = out_dir / "identity_mean.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=180)
    plt.close()
    files.append(fp.name)

    # Identity trajectories (first episode of each agent)
    plt.figure()
    for a in agents:
        trace = results["episodes"][a][0]["identity_trace"]
        plt.plot(trace, label=a)
    plt.ylim(0, 1.0)
    plt.xlabel("t")
    plt.ylabel("Identity (overall)")
    plt.title("Identity trajectories (episode 0)")
    plt.legend()
    fp = out_dir / "identity_traces_ep0.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=180)
    plt.close()
    files.append(fp.name)

    # Collapse counts stacked bar
    collapse_keys = sorted(list(results["summary"][agents[0]]["collapse_counts_mean"].keys()))
    totals = {a: [results["summary"][a]["collapse_counts_mean"].get(k, 0.0) for k in collapse_keys] for a in agents}

    plt.figure()
    bottom = np.zeros(len(agents), dtype=np.float64)
    for i, k in enumerate(collapse_keys):
        vals = [totals[a][i] for a in agents]
        plt.bar(agents, vals, bottom=bottom, label=k)
        bottom += np.array(vals, dtype=np.float64)
    plt.xticks(rotation=20, ha="right")
    plt.ylabel("Mean collapse count")
    plt.title("Collapse counts (mean per episode)")
    plt.legend()
    fp = out_dir / "collapse_counts_mean.png"
    plt.tight_layout()
    plt.savefig(fp, dpi=180)
    plt.close()
    files.append(fp.name)

    return files
