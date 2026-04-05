"""
SIS A/B Testing Framework

Provides structured A/B experiments for individual SIS hooks.
Each experiment runs a control (hook disabled) and treatment (hook enabled)
across multiple seeds, logging per-generation metrics for later comparison.

Usage:
    from genetic_algorithm.intelligence.ab_framework import ABExperiment, ABResult

    exp = ABExperiment(
        name="seed_filtering_v1",
        hook="seed_filtering",
        base_config=config,
        seeds=[42, 123, 456],
        generations=20,
    )
    result = exp.compare_from_logs("output/ctrl", "output/exp")
    print(result.summary())
"""

import json
import logging
import copy
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ABResult:
    """Results of an A/B experiment comparing control vs treatment."""

    experiment_name: str
    hook: str
    seeds: List[int]
    generations: int

    # Per-seed final fitness (control, treatment)
    ctrl_final_fitness: List[float] = field(default_factory=list)
    treat_final_fitness: List[float] = field(default_factory=list)

    # Per-seed best fitness
    ctrl_best_fitness: List[float] = field(default_factory=list)
    treat_best_fitness: List[float] = field(default_factory=list)

    # Per-seed convergence generation (gen where best was found)
    ctrl_convergence_gen: List[int] = field(default_factory=list)
    treat_convergence_gen: List[int] = field(default_factory=list)

    # Per-seed diversity at end
    ctrl_diversity: List[float] = field(default_factory=list)
    treat_diversity: List[float] = field(default_factory=list)

    # Per-seed HoF count
    ctrl_hof_count: List[int] = field(default_factory=list)
    treat_hof_count: List[int] = field(default_factory=list)

    def fitness_delta(self) -> float:
        """Mean treatment best fitness minus mean control best fitness."""
        if not self.ctrl_best_fitness or not self.treat_best_fitness:
            return 0.0
        return float(np.mean(self.treat_best_fitness) - np.mean(self.ctrl_best_fitness))

    def convergence_delta(self) -> float:
        """Mean treatment convergence gen minus control (negative = faster)."""
        if not self.ctrl_convergence_gen or not self.treat_convergence_gen:
            return 0.0
        return float(np.mean(self.treat_convergence_gen) - np.mean(self.ctrl_convergence_gen))

    def diversity_delta(self) -> float:
        """Mean treatment diversity minus control."""
        if not self.ctrl_diversity or not self.treat_diversity:
            return 0.0
        return float(np.mean(self.treat_diversity) - np.mean(self.ctrl_diversity))

    def is_significant(self, min_seeds: int = 3) -> bool:
        """Check if we have enough data for meaningful comparison."""
        return (
            len(self.ctrl_best_fitness) >= min_seeds
            and len(self.treat_best_fitness) >= min_seeds
        )

    def effect_size(self) -> float:
        """Cohen's d effect size for best fitness."""
        if not self.is_significant():
            return 0.0
        ctrl = np.array(self.ctrl_best_fitness)
        treat = np.array(self.treat_best_fitness)
        pooled_std = np.sqrt((ctrl.std() ** 2 + treat.std() ** 2) / 2)
        if pooled_std < 1e-9:
            return 0.0
        return float((treat.mean() - ctrl.mean()) / pooled_std)

    def summary(self) -> str:
        """Human-readable summary of results."""
        lines = [
            f"A/B Experiment: {self.experiment_name}",
            f"Hook: {self.hook}",
            f"Seeds: {len(self.seeds)}, Generations: {self.generations}",
            "",
        ]

        if not self.is_significant():
            lines.append("INSUFFICIENT DATA — need at least 3 seeds per arm")
            return "\n".join(lines)

        ctrl_mean = np.mean(self.ctrl_best_fitness)
        treat_mean = np.mean(self.treat_best_fitness)
        delta = self.fitness_delta()
        d = self.effect_size()

        lines.extend([
            f"Control  best fitness: {ctrl_mean:.4f} ± {np.std(self.ctrl_best_fitness):.4f}",
            f"Treatm.  best fitness: {treat_mean:.4f} ± {np.std(self.treat_best_fitness):.4f}",
            f"Fitness delta:         {delta:+.4f}",
            f"Effect size (Cohen d): {d:+.3f}",
            "",
        ])

        if self.ctrl_convergence_gen and self.treat_convergence_gen:
            lines.append(
                f"Convergence delta:     {self.convergence_delta():+.1f} gens "
                f"({'faster' if self.convergence_delta() < 0 else 'slower'})"
            )

        if self.ctrl_diversity and self.treat_diversity:
            lines.append(f"Diversity delta:       {self.diversity_delta():+.4f}")

        lines.append("")

        # Verdict
        if d > 0.5:
            verdict = "STRONG POSITIVE — hook significantly improves evolution"
        elif d > 0.2:
            verdict = "MODERATE POSITIVE — hook shows promising improvement"
        elif d > -0.2:
            verdict = "NEUTRAL — no meaningful effect detected"
        elif d > -0.5:
            verdict = "MODERATE NEGATIVE — hook may be hurting evolution"
        else:
            verdict = "STRONG NEGATIVE — hook significantly hurts evolution"

        lines.append(f"Verdict: {verdict}")

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dict for JSON storage."""
        return {
            "experiment_name": self.experiment_name,
            "hook": self.hook,
            "seeds": self.seeds,
            "generations": self.generations,
            "ctrl_best_fitness": self.ctrl_best_fitness,
            "treat_best_fitness": self.treat_best_fitness,
            "ctrl_final_fitness": self.ctrl_final_fitness,
            "treat_final_fitness": self.treat_final_fitness,
            "ctrl_convergence_gen": self.ctrl_convergence_gen,
            "treat_convergence_gen": self.treat_convergence_gen,
            "ctrl_diversity": self.ctrl_diversity,
            "treat_diversity": self.treat_diversity,
            "ctrl_hof_count": self.ctrl_hof_count,
            "treat_hof_count": self.treat_hof_count,
            "fitness_delta": self.fitness_delta(),
            "effect_size": self.effect_size(),
            "convergence_delta": self.convergence_delta(),
        }


class ABExperiment:
    """Configure and analyze A/B experiments for individual SIS hooks.

    Does NOT run the GA itself — generates config pairs and analyzes results
    from completed runs.
    """

    def __init__(
        self,
        name: str,
        hook: str,
        base_config: Dict[str, Any],
        seeds: Optional[List[int]] = None,
        generations: int = 20,
    ):
        self.name = name
        self.hook = hook
        self.base_config = base_config
        self.seeds = seeds or [42, 123, 456]
        self.generations = generations

        if hook not in (
            "seed_filtering", "immigrants", "indicator_weights",
            "operator_weights", "synergy_weights",
        ):
            raise ValueError(f"Unknown hook: {hook}")

    def generate_config_pair(self, seed: int) -> Tuple[Dict, Dict]:
        """Generate (control_config, treatment_config) for a given seed.

        Control: target hook disabled, all others match base.
        Treatment: target hook enabled, all others match base.
        """
        ctrl = copy.deepcopy(self.base_config)
        treat = copy.deepcopy(self.base_config)

        # Ensure SIS is enabled in both
        ctrl.setdefault('sis', {})['enabled'] = True
        treat.setdefault('sis', {})['enabled'] = True

        # Set generations
        ctrl.setdefault('genetic_algorithm', {})['generations'] = self.generations
        treat.setdefault('genetic_algorithm', {})['generations'] = self.generations

        # Set seed
        ctrl.setdefault('genetic_algorithm', {})['seed'] = seed
        treat.setdefault('genetic_algorithm', {})['seed'] = seed

        # Control: disable only the target hook
        ctrl_hooks = ctrl['sis'].setdefault('hooks', {})
        treat_hooks = treat['sis'].setdefault('hooks', {})

        # Start with all hooks disabled in both to isolate
        for h in ("seed_filtering", "immigrants", "indicator_weights",
                   "operator_weights", "synergy_weights"):
            ctrl_hooks[h] = False
            treat_hooks[h] = False

        # Enable the target hook in treatment only
        treat_hooks[self.hook] = True

        # Label configs
        ctrl['_experiment'] = {
            'name': self.name, 'arm': 'control',
            'hook': self.hook, 'seed': seed,
        }
        treat['_experiment'] = {
            'name': self.name, 'arm': 'treatment',
            'hook': self.hook, 'seed': seed,
        }

        return ctrl, treat

    def generate_all_configs(self) -> List[Tuple[int, Dict, Dict]]:
        """Generate (seed, ctrl, treat) triples for all seeds."""
        return [(s, *self.generate_config_pair(s)) for s in self.seeds]

    def save_configs(self, output_dir: Path) -> List[Tuple[Path, Path]]:
        """Write all config pairs to output_dir.

        Returns list of (ctrl_path, treat_path).
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        paths = []
        for seed, ctrl, treat in self.generate_all_configs():
            ctrl_path = output_dir / f"{self.name}_seed{seed}_ctrl.json"
            treat_path = output_dir / f"{self.name}_seed{seed}_treat.json"
            with open(ctrl_path, 'w') as f:
                json.dump(ctrl, f, indent=2, default=str)
            with open(treat_path, 'w') as f:
                json.dump(treat, f, indent=2, default=str)
            paths.append((ctrl_path, treat_path))
            logger.info(f"[AB] Saved configs: {ctrl_path.name} / {treat_path.name}")

        return paths

    def compare_from_logs(
        self,
        ctrl_dirs: List[Path],
        treat_dirs: List[Path],
    ) -> ABResult:
        """Analyze completed run directories and produce ABResult.

        Each directory should contain:
        - sis_log.jsonl (SIS generation log)
        - hall_of_fame/hall_of_fame.json (HoF entries)
        - Or generation CSV / stats files

        Args:
            ctrl_dirs: List of control run output directories (one per seed).
            treat_dirs: List of treatment run output directories (one per seed).
        """
        result = ABResult(
            experiment_name=self.name,
            hook=self.hook,
            seeds=self.seeds,
            generations=self.generations,
        )

        for ctrl_dir in ctrl_dirs:
            metrics = self._extract_run_metrics(Path(ctrl_dir))
            if metrics:
                result.ctrl_best_fitness.append(metrics['best_fitness'])
                result.ctrl_final_fitness.append(metrics['final_fitness'])
                result.ctrl_convergence_gen.append(metrics['convergence_gen'])
                result.ctrl_diversity.append(metrics.get('diversity', 0))
                result.ctrl_hof_count.append(metrics.get('hof_count', 0))

        for treat_dir in treat_dirs:
            metrics = self._extract_run_metrics(Path(treat_dir))
            if metrics:
                result.treat_best_fitness.append(metrics['best_fitness'])
                result.treat_final_fitness.append(metrics['final_fitness'])
                result.treat_convergence_gen.append(metrics['convergence_gen'])
                result.treat_diversity.append(metrics.get('diversity', 0))
                result.treat_hof_count.append(metrics.get('hof_count', 0))

        return result

    @staticmethod
    def _extract_run_metrics(run_dir: Path) -> Optional[Dict[str, Any]]:
        """Extract key metrics from a completed run directory."""

        # Try SIS JSONL log first
        sis_log = run_dir / "sis_log.jsonl"
        if sis_log.exists():
            return ABExperiment._parse_sis_log(sis_log)

        # Try generation CSV
        gen_csv = run_dir / "generation_stats.csv"
        if gen_csv.exists():
            return ABExperiment._parse_gen_csv(gen_csv)

        # Try generation snapshots
        snapshots = sorted(run_dir.glob("gen_*.json"))
        if snapshots:
            return ABExperiment._parse_gen_snapshots(snapshots)

        logger.warning(f"[AB] No usable data in {run_dir}")
        return None

    @staticmethod
    def _parse_sis_log(path: Path) -> Optional[Dict[str, Any]]:
        """Parse sis_log.jsonl for per-generation fitness."""
        entries = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entries.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue

        if not entries:
            return None

        best_fitness = max(e.get('best_fitness', 0) or 0 for e in entries)
        final = entries[-1]

        # Find convergence gen (when best was first achieved)
        convergence_gen = 0
        running_best = 0.0
        for e in entries:
            bf = e.get('best_fitness', 0) or 0
            if bf > running_best:
                running_best = bf
                convergence_gen = e.get('gen', 0)

        return {
            'best_fitness': best_fitness,
            'final_fitness': final.get('best_fitness', 0) or 0,
            'convergence_gen': convergence_gen,
            'diversity': 0,  # Not in SIS log
            'hof_count': 0,  # Would need HoF file
        }

    @staticmethod
    def _parse_gen_csv(path: Path) -> Optional[Dict[str, Any]]:
        """Parse generation_stats.csv."""
        try:
            import pandas as pd
            df = pd.read_csv(path)
            if df.empty:
                return None

            best_col = None
            for col in ('best_fitness', 'best_raw_fitness', 'best'):
                if col in df.columns:
                    best_col = col
                    break
            if best_col is None:
                return None

            best_fitness = float(df[best_col].max())
            final_fitness = float(df[best_col].iloc[-1])

            # Convergence: gen where running max was achieved
            running_max = df[best_col].cummax()
            convergence_gen = int(running_max.idxmax())

            diversity = 0.0
            if 'genetic_diversity' in df.columns:
                diversity = float(df['genetic_diversity'].iloc[-1])

            return {
                'best_fitness': best_fitness,
                'final_fitness': final_fitness,
                'convergence_gen': convergence_gen,
                'diversity': diversity,
                'hof_count': 0,
            }
        except Exception as e:
            logger.warning(f"[AB] Failed to parse {path}: {e}")
            return None

    @staticmethod
    def _parse_gen_snapshots(snapshots: List[Path]) -> Optional[Dict[str, Any]]:
        """Parse gen_NNNN.json snapshot files."""
        best_fitness = 0.0
        convergence_gen = 0
        final_fitness = 0.0
        final_diversity = 0.0

        for snap_path in snapshots:
            try:
                with open(snap_path) as f:
                    data = json.load(f)
                stats = data.get('stats', {})
                bf = stats.get('best_fitness', 0) or 0
                gen = data.get('generation', 0)

                if bf > best_fitness:
                    best_fitness = bf
                    convergence_gen = gen

                final_fitness = bf
                final_diversity = stats.get('genetic_diversity', 0) or 0
            except Exception:
                continue

        if best_fitness == 0:
            return None

        return {
            'best_fitness': best_fitness,
            'final_fitness': final_fitness,
            'convergence_gen': convergence_gen,
            'diversity': final_diversity,
            'hof_count': 0,
        }


def run_ab_experiment_cli():
    """CLI entry point for A/B experiments.

    Usage:
        python -m genetic_algorithm.intelligence.ab_framework \\
            --hook seed_filtering \\
            --config config.json \\
            --seeds 42,123,456 \\
            --generations 20 \\
            --output-dir ab_results/seed_filtering_v1
    """
    import argparse

    parser = argparse.ArgumentParser(description="SIS A/B Testing Framework")
    sub = parser.add_subparsers(dest="command")

    # Generate configs
    gen = sub.add_parser("generate", help="Generate control/treatment config pairs")
    gen.add_argument("--hook", required=True,
                     choices=["seed_filtering", "immigrants", "indicator_weights",
                              "operator_weights", "synergy_weights"])
    gen.add_argument("--config", required=True, help="Base config JSON path")
    gen.add_argument("--name", default=None, help="Experiment name")
    gen.add_argument("--seeds", default="42,123,456",
                     help="Comma-separated random seeds")
    gen.add_argument("--generations", type=int, default=20)
    gen.add_argument("--output-dir", required=True)

    # Analyze results
    analyze = sub.add_parser("analyze", help="Analyze completed A/B runs")
    analyze.add_argument("--hook", required=True)
    analyze.add_argument("--name", default=None)
    analyze.add_argument("--ctrl-dirs", nargs="+", required=True,
                         help="Control run output directories")
    analyze.add_argument("--treat-dirs", nargs="+", required=True,
                         help="Treatment run output directories")
    analyze.add_argument("--seeds", default="42,123,456")
    analyze.add_argument("--generations", type=int, default=20)
    analyze.add_argument("--output", default=None, help="Save JSON result to file")

    args = parser.parse_args()

    if args.command == "generate":
        with open(args.config) as f:
            base_config = json.load(f)
        name = args.name or f"ab_{args.hook}"
        seeds = [int(s) for s in args.seeds.split(",")]

        exp = ABExperiment(
            name=name,
            hook=args.hook,
            base_config=base_config,
            seeds=seeds,
            generations=args.generations,
        )
        paths = exp.save_configs(Path(args.output_dir))
        print(f"Generated {len(paths)} config pairs in {args.output_dir}/")
        for ctrl_p, treat_p in paths:
            print(f"  {ctrl_p.name}  |  {treat_p.name}")

    elif args.command == "analyze":
        name = args.name or f"ab_{args.hook}"
        seeds = [int(s) for s in args.seeds.split(",")]

        exp = ABExperiment(
            name=name,
            hook=args.hook,
            base_config={},
            seeds=seeds,
            generations=args.generations,
        )
        result = exp.compare_from_logs(
            [Path(d) for d in args.ctrl_dirs],
            [Path(d) for d in args.treat_dirs],
        )
        print(result.summary())

        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result.to_dict(), f, indent=2)
            print(f"\nSaved to {args.output}")

    else:
        parser.print_help()


if __name__ == "__main__":
    run_ab_experiment_cli()
