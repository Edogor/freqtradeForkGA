"""
RunEngine: Thin orchestrator for the evolution loop lifecycle.

Handles:
- Signal management (graceful shutdown via SIGINT/SIGTERM)
- Loop iteration with timing / time-limit checks
- Web control events (stop, pause, strategy injection)
- Pre/post generation diagnostics and monitoring callbacks
- Checkpoint scheduling (interval-based)
- Resource checks (disk space)
- Post-evolution teardown and summary reporting

Domain logic (evaluation, selection, fitness, holdout, convergence)
stays in GeneticAlgorithm via the ``process_generation`` method.
"""

from __future__ import annotations

import logging
import signal
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from genetic_algorithm.engine.population import Population

logger = logging.getLogger(__name__)


@dataclass
class GenerationResult:
    """Result of one generation's domain processing."""

    population: Population
    stats: Any  # PopulationStats or dict
    should_break: bool = False
    break_reason: str = ""
    extras: dict = field(default_factory=dict)


class RunEngine:
    """Orchestrates the evolution loop lifecycle.

    ``RunEngine`` owns the *loop structure* while
    ``GeneticAlgorithm.process_generation`` owns the *per-generation
    domain logic* (evaluate → post-eval hooks → holdout → convergence →
    next generation).

    Usage::

        engine = RunEngine(ga)
        results = engine.run(resume_from=checkpoint_path)
    """

    def __init__(self, ga: Any) -> None:
        self.ga = ga
        self.logger = ga.logger

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self, resume_from: Optional[str] = None):
        """Run the full evolution with signal handling."""
        original_sigint = signal.getsignal(signal.SIGINT)
        original_sigterm = signal.getsignal(signal.SIGTERM)

        def _graceful_shutdown(signum, frame):
            if self.ga._shutdown_requested:
                self.logger.warning("[SHUTDOWN] Force quit requested")
                signal.signal(signal.SIGINT, original_sigint)
                raise KeyboardInterrupt
            self.ga.request_shutdown()

        signal.signal(signal.SIGINT, _graceful_shutdown)
        signal.signal(signal.SIGTERM, _graceful_shutdown)

        try:
            return self._run_loop(resume_from)
        finally:
            signal.signal(signal.SIGINT, original_sigint)
            signal.signal(signal.SIGTERM, original_sigterm)

    # ------------------------------------------------------------------
    # Loop
    # ------------------------------------------------------------------

    def _run_loop(self, resume_from: Optional[str] = None):
        """Inner loop: setup → generations → teardown."""
        population, start_gen, pareto_archive = self._setup(resume_from)

        evolution_start = time.time()
        if self.ga.max_runtime_minutes:
            self.logger.info(f"  Max runtime: {self.ga.max_runtime_minutes} minutes")

        for gen in range(start_gen, self.ga.generations):
            self.ga.current_generation = gen

            # ── Pre-generation orchestration ──
            if self._check_time_limit(evolution_start, gen, population):
                break
            if self._handle_web_events(gen, population):
                break

            self._log_generation_header(gen)
            self.ga.diagnostics.start_generation(gen)
            self.ga.monitor.on_generation_start(gen, self.ga.generations)

            # ── Domain: evaluate → post-eval → holdout → stats ──
            result = self.ga.process_generation(population, gen, pareto_archive)
            population = result.population

            # ── Post-generation orchestration ──
            self._end_generation(gen, result, population)

            should_stop = self._check_resources(gen, population)
            if result.should_break or should_stop:
                break

            # ── Domain: convergence → catastrophic restart → next gen ──
            population, adv_break = self.ga.advance_generation(
                population, gen, result.stats,
            )
            if adv_break:
                break

        return self._teardown(population, pareto_archive)

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _setup(self, resume_from):
        """Pre-loop initialisation: diagnostics, monitor, population, pareto."""
        # T1.2 — enforce the holdout lockbox *before* anything else reads
        # the training timerange.  This guarantees the GA can never train
        # on data reserved for ex-post hold-out validation.
        try:
            from genetic_algorithm.tools.holdout import enforce_lockbox
            lockbox = enforce_lockbox(self.ga.config)
            if lockbox.applied and lockbox.holdout_timerange:
                self.ga.config.setdefault("holdout", {})["_resolved_timerange"] = (
                    lockbox.holdout_timerange
                )
        except Exception as exc:  # pragma: no cover - defensive
            self.logger.warning(f"[HOLDOUT] Lockbox enforcement failed: {exc}")

        self.ga.diagnostics.start_run(self.ga.config)
        self.ga.monitor.start(self.ga.config)

        start_gen = 0

        if resume_from:
            self.logger.info("=" * 70)
            self.logger.info("RESUMING EVOLUTION FROM CHECKPOINT")
            self.logger.info("=" * 70)
            population, start_gen = self.ga.load_checkpoint(resume_from)
            self.logger.info(
                f"  Resuming at generation {start_gen + 1}/{self.ga.generations}"
            )
            self.logger.info(
                f"  Population: {len(population.individuals)} | "
                f"Best so far: {self.ga.best_fitness_ever:.4f}"
            )
            self.logger.info(
                f"  Mutation rate: {self.ga.mutation_rate:.2%} | "
                f"No-improvement: {self.ga.no_improvement_count}"
            )
            self.logger.info("=" * 70)
        else:
            self.logger.info("=" * 70)
            self.logger.info("GENETIC ALGORITHM STARTING")
            self.logger.info("=" * 70)
            self.logger.info(
                f"  Population: {self.ga.population_size} | "
                f"Generations: {self.ga.generations}"
            )
            self.logger.info(
                f"  Mutation: {self.ga.mutation_rate:.2%} | "
                f"Crossover: {self.ga.crossover_rate:.2%} "
                f"({self.ga.crossover_method})"
            )
            self.logger.info(
                f"  Selection: {self.ga.selection_method} | "
                f"Elite size: {self.ga.elite_size}"
            )
            self.logger.info("=" * 70)
            population = self.ga.initialize_population()

        # Incremental evolution: re-evaluate on updated data
        if self.ga._incremental.enabled and not resume_from:
            inc_pop = self.ga._incremental.resume(self.ga)
            if inc_pop is not None:
                population = inc_pop
                start_gen = self.ga.current_generation

        # Pareto archive (NSGA-II)
        pareto_archive = None
        archive_config = self.ga.config.get("pareto_archive", {})
        if self.ga.mode == "nsga2" and archive_config.get("enabled", False):
            from genetic_algorithm.core.pareto_archive import ParetoArchive

            pareto_archive = ParetoArchive(
                max_size=archive_config.get("max_size", 100),
                decay_rate=archive_config.get("decay_rate", 0.95),
            )
            self.logger.info(
                f"[ARCHIVE] Pareto archive enabled "
                f"(max_size={pareto_archive.max_size}, "
                f"decay={pareto_archive.decay_rate})"
            )

        return population, start_gen, pareto_archive

    # ------------------------------------------------------------------
    # Per-generation checks
    # ------------------------------------------------------------------

    def _check_time_limit(self, start_time: float, gen: int, population) -> bool:
        """Return True when the time budget is exhausted."""
        if not self.ga.max_runtime_minutes:
            return False
        elapsed_min = (time.time() - start_time) / 60.0
        if elapsed_min >= self.ga.max_runtime_minutes:
            self.logger.info(
                f"[TIME LIMIT] Reached {elapsed_min:.1f} min "
                f"(limit: {self.ga.max_runtime_minutes} min) — stopping"
            )
            self.ga.save_checkpoint(population, max(gen - 1, 0))
            return True
        return False

    def _handle_web_events(self, gen: int, population) -> bool:
        """Handle web-dashboard stop / pause / inject.  Return True to break."""
        # Stop
        if self.ga._web_stop_event and self.ga._web_stop_event.is_set():
            self.logger.info(
                "[WEB] Stop signal received — saving checkpoint and exiting"
            )
            self.ga.save_checkpoint(population, max(gen - 1, 0))
            return True

        # Pause
        if self.ga._web_pause_event and self.ga._web_pause_event.is_set():
            self.logger.info("[WEB] Paused — waiting for resume signal...")
            while self.ga._web_pause_event.is_set():
                if self.ga._web_stop_event and self.ga._web_stop_event.is_set():
                    break
                import time as _time

                _time.sleep(0.5)
            self.logger.info("[WEB] Resumed")
            if self.ga._web_stop_event and self.ga._web_stop_event.is_set():
                self.ga.save_checkpoint(population, max(gen - 1, 0))
                return True

        # Strategy injection
        if self.ga._web_injection_queue:
            self.ga._drain_injection_queue(population, gen)

        return False

    def _log_generation_header(self, gen: int) -> None:
        self.logger.info("")
        self.logger.info(f"{'─' * 70}")
        self.logger.info(f"GENERATION {gen + 1}/{self.ga.generations}")
        self.logger.info(f"{'─' * 70}")

    # ------------------------------------------------------------------
    # Post-generation orchestration
    # ------------------------------------------------------------------

    def _end_generation(self, gen: int, result: GenerationResult, population) -> None:
        """Diagnostics, monitor callbacks, tracker, checkpoint scheduling."""
        stats = result.stats
        extras = result.extras

        self.ga.diagnostics.end_generation(gen, stats, population, extras=extras)

        # Terminal / web monitor
        gen_timing = (
            self.ga.diagnostics.timing.history[-1]
            if self.ga.diagnostics.timing.history
            else None
        )
        if self.ga._web_monitor and hasattr(
            self.ga._web_monitor, "store_population_snapshot"
        ):
            try:
                pop_dicts = [ind.to_dict() for ind in population.individuals]
                self.ga._web_monitor.store_population_snapshot(pop_dicts)
            except Exception as e:
                self.logger.debug(f"Population snapshot failed: {e}")

        self.ga.monitor.on_generation_end(
            gen, stats, gen_timing, self.ga.best_individual, extras=extras,
        )

        # Experiment tracker
        try:
            self.ga._tracker.save_generation(gen, stats, population)
        except Exception as e:
            self.logger.debug(f"[TRACKER] Generation save failed: {e}")

        # WF cache stats
        try:
            if hasattr(self.ga.fitness_evaluator, "log_wf_cache_stats"):
                self.ga.fitness_evaluator.log_wf_cache_stats()
            elif hasattr(self.ga.fitness_evaluator, "base_evaluator"):
                base = self.ga.fitness_evaluator.base_evaluator
                if hasattr(base, "log_wf_cache_stats"):
                    base.log_wf_cache_stats()
        except Exception:
            pass

        # Checkpoint at configured intervals
        if (
            self.ga.checkpoint_interval > 0
            and (gen + 1) % self.ga.checkpoint_interval == 0
        ):
            self.ga.save_checkpoint(population, gen)

    def _check_resources(self, gen: int, population) -> bool:
        """Disk-space and shutdown checks.  Return True to break."""
        # Disk space
        try:
            import shutil as _shutil

            _disk = _shutil.disk_usage(str(self.ga.checkpoint_dir))
            _free_gb = _disk.free / (1024**3)
            _min_gb = self.ga.config.get("storage", {}).get(
                "min_disk_gb_runtime", 2.0,
            )
            if _free_gb < _min_gb:
                self.logger.warning(
                    f"[DISK] Low disk space: {_free_gb:.1f} GB free "
                    f"(threshold: {_min_gb:.1f} GB). "
                    f"Saving checkpoint and pausing evolution."
                )
                cp = self.ga.save_checkpoint(population, gen)
                self.logger.warning(
                    f"[DISK] Checkpoint saved: {cp}. "
                    f"Free up disk space and resume with: --resume {cp}"
                )
                return True
        except Exception:
            pass

        # Graceful shutdown
        if self.ga._shutdown_requested:
            self.logger.info("[SHUTDOWN] Saving checkpoint before shutdown...")
            cp = self.ga.save_checkpoint(population, gen)
            self.logger.info(f"[SHUTDOWN] Checkpoint saved: {cp}")
            self.logger.info(f"[SHUTDOWN] Resume with: --resume {cp}")
            return True

        return False

    # ------------------------------------------------------------------
    # Teardown
    # ------------------------------------------------------------------

    def _teardown(self, population, pareto_archive):
        """Post-evolution: reports, cleanup, return results."""
        ga = self.ga

        # Feature importance
        try:
            ga.feature_tracker.log_summary(top_n=10)
        except Exception as e:
            self.logger.warning(f"Final feature importance report failed: {e}")

        # Hall of Fame
        try:
            hof_summary = ga.hall_of_fame.get_summary()
            if hof_summary["size"] > 0:
                self.logger.info("")
                self.logger.info(
                    f"[HALL OF FAME] {hof_summary['size']} strategies archived"
                )
                self.logger.info(
                    f"  Best: {hof_summary['best_fitness']:.4f}  "
                    f"Avg: {hof_summary['avg_fitness']:.4f}"
                )
                for i, entry in enumerate(hof_summary["top_5"]):
                    self.logger.info(
                        f"  #{i + 1}: fitness={entry['fitness']:.4f}  "
                        f"profit={entry['profit']:.2f}%  "
                        f"sharpe={entry['sharpe']:.2f}"
                    )
        except Exception as e:
            self.logger.warning(f"Hall of fame summary failed: {e}")

        # LLM usage report
        if ga.llm_enabled and ga.strategy_designer.enabled:
            self._log_llm_report()

        # Final checkpoint
        ga.save_checkpoint(
            population,
            ga.current_generation,
            filepath=str(ga.checkpoint_dir / "checkpoint_final.json"),
        )

        # Final summary banner
        self._log_final_summary(population, pareto_archive)

        # Diagnostics
        timing_summary = ga.diagnostics.end_run(
            top_strategies=population.get_best(10) if population else None,
        )
        if timing_summary:
            self.logger.info(f"[TIMING] {timing_summary}")

        # Close visualisation
        if ga.visualizer:
            ga.visualizer.close()

        # Shutdown parallel evaluator
        if ga.parallel_evaluator:
            ga.parallel_evaluator.shutdown()

        # Terminal monitor
        ga.monitor.on_evolution_complete(
            {
                "generations": ga.current_generation + 1,
                "best_fitness": (
                    ga.best_individual.fitness if ga.best_individual else None
                ),
            }
        )

        # Tracker final
        try:
            ga._tracker.save_final_results(ga.best_individual, ga.generation_stats)
        except Exception as e:
            self.logger.debug(f"[TRACKER] Final results save failed: {e}")

        # Trade visualisation (final mode)
        if ga.trade_visualizer and ga.trade_vis_mode == "final":
            self.logger.info(
                "[TRADE VIS] Generating trade charts for top strategies..."
            )
            top = population.get_best(ga.trade_vis_top_n)
            for idx, ind in enumerate(top):
                ga._visualize_strategy_trades(ind, ga.current_generation, idx)
            self.logger.info(
                f"[TRADE VIS] Generated charts for {len(top)} strategies"
            )

        # Holdout test
        holdout_cfg = ga.config.get("holdout_test", {})
        if holdout_cfg.get("enabled", False):
            ga._run_holdout_test(population, holdout_cfg)

        # CPCV
        cpcv_cfg = ga.config.get("cpcv", {})
        if cpcv_cfg.get("enabled", False):
            ga._run_post_evolution_cpcv(population, cpcv_cfg)

        # Ensemble co-evolution
        ensemble_cfg = ga.config.get("ensemble", {})
        if ensemble_cfg.get("enabled", False):
            self._run_ensemble(population, ensemble_cfg)

        # Coevolution finishing phase
        if ga._coevolution.enabled:
            try:
                template = (
                    population.get_best(1)[0].strategy_gene
                    if population.individuals
                    else None
                )
                coevo_genes = ga._coevolution.run(template_gene=template)
                if coevo_genes:
                    self.logger.info(
                        f"[COEVOLUTION] Produced {len(coevo_genes)} modular strategies"
                    )
            except Exception as e:
                self.logger.warning(f"Coevolution finishing phase failed: {e}")

        # Lifecycle manager
        if ga._lifecycle.enabled:
            try:
                top = population.get_best(5) if population else []
                for ind in top:
                    ga._lifecycle.register_strategy(
                        strategy_id=ind.id,
                        gene_dict=ind.strategy_gene.to_dict(),
                        metrics=ind.metrics,
                    )
                ga._lifecycle.save_state()
                self.logger.info(
                    f"[LIFECYCLE] Registered {len(top)} strategies for monitoring"
                )
            except Exception as e:
                self.logger.warning(f"Lifecycle registration failed: {e}")

        # Return results
        if ga.mode == "nsga2":
            from genetic_algorithm.engine.nsga2 import (
                get_pareto_front,
                nsga2_crowded_comparison_sort,
            )

            if pareto_archive is not None and pareto_archive.size > 0:
                self.logger.info(
                    f"[ARCHIVE] Returning {pareto_archive.size} archive members"
                )
                return nsga2_crowded_comparison_sort(
                    pareto_archive.get_archive()
                )[: ga.pareto_front_size]
            pareto_front = get_pareto_front(list(population.individuals))
            return nsga2_crowded_comparison_sort(pareto_front)[
                : ga.pareto_front_size
            ]
        else:
            population.sort_by_fitness(reverse=True)
            return population.get_best(10)

    # ------------------------------------------------------------------
    # Teardown helpers
    # ------------------------------------------------------------------

    def _log_llm_report(self) -> None:
        ga = self.ga
        try:
            llm_stats = ga.strategy_designer.get_stats()
            self.logger.info("")
            self.logger.info("=" * 60)
            self.logger.info("LLM USAGE REPORT")
            self.logger.info("=" * 60)
            self.logger.info(f"  Total API calls:  {llm_stats['total_requests']}")
            self.logger.info(f"  Successful:       {llm_stats['successful']}")
            self.logger.info(f"  Failed:           {llm_stats['failed']}")
            self.logger.info(f"  Validation fixed: {llm_stats['validation_fixed']}")
            self.logger.info(
                f"  Budget used:      "
                f"{llm_stats.get('calls_this_run', 0)} / "
                f"{llm_stats.get('budget_remaining', 0) + llm_stats.get('calls_this_run', 0)}"
            )
            calls_by_type = llm_stats.get("calls_by_type", {})
            if any(v > 0 for v in calls_by_type.values()):
                self.logger.info("  Calls by type:")
                for ctype, count in calls_by_type.items():
                    if count > 0:
                        self.logger.info(f"    {ctype}: {count}")

            perf = llm_stats.get("llm_performance", {})
            gens_tracked = llm_stats.get("generations_tracked", 0)
            if gens_tracked > 0:
                avg_llm = perf.get("avg_llm_fitness", 0)
                avg_rand = perf.get("avg_random_fitness", 0)
                best_llm = perf.get("best_llm_fitness", 0)
                best_gen = perf.get("best_llm_generation", -1)
                advantage = avg_llm - avg_rand
                self.logger.info(
                    f"  [LLM vs RANDOM] Tracked over {gens_tracked} generations:"
                )
                self.logger.info(f"    Avg LLM fitness:    {avg_llm:.4f}")
                self.logger.info(f"    Avg Random fitness: {avg_rand:.4f}")
                self.logger.info(
                    f"    LLM advantage:      "
                    f"{'+' if advantage >= 0 else ''}{advantage:.4f}"
                )
                self.logger.info(
                    f"    Best LLM strategy:  {best_llm:.4f} (gen {best_gen})"
                )
                if advantage > 0.05:
                    self.logger.info(
                        "    --> LLM strategies are contributing meaningful value!"
                    )
                elif advantage < -0.05:
                    self.logger.info(
                        "    --> LLM strategies are underperforming. "
                        "Consider prompt tuning."
                    )
                else:
                    self.logger.info(
                        "    --> LLM strategies performing on par with random."
                    )
            self.logger.info("=" * 60)
        except Exception as e:
            self.logger.warning(f"LLM stats summary failed: {e}")

    def _log_final_summary(self, population, pareto_archive) -> None:
        ga = self.ga
        self.logger.info("")
        self.logger.info("=" * 70)
        if ga._shutdown_requested:
            self.logger.info("EVOLUTION PAUSED (graceful shutdown)")
        else:
            self.logger.info("EVOLUTION COMPLETE")
        self.logger.info("=" * 70)
        self.logger.info(f"  Total generations: {ga.current_generation + 1}")

        if ga.mode == "nsga2":
            from genetic_algorithm.engine.nsga2 import get_pareto_front

            pareto_front = get_pareto_front(list(population.individuals))
            self.logger.info(f"  Pareto front size: {len(pareto_front)}")
            self.logger.info("  Top Pareto-optimal strategies:")
            sorted_front = sorted(
                pareto_front,
                key=lambda x: x.objectives[0] if x.objectives else 0,
                reverse=True,
            )
            for i, ind in enumerate(sorted_front[:5]):
                if ind.objectives and ind.metrics:
                    m = ind.metrics
                    self.logger.info(
                        f"    {i + 1}. {ind.id}: "
                        f"profit={m.get('profit', 0):.2f}%, "
                        f"drawdown={m.get('max_drawdown', 0):.1%}, "
                        f"sharpe={m.get('sharpe_ratio', 0):.2f}"
                    )
        else:
            if ga.best_individual:
                self.logger.info(f"  Best individual: {ga.best_individual.id}")
                self.logger.info(
                    f"  Best fitness: {ga.best_individual.fitness:.4f}"
                )
                if ga.best_individual.metrics:
                    m = ga.best_individual.metrics
                    self.logger.info(
                        f"  Best profit: {m.get('profit', 0):.2f}% | "
                        f"Win rate: {m.get('win_rate', 0):.1%}"
                    )
        self.logger.info("=" * 70)

    def _run_ensemble(self, population, ensemble_cfg: dict) -> None:
        try:
            from genetic_algorithm.core.ensemble_evolution import EnsembleEvolver

            candidates = population.get_best(
                ensemble_cfg.get("candidate_pool_size", 20)
            )
            evolver = EnsembleEvolver(self.ga.config, candidates)
            best_portfolio = evolver.run()
            if best_portfolio:
                self.logger.info("")
                self.logger.info("=" * 60)
                self.logger.info("ENSEMBLE EVOLUTION RESULTS")
                self.logger.info("=" * 60)
                self.logger.info(
                    f"  Strategies: {len(best_portfolio.strategy_ids)}"
                )
                self.logger.info(
                    f"  Portfolio fitness: {best_portfolio.fitness:.4f}"
                )
                pm = best_portfolio.metrics
                self.logger.info(
                    f"  Profit: {pm.get('total_profit_pct', 0):.2f}% | "
                    f"Sharpe: {pm.get('sharpe_ratio', 0):.2f} | "
                    f"Drawdown: {pm.get('max_drawdown', 0):.1%}"
                )
                for sid, w in zip(
                    best_portfolio.strategy_ids, best_portfolio.weights
                ):
                    self.logger.info(f"    {sid}: weight={w:.2%}")
                self.logger.info("=" * 60)
        except Exception as e:
            self.logger.warning(f"Ensemble co-evolution failed: {e}")
