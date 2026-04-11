"""GenerationStep — encapsulates one generation cycle of the GA.

Extracts the ``create_next_generation`` logic from :class:`GeneticAlgorithm`
into a standalone, dependency-injected class.  This makes generation logic
independently testable and reduces the god-class surface.
"""
from __future__ import annotations

import logging
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

from genetic_algorithm.engine.population import (
    Population,
    PopulationStats,
    calculate_strategy_distance,
)
from genetic_algorithm.genome.individual import Individual
from genetic_algorithm.engine.operators.selection import select_parents
from genetic_algorithm.engine.operators.crossover import (
    crossover,
    _enforce_min_entry_conditions,
    _fix_invalid_operators,
)
from genetic_algorithm.engine.operators.mutation import mutate
from genetic_algorithm.engine.nsga2 import (
    fast_non_dominated_sort,
    crowding_distance_assignment,
)


class GenerationStep:
    """Creates one generation of offspring from a population.

    Receives explicit parameters and subsystem references in its constructor
    (no back-reference to GeneticAlgorithm).  Per-generation dynamic state
    is passed to :meth:`execute`.
    """

    # -- construction --------------------------------------------------------

    def __init__(
        self,
        *,
        population_size: int,
        elite_size: int,
        mode: str,
        config: dict,
        logger: logging.Logger,
        strategy_generator: Any,
        fitness_evaluator: Any = None,
        parallel_enabled: bool = False,
        parallel_evaluator: Any = None,
        crossover_rate: float,
        crossover_method: str,
        tournament_size: int,
        selection_method: str,
        allow_self_crossover: bool = True,
        adaptive_tournament: bool = False,
        random_immigrants: int,
        diversity_threshold: float,
        map_elites: Any = None,
        aos: Any = None,
        immigrant_provider: Optional[Callable] = None,
        llm_enabled: bool = False,
        strategy_designer: Any = None,
        feature_tracker: Any = None,
    ):
        # Core parameters
        self.population_size = population_size
        self.elite_size = elite_size
        self.mode = mode
        self.config = config
        self.logger = logger

        # Subsystems
        self.strategy_generator = strategy_generator
        self.fitness_evaluator = fitness_evaluator
        self.parallel_enabled = parallel_enabled
        self.parallel_evaluator = parallel_evaluator

        # Operator parameters
        self.crossover_rate = crossover_rate
        self.crossover_method = crossover_method
        self.tournament_size = tournament_size
        self.selection_method = selection_method
        self.allow_self_crossover = allow_self_crossover
        self.adaptive_tournament = adaptive_tournament

        # Diversity parameters
        self.random_immigrants = random_immigrants
        self.diversity_threshold = diversity_threshold

        # Optional subsystems
        self.map_elites = map_elites
        self.aos = aos
        self.immigrant_provider = immigrant_provider
        self.llm_enabled = llm_enabled
        self.strategy_designer = strategy_designer
        self.feature_tracker = feature_tracker

    # -- public API ----------------------------------------------------------

    def execute(
        self,
        population: Population,
        *,
        current_generation: int,
        mutation_rate: float,
        external_immigrants: Optional[List[Individual]] = None,
        no_improvement_count: int = 0,
        best_fitness_ever: float = 0.0,
        generation_stats: Optional[List[PopulationStats]] = None,
    ) -> Tuple[Population, dict]:
        """Create the next generation from *population*.

        Args:
            population: Current evaluated population (sorted in-place by fitness).
            current_generation: Zero-based index of the current generation.
            mutation_rate: Current (possibly adaptive) mutation rate.
            external_immigrants: Pre-queued immigrant individuals.
            no_improvement_count: Consecutive generations without improvement.
            best_fitness_ever: Best raw fitness seen so far.
            generation_stats: History of :class:`PopulationStats` per generation.

        Returns:
            ``(next_gen, op_stats)`` where *next_gen* is the new
            :class:`Population` and *op_stats* is a dict of operator counts.
        """
        self.logger.info(f"[STEP] Creating generation {current_generation + 1}")
        next_gen_num = current_generation + 1

        population.sort_by_fitness(reverse=True)
        next_gen = Population(size=self.population_size, generation=next_gen_num)

        # Step 1: Elitism
        elites, ranked_by_raw = self._select_elites(population, next_gen, next_gen_num)

        # Step 1b: Parsimony pressure on elites
        if elites and self.mode != "nsga2":
            self._apply_parsimony(next_gen, next_gen_num)

        # Step 2: Inject immigrants
        immigrant_stats = self._inject_immigrants(
            population,
            next_gen,
            next_gen_num,
            external_immigrants=external_immigrants or [],
            no_improvement_count=no_improvement_count,
            best_fitness_ever=best_fitness_ever,
            generation_stats=generation_stats or [],
        )

        # Step 3: Create offspring
        offspring_stats = self._create_offspring(
            population, next_gen, next_gen_num, mutation_rate,
        )

        # Step 3b: Fill population if undersized
        if len(next_gen) < self.population_size:
            self._fill_random(next_gen, next_gen_num)

        # Step 4: LLM-guided mutation on top-K
        llm_mutation_count = 0
        if ranked_by_raw:
            llm_mutation_count = self._apply_llm_mutations(
                next_gen, ranked_by_raw, next_gen_num,
                mutation_rate=mutation_rate,
                no_improvement_count=no_improvement_count,
            )

        # Log generation summary
        self._log_summary(offspring_stats, immigrant_stats, llm_mutation_count)

        # NSGA-II environmental selection
        if self.mode == "nsga2":
            next_gen = self._nsga2_environmental_selection(population, next_gen)

        op_stats = {**offspring_stats, **immigrant_stats, "llm_mutations": llm_mutation_count}
        return next_gen, op_stats

    # -- Step 1: Elitism -----------------------------------------------------

    def _select_elites(
        self, population: Population, next_gen: Population, next_gen_num: int,
    ) -> Tuple[list, list]:
        """Diversity-aware elite selection.

        Returns ``(elite_copies, ranked_by_raw)`` where *ranked_by_raw* is
        the raw-fitness-sorted candidate list (used later for LLM mutation).
        """
        if self.mode == "nsga2":
            self.logger.debug(
                "[ELITISM] NSGA-II mode: skipping raw-fitness elitism "
                "(handled by environmental selection)"
            )
            return [], []

        self.logger.debug(
            f"[ELITISM] Preserving top {self.elite_size} individuals (by raw fitness)"
        )

        ranked_by_raw = sorted(
            [ind for ind in population.individuals if ind.raw_fitness is not None],
            key=lambda x: x.raw_fitness,
            reverse=True,
        )

        # Diversity-aware greedy selection
        diversity_threshold = self.config.get("elite_diversity_threshold", 0.15)
        elites: list = []
        for candidate in ranked_by_raw:
            if len(elites) >= self.elite_size:
                break
            too_close = any(
                calculate_strategy_distance(candidate, e) < diversity_threshold
                for e in elites
            )
            if not too_close:
                elites.append(candidate)
        # Fill remaining slots from top if not enough diverse candidates
        if len(elites) < self.elite_size:
            for candidate in ranked_by_raw:
                if len(elites) >= self.elite_size:
                    break
                if candidate not in elites:
                    elites.append(candidate)

        # Create copies and add to next_gen
        for individual in elites:
            gene_copy = individual.strategy_gene.copy()
            gene_copy.generation = next_gen_num
            # Preserve self-adaptive mutation rate through elite carry-over
            if getattr(individual.strategy_gene, "self_mutation_rate", None) is not None:
                gene_copy.self_mutation_rate = individual.strategy_gene.self_mutation_rate
                gene_copy.self_crossover_pref = individual.strategy_gene.self_crossover_pref
            elite_copy = Individual(strategy_gene=gene_copy)
            # Restore pre-holdout raw fitness to avoid compounding penalties
            pre_holdout = getattr(individual, "_pre_holdout_raw_fitness", None)
            elite_copy.raw_fitness = (
                pre_holdout if pre_holdout is not None else individual.raw_fitness
            )
            elite_copy.fitness = elite_copy.raw_fitness
            elite_copy.metrics = individual.metrics.copy() if individual.metrics else {}
            elite_copy.evaluated = True
            _enforce_min_entry_conditions(elite_copy.strategy_gene, self.config)
            next_gen.add_individual(elite_copy)

        self.logger.info(f"[ELITISM] Preserved {self.elite_size} elite individuals")
        return elites, ranked_by_raw

    # -- Step 1b: Parsimony --------------------------------------------------

    def _apply_parsimony(self, next_gen: Population, next_gen_num: int) -> None:
        """Apply parsimony pressure to simplify elite copies."""
        parsimony_config = self.config.get("parsimony", {})
        indicator_config = self.config.get("indicators", {})
        parsimony_config["min_entry_conditions"] = indicator_config.get(
            "min_entry_conditions", 2
        )
        if not parsimony_config.get("enabled", False):
            return

        elite_list = list(next_gen.individuals)

        if self.parallel_enabled:
            from genetic_algorithm.evaluation.parallel import parallel_parsimony

            parallel_cfg = self.config.get("parallel_evaluation", {})
            num_workers = parallel_cfg.get("num_workers") or (os.cpu_count() - 1)
            bt_timeout = parallel_cfg.get("backtest_timeout", 120)
            removed = parallel_parsimony(
                elite_list,
                parsimony_config,
                self.config,
                num_workers=num_workers,
                backtest_timeout=bt_timeout,
                evaluator=self.parallel_evaluator,
            )
        else:
            from genetic_algorithm.core.parsimony import apply_parsimony_to_elites

            def _eval_fn(gene):
                return self.fitness_evaluator.evaluate(gene)

            removed = apply_parsimony_to_elites(elite_list, _eval_fn, parsimony_config)

        if removed > 0:
            self.logger.info(f"[PARSIMONY] Removed {removed} component(s) from elites")

    # -- Step 2: Immigrants --------------------------------------------------

    def _inject_immigrants(
        self,
        population: Population,
        next_gen: Population,
        next_gen_num: int,
        *,
        external_immigrants: List[Individual],
        no_improvement_count: int,
        best_fitness_ever: float,
        generation_stats: List,
    ) -> dict:
        """Inject immigrants (external, MAP-Elites, LLM, random) into *next_gen*.

        Returns a stats dict with immigrant counts.
        """
        stats = population.get_stats()
        immigrant_count = self.random_immigrants

        # Double immigrant count if diversity is low
        if stats.genetic_diversity is not None and stats.genetic_diversity < self.diversity_threshold:
            immigrant_count = self.random_immigrants * 2
            self.logger.warning(
                f"[DIVERSITY] Low diversity ({stats.genetic_diversity:.4f}), "
                f"doubling immigrants to {immigrant_count}"
            )

        # Collect external immigrants (already queued + provider + MAP-Elites)
        all_external: List[Individual] = list(external_immigrants)

        # MAP-Elites diverse injection
        if self.map_elites is not None and getattr(self.map_elites, "enabled", False):
            if self.map_elites.filled_cells > 0:
                me_diverse = self.map_elites.sample_diverse(
                    n=self.map_elites.injection_count
                )
                for ind in me_diverse:
                    clone = Individual(strategy_gene=ind.strategy_gene.copy())
                    clone.evaluated = False
                    clone.fitness = None
                    clone.raw_fitness = None
                    clone.metrics = {"origin": "map_elites_injection"}
                    all_external.append(clone)
                if me_diverse:
                    self.logger.info(
                        f"[MAP-ELITES] Injected {len(me_diverse)} diverse "
                        f"strategies as immigrants"
                    )

        # External provider callback
        if self.immigrant_provider:
            try:
                provider_immigrants = self.immigrant_provider(next_gen_num)
                if provider_immigrants:
                    all_external.extend(provider_immigrants)
            except Exception as e:
                self.logger.warning(f"[IMMIGRANTS] External provider failed: {e}")

        # LLM immigrants
        llm_immigrant_count = 0
        if self.llm_enabled and self.strategy_designer and self.strategy_designer.enabled:
            llm_immigrant_count, llm_individuals = self._generate_llm_immigrants(
                population,
                next_gen,
                next_gen_num,
                immigrant_count=immigrant_count,
                best_fitness_ever=best_fitness_ever,
                generation_stats=generation_stats,
            )
            all_external.extend(llm_individuals)

        # Inject external immigrants first
        immigrants_before = len(next_gen)
        external_injected = 0
        for ext_ind in all_external:
            if len(next_gen) >= self.population_size or external_injected >= immigrant_count:
                break
            ext_ind.strategy_gene.generation = next_gen_num
            ext_ind.strategy_gene.individual_id = len(next_gen)
            ext_ind.evaluated = False
            ext_ind.fitness = None
            ext_ind.raw_fitness = None
            next_gen.add_individual(ext_ind)
            external_injected += 1

        # Fill remaining immigrant slots with random immigrants
        random_immigrant_slots = max(0, immigrant_count - external_injected)
        for _ in range(random_immigrant_slots):
            if len(next_gen) >= self.population_size:
                break
            immigrant_gene = self.strategy_generator.generate_random_strategy(
                generation=next_gen_num,
                individual_id=len(next_gen),
            )
            next_gen.add_individual(Individual(strategy_gene=immigrant_gene))

        actual_added = len(next_gen) - immigrants_before
        random_added = actual_added - external_injected

        # Build log message
        parts: list = []
        if llm_immigrant_count > 0:
            parts.append(f"{llm_immigrant_count} LLM")
        if external_injected - llm_immigrant_count > 0:
            parts.append(f"{external_injected - llm_immigrant_count} external")
        if random_added > 0:
            parts.append(f"{random_added} random")
        self.logger.info(
            f"[IMMIGRANTS] Added {actual_added} immigrants ({', '.join(parts)})"
        )

        return {
            "immigrants_total": actual_added,
            "immigrants_llm": llm_immigrant_count,
            "immigrants_external": external_injected - llm_immigrant_count,
            "immigrants_random": random_added,
        }

    def _generate_llm_immigrants(
        self,
        population: Population,
        next_gen: Population,
        next_gen_num: int,
        *,
        immigrant_count: int,
        best_fitness_ever: float,
        generation_stats: List,
    ) -> Tuple[int, List[Individual]]:
        """Generate LLM-designed immigrant strategies.

        Returns ``(count, list_of_individuals)``.
        """
        sd = self.strategy_designer
        llm_count = max(1, int(immigrant_count * sd.immigrant_ratio))

        # Top performers context
        top_inds = sorted(
            [ind for ind in population.individuals if ind.fitness is not None],
            key=lambda x: x.fitness,
            reverse=True,
        )[:5]
        top_summaries = sd.get_top_performer_summaries(top_inds)
        weaknesses = sd.get_population_weaknesses(top_inds)

        # Feedback context
        feedback = None
        try:
            feature_report = self.feature_tracker.get_report() if self.feature_tracker else {}

            plateau_gens = 0
            if len(generation_stats) >= 2:
                for i in range(len(generation_stats) - 1, 0, -1):
                    if abs(generation_stats[i].best_fitness - generation_stats[i - 1].best_fitness) < 0.001:
                        plateau_gens += 1
                    else:
                        break

            evolution_progress = {
                "generation": next_gen_num,
                "total_generations": self.config.get("genetic_algorithm", {}).get("generations", 0),
                "best_fitness": best_fitness_ever,
                "plateau_generations": plateau_gens,
                "diversity": (
                    generation_stats[-1].genetic_diversity if generation_stats else None
                ),
            }
            feedback = sd.build_feedback_context(
                feature_report=feature_report,
                evolution_progress=evolution_progress,
            )
            self.logger.debug(
                f"[LLM FEEDBACK] Built feedback context with "
                f"{len(feedback.get('llm_strategy_results', []))} historical results, "
                f"{len(feedback.get('feature_importance', []))} feature scores"
            )
        except Exception as e:
            self.logger.warning(f"[LLM FEEDBACK] Failed to build feedback context: {e}")

        # Generate immigrants (batch or sequential)
        if sd.batch_enabled and llm_count > 1:
            llm_genes = sd.generate_immigrants_batch(
                count=llm_count,
                generation=next_gen_num,
                start_id=len(next_gen),
                top_performers=top_summaries,
                weaknesses=weaknesses,
                feedback=feedback,
            )
        else:
            llm_genes = sd.generate_immigrants(
                count=llm_count,
                generation=next_gen_num,
                start_id=len(next_gen),
                top_performers=top_summaries,
                weaknesses=weaknesses,
                feedback=feedback,
            )

        individuals: List[Individual] = []
        for gene in llm_genes:
            if len(next_gen) + len(individuals) >= self.population_size:
                break
            ind = Individual(strategy_gene=gene)
            ind.metrics["origin"] = "llm_immigrant"
            if hasattr(sd, "_last_provider_used"):
                ind.metrics["llm_provider"] = sd._last_provider_used
            individuals.append(ind)

        return len(llm_genes), individuals

    # -- Step 3: Offspring ---------------------------------------------------

    def _create_offspring(
        self,
        population: Population,
        next_gen: Population,
        next_gen_num: int,
        mutation_rate: float,
    ) -> dict:
        """Fill remaining slots via selection → crossover → mutation.

        Returns a stats dict with operator counts.
        """
        crossover_count = 0
        mutation_count = 0
        crossover_failures = 0
        mutation_failures = 0
        offspring_added = 0

        remaining = self.population_size - len(next_gen)
        self.logger.debug(f"[OFFSPRING] Creating offspring to fill remaining {remaining} slots")

        # Adaptive tournament size
        stats = population.get_stats()
        effective_tournament_size = self.tournament_size
        if self.adaptive_tournament and stats.genetic_diversity is not None:
            if stats.genetic_diversity > 0.4:
                effective_tournament_size = min(
                    self.tournament_size + 2,
                    max(3, self.population_size // 2),
                )
            elif stats.genetic_diversity < self.diversity_threshold:
                effective_tournament_size = max(2, self.tournament_size - 1)
            if effective_tournament_size != self.tournament_size:
                self.logger.info(
                    f"[ADAPTIVE TOURNAMENT] diversity={stats.genetic_diversity:.3f} "
                    f"→ tournament_size {self.tournament_size} → {effective_tournament_size}"
                )

        max_attempts = self.population_size * 10
        attempts = 0

        while len(next_gen) < self.population_size:
            attempts += 1
            if attempts > max_attempts:
                self.logger.warning(
                    f"[OFFSPRING] Reached max attempts ({max_attempts}). "
                    f"Population has {len(next_gen)}/{self.population_size} individuals."
                )
                break

            parent1, parent2 = select_parents(
                population,
                num_parents=2,
                method=self.selection_method,
                tournament_size=effective_tournament_size,
                allow_duplicates=self.allow_self_crossover,
            )

            child1_id = len(next_gen)
            child2_id = len(next_gen) + 1

            # AOS-driven crossover method selection
            cx_method = (
                self.aos.select_crossover()
                if self.aos is not None and self.aos.enabled
                else self.crossover_method
            )

            # Crossover or copy
            try:
                if random.random() < self.crossover_rate:
                    child1, child2 = crossover(
                        parent1, parent2,
                        generation=next_gen_num,
                        ind_id=child1_id,
                        config=self.config,
                        method=cx_method,
                    )
                    crossover_count += 1
                    parent_fit = max(parent1.fitness or 0, parent2.fitness or 0)
                    for c in (child1, child2):
                        c.metrics["_aos_cx"] = cx_method
                        c.metrics["_aos_parent_fit"] = parent_fit
                else:
                    child1 = self._clone_child(parent1.strategy_gene, child1_id, next_gen_num)
                    child2 = self._clone_child(parent2.strategy_gene, child2_id, next_gen_num)
            except (ValueError, KeyError, AttributeError, TypeError) as e:
                self.logger.debug(f"[CROSSOVER] Failed: {e}")
                crossover_failures += 1
                try:
                    child1 = self._clone_child(parent1.strategy_gene, child1_id, next_gen_num)
                    child2 = self._clone_child(parent2.strategy_gene, child2_id, next_gen_num)
                except (ValueError, KeyError, AttributeError, TypeError) as e2:
                    self.logger.debug(f"[CROSSOVER] Fallback clone also failed: {e2}")
                    continue

            # Mutation
            for child in [child1, child2]:
                if len(next_gen) >= self.population_size:
                    break
                try:
                    child = mutate(child, mutation_rate, self.config)
                    mutation_count += 1
                    _fix_invalid_operators(child.strategy_gene)
                    _enforce_min_entry_conditions(child.strategy_gene, self.config)
                    if "origin" not in child.metrics:
                        child.metrics["origin"] = "ga_offspring"
                    next_gen.add_individual(child)
                    offspring_added += 1
                except (ValueError, KeyError, AttributeError, TypeError) as e:
                    self.logger.debug(f"[MUTATION] Failed: {e}")
                    mutation_failures += 1
                    try:
                        clone = self._clone_child(child.strategy_gene, len(next_gen), next_gen_num)
                        _fix_invalid_operators(clone.strategy_gene)
                        clone.metrics["origin"] = "ga_offspring_unmutated"
                        next_gen.add_individual(clone)
                        offspring_added += 1
                    except Exception:
                        pass
                    continue

        self.logger.debug(
            f"[OFFSPRING] Added {offspring_added} offspring "
            f"(crossovers: {crossover_count}, mutations: {mutation_count})"
        )

        # Warn when failure rates exceed 10%
        total_cx = crossover_count + crossover_failures
        total_mut = mutation_count + mutation_failures
        if total_cx > 0 and crossover_failures / total_cx > 0.10:
            self.logger.warning(
                f"[OFFSPRING] High crossover failure rate: "
                f"{crossover_failures}/{total_cx} ({crossover_failures/total_cx:.0%})"
            )
        if total_mut > 0 and mutation_failures / total_mut > 0.10:
            self.logger.warning(
                f"[OFFSPRING] High mutation failure rate: "
                f"{mutation_failures}/{total_mut} ({mutation_failures/total_mut:.0%})"
            )

        return {
            "offspring_added": offspring_added,
            "crossover_count": crossover_count,
            "mutation_count": mutation_count,
            "crossover_failures": crossover_failures,
            "mutation_failures": mutation_failures,
        }

    # -- Step 3b: Random fill ------------------------------------------------

    def _fill_random(self, next_gen: Population, next_gen_num: int) -> None:
        """Guarantee population size by filling with random individuals."""
        fill_needed = self.population_size - len(next_gen)
        self.logger.warning(
            f"[OFFSPRING] Population undersized ({len(next_gen)}/{self.population_size}). "
            f"Filling {fill_needed} slots with random individuals."
        )
        for _ in range(fill_needed):
            try:
                filler_gene = self.strategy_generator.generate_random_strategy(
                    generation=next_gen_num,
                    individual_id=len(next_gen),
                )
                filler_ind = Individual(strategy_gene=filler_gene)
                filler_ind.metrics["origin"] = "random_fill"
                next_gen.add_individual(filler_ind)
            except Exception as e:
                self.logger.debug(f"[FILL] Random fill failed: {e}")

    # -- Step 4: LLM-guided mutation -----------------------------------------

    def _apply_llm_mutations(
        self,
        next_gen: Population,
        ranked_by_raw: list,
        next_gen_num: int,
        *,
        mutation_rate: float,
        no_improvement_count: int,
    ) -> int:
        """Apply targeted LLM mutations on top-K offspring when stagnating.

        Returns the number of LLM mutations applied.
        """
        sd = self.strategy_designer
        if not (
            self.llm_enabled
            and sd is not None
            and sd.enabled
            and getattr(sd, "mutation_enabled", False)
            and no_improvement_count >= getattr(sd, "mutation_stagnation_threshold", 999)
        ):
            return 0

        top_k = sd.mutation_top_k
        mutation_prob = sd.mutation_probability

        # Escalation during extended stagnation
        effective_prob = mutation_prob
        llm_cfg = self.config.get("advanced", {}).get("llm", {})
        escalation_threshold = llm_cfg.get("escalation_threshold", 5)
        if no_improvement_count >= escalation_threshold:
            escalation_prob = llm_cfg.get("escalation_mutation_probability", 0.25)
            effective_prob = max(mutation_prob, escalation_prob)

        mutation_candidates = ranked_by_raw[:top_k]
        llm_mutation_count = 0

        for elite in mutation_candidates:
            if len(next_gen) >= self.population_size:
                break
            if random.random() > effective_prob:
                continue

            try:
                metrics = getattr(elite, "metrics", {}) or {}
                gene = elite.strategy_gene
                metrics_with_complexity = dict(metrics)
                metrics_with_complexity["indicator_count"] = len(gene.indicators)
                metrics_with_complexity["condition_count"] = (
                    len(gene.entry_conditions) + len(gene.exit_conditions)
                )
                metrics_with_complexity["fitness"] = elite.raw_fitness or 0

                mutated_gene = sd.mutate_strategy(
                    parent_gene=gene,
                    metrics=metrics_with_complexity,
                    generation=next_gen_num,
                    individual_id=len(next_gen),
                )
                if mutated_gene:
                    mutated_ind = Individual(strategy_gene=mutated_gene)
                    mutated_ind.metrics["origin"] = "llm_mutation"
                    mutated_ind.metrics["parent_id"] = gene.individual_id
                    if hasattr(sd, "_last_provider_used"):
                        mutated_ind.metrics["llm_provider"] = sd._last_provider_used
                    next_gen.add_individual(mutated_ind)
                    llm_mutation_count += 1
            except Exception as e:
                self.logger.debug(f"[LLM MUTATION] Failed: {e}")
                try:
                    fallback_ind = mutate(elite, mutation_rate, self.config)
                    fallback_ind.strategy_gene.generation = next_gen_num
                    fallback_ind.strategy_gene.individual_id = len(next_gen)
                    fallback_ind.metrics = {"origin": "llm_fallback_mutation"}
                    next_gen.add_individual(fallback_ind)
                    self.logger.info("[LLM MUTATION] Fell back to GA mutation for elite")
                except Exception as e2:
                    self.logger.warning(f"[LLM MUTATION] GA fallback also failed: {e2}")

        if llm_mutation_count > 0:
            self.logger.info(
                f"[LLM MUTATION] Applied {llm_mutation_count} LLM-guided mutations "
                f"(stagnation: {no_improvement_count} gens)"
            )
        return llm_mutation_count

    # -- NSGA-II environmental selection -------------------------------------

    def _nsga2_environmental_selection(
        self, parents: Population, offspring: Population,
    ) -> Population:
        """NSGA-II (μ+λ) survivor selection.

        Merges parent and offspring populations, performs non-dominated sorting
        and crowding distance assignment, then selects the top μ individuals.
        """
        combined = [ind for ind in parents.individuals if ind.objectives is not None]
        combined += [ind for ind in offspring.individuals if ind.objectives is not None]
        unevaluated = [ind for ind in offspring.individuals if ind.objectives is None]

        if not combined:
            self.logger.warning(
                "[NSGA-II ENV] No evaluated individuals — returning offspring as-is"
            )
            return offspring

        fronts = fast_non_dominated_sort(combined)
        for front in fronts:
            crowding_distance_assignment(front)

        next_gen = Population(size=self.population_size, generation=offspring.generation)
        for front in fronts:
            if len(next_gen) + len(front) <= self.population_size:
                for ind in front:
                    ind.strategy_gene.generation = offspring.generation
                    next_gen.add_individual(ind)
            else:
                front_sorted = sorted(
                    front, key=lambda x: x.crowding_distance, reverse=True,
                )
                remaining = self.population_size - len(next_gen)
                for ind in front_sorted[:remaining]:
                    ind.strategy_gene.generation = offspring.generation
                    next_gen.add_individual(ind)
                break

        for ind in unevaluated:
            if len(next_gen) >= self.population_size:
                break
            next_gen.add_individual(ind)

        self.logger.info(
            f"[NSGA-II ENV] (μ+λ) selection: {len(combined)} combined → "
            f"{len(next_gen)} survivors across {len(fronts)} fronts"
        )
        return next_gen

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _clone_child(parent_gene, ind_id: int, generation: int) -> Individual:
        """Create an unevaluated child from a parent gene copy."""
        gene = parent_gene.copy()
        gene.generation = generation
        gene.individual_id = ind_id
        return Individual(strategy_gene=gene)

    def _log_summary(self, offspring_stats: dict, immigrant_stats: dict, llm_mutations: int) -> None:
        """Log generation creation summary."""
        parts = [
            f"crossovers: {offspring_stats['crossover_count']}",
            f"mutations: {offspring_stats['mutation_count']}",
        ]
        if llm_mutations > 0:
            parts.append(f"LLM mutations: {llm_mutations}")
        self.logger.info(
            f"[OFFSPRING] Added {offspring_stats['offspring_added']} offspring "
            f"({', '.join(parts)})"
        )
        cx_fail = offspring_stats["crossover_failures"]
        mut_fail = offspring_stats["mutation_failures"]
        if cx_fail > 0 or mut_fail > 0:
            self.logger.warning(f"[FAILURES] Crossover: {cx_fail}, Mutation: {mut_fail}")
