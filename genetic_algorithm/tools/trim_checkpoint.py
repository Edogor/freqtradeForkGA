#!/usr/bin/env python3
"""
Trim an island-model checkpoint to keep only the top-N individuals per island.

Creates a new checkpoint file compatible with GenericIslandModelEvolution's
load_island_checkpoint() so a run can resume from the trimmed population.

Usage:
    # Trim wave28 gen11 checkpoint to top 25 per island, save into v2 checkpoint dir:
    python genetic_algorithm/tools/trim_checkpoint.py \
        --input genetic_algorithm/data/checkpoints_wave28_A1/island_checkpoint_gen11_20260401_182711.json \
        --output-dir genetic_algorithm/data/checkpoints_wave28_A1v2 \
        --top 25

    # Dry-run to see what would be extracted:
    python genetic_algorithm/tools/trim_checkpoint.py \
        --input genetic_algorithm/data/checkpoints_wave28_A1/island_checkpoint_gen11_20260401_182711.json \
        --top 25 --dry-run
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path


def extract_fitness(individual: dict) -> float:
    """Extract fitness value from an individual dict, handling multiple formats."""
    # Direct fitness field (shared fitness or raw)
    fitness = individual.get('fitness')
    if fitness is not None:
        return float(fitness)

    # raw_fitness field
    raw = individual.get('raw_fitness')
    if raw is not None:
        return float(raw)

    # Metrics-based fallback
    metrics = individual.get('metrics', {})
    if metrics:
        return float(metrics.get('fitness', metrics.get('total_profit', -999)))

    return -999.0


def trim_checkpoint(input_path: str, output_dir: str, top_n: int, dry_run: bool = False):
    """Load checkpoint, keep top-N per island, save trimmed version.

    The checkpoint typically contains elite carry-overs (with fitness) plus
    unevaluated offspring from create_next_generation(). We prioritize keeping
    all evaluated individuals first, then fill remaining slots with unevaluated
    offspring (which carry genetically evolved material from prior generations).
    """

    with open(input_path, 'r') as f:
        checkpoint = json.load(f)

    # Remove checksum (will be recomputed)
    checkpoint.pop('checksum', None)

    source_gen = checkpoint['generation']
    num_islands = checkpoint.get('num_islands', 0)
    orig_pop = checkpoint.get('population_per_island', 0)

    print(f"Source checkpoint: generation {source_gen}, {num_islands} islands × {orig_pop} pop")
    print(f"Trimming to top {top_n} per island")
    print()

    island_pops = checkpoint.get('island_populations', {})
    total_kept = 0
    total_dropped = 0
    total_evaluated_kept = 0
    total_sanitized = 0

    for island_name, pop_data in island_pops.items():
        individuals = pop_data.get('individuals', [])

        # Sanitize: remove individuals that would fail deserialization
        # (e.g., empty entry_conditions which StrategyGene.__post_init__ rejects)
        valid = []
        for ind in individuals:
            gene = ind.get('strategy_gene', {})
            if not gene.get('entry_conditions'):
                total_sanitized += 1
                continue
            valid.append(ind)
        individuals = valid
        orig_count = len(individuals)

        # Split into evaluated (have real fitness) and unevaluated
        evaluated = [i for i in individuals if i.get('fitness') is not None]
        unevaluated = [i for i in individuals if i.get('fitness') is None]

        # Sort evaluated by fitness descending
        evaluated_sorted = sorted(evaluated, key=extract_fitness, reverse=True)

        # Keep all evaluated first, then fill with unevaluated offspring
        kept = evaluated_sorted[:top_n]
        remaining_slots = top_n - len(kept)
        if remaining_slots > 0:
            kept.extend(unevaluated[:remaining_slots])

        n_eval_kept = min(len(evaluated_sorted), top_n)
        n_uneval_kept = len(kept) - n_eval_kept
        dropped = orig_count - len(kept)

        best_f = extract_fitness(evaluated_sorted[0]) if evaluated_sorted else 0

        status = f"  {island_name:30s}: {orig_count:3d} → {len(kept):3d}"
        status += f"  ({n_eval_kept} elite + {n_uneval_kept} offspring"
        if evaluated_sorted:
            status += f", best={best_f:.4f}"
        status += ")"
        print(status)

        total_kept += len(kept)
        total_dropped += dropped
        total_evaluated_kept += n_eval_kept

        # Update in place
        pop_data['individuals'] = kept
        pop_data['size'] = len(kept)

    print()
    if total_sanitized:
        print(f"Sanitized: removed {total_sanitized} invalid individuals (empty entry conditions)")
    print(f"Total: kept {total_kept} ({total_evaluated_kept} elite + {total_kept - total_evaluated_kept} offspring), dropped {total_dropped}")
    print(f"New total individuals: {total_kept} ({num_islands} islands × ~{top_n})")

    # Update metadata
    checkpoint['population_per_island'] = top_n
    checkpoint['timestamp'] = datetime.now().isoformat()
    checkpoint['trimmed_from'] = {
        'source_file': str(input_path),
        'original_pop_per_island': orig_pop,
        'top_n': top_n,
        'trimmed_at': datetime.now().isoformat(),
    }

    if dry_run:
        print("\n[DRY RUN] No files written.")
        return

    # Write output
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"island_checkpoint_gen{source_gen}_{timestamp}.json"
    filepath = output_path / filename

    # Compute checksum
    json_bytes = json.dumps(checkpoint, indent=2, default=str).encode('utf-8')
    checkpoint['checksum'] = hashlib.sha256(json_bytes).hexdigest()

    with open(filepath, 'w') as f:
        json.dump(checkpoint, f, indent=2, default=str)

    size_mb = filepath.stat().st_size / (1024 * 1024)
    print(f"\nSaved trimmed checkpoint: {filepath} ({size_mb:.1f} MB)")
    print(f"Resume will start from generation {source_gen + 1}")


def main():
    parser = argparse.ArgumentParser(
        description="Trim island-model checkpoint to top-N individuals per island",
    )
    parser.add_argument(
        '--input', required=True,
        help='Path to source island checkpoint JSON',
    )
    parser.add_argument(
        '--output-dir', default=None,
        help='Directory for output checkpoint (default: same as input)',
    )
    parser.add_argument(
        '--top', type=int, default=25,
        help='Number of top individuals to keep per island (default: 25)',
    )
    parser.add_argument(
        '--dry-run', action='store_true',
        help='Show what would be extracted without writing files',
    )
    args = parser.parse_args()

    if not Path(args.input).exists():
        print(f"ERROR: Input file not found: {args.input}", file=sys.stderr)
        sys.exit(1)

    output_dir = args.output_dir or str(Path(args.input).parent)

    trim_checkpoint(args.input, output_dir, args.top, args.dry_run)


if __name__ == '__main__':
    main()
