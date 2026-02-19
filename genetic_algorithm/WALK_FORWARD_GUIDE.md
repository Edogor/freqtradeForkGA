# Walk-Forward Optimization - Usage Guide

## Quick Start

Walk-forward optimization prevents overfitting by evaluating strategies on multiple time windows. Here's how to use it effectively:

### Basic Configuration

```yaml
walk_forward:
  enabled: true
  train_days: 60        # Training window size
  validation_days: 15   # Validation window size
  step_days: 30         # Slide forward by 30 days
  mode: 'rolling'       # Fixed-size windows
  aggregation: 'mean'   # Average across windows
  max_windows: 10       # Safety limit
```

### Important: Choose Parameters Wisely!

**The number of windows directly impacts evaluation time:**

- Each window requires a full backtest per strategy
- More windows = more robust but MUCH slower
- Example: 10 windows × 20 individuals × 10 generations = 2,000 backtests!

### Recommended Settings

#### For Testing (Fast)
```yaml
timerange: "20241120-20250218"  # 90 days
train_days: 60
validation_days: 15
step_days: 30
# Result: 1-2 windows (20-40 backtests per generation)
```

#### For Production (Robust)
```yaml
timerange: "20231120-20241120"  # 365 days
train_days: 60
validation_days: 15
step_days: 60
# Result: 4-5 windows (80-100 backtests per generation)
```

#### ⚠️ AVOID: Too Many Windows
```yaml
# DON'T DO THIS:
timerange: "20231120-20260218"  # 820 days
train_days: 40
validation_days: 10
step_days: 10
# Result: 77 windows! (1,540 backtests per generation - will take hours/days!)
```

## Performance Guidelines

| Windows | Pop Size | Generations | Total Backtests | Est. Time* |
|---------|----------|-------------|-----------------|------------|
| 1       | 20       | 10          | 200             | 10 min     |
| 3       | 20       | 10          | 600             | 30 min     |
| 5       | 20       | 10          | 1,000           | 50 min     |
| 10      | 20       | 10          | 2,000           | 100 min    |
| 20      | 20       | 10          | 4,000           | 200 min    |
| 40      | 20       | 10          | 8,000           | 400 min    |

*Assumes ~3 seconds per backtest. Actual time varies based on:
- Data size and timeframe
- Number of trading pairs
- Strategy complexity
- CPU performance

## Safety Features

### 1. max_windows Parameter

Prevents accidentally creating too many windows:

```yaml
walk_forward:
  max_windows: 10  # Hard limit on number of windows
```

Without this, a misconfigured timerange could create 50+ windows and hang for hours.

### 2. Automatic Warnings

The system warns you when creating many windows:

```
⚠️  This configuration will create ~41 windows!
⚠️  Each individual requires 41 backtests - this may be very slow!
⚠️  Consider: increasing step_days, reducing timerange, or setting max_windows
```

### 3. Progress Logging

Clear progress indicators show evaluation status:

```
[1/5] Window 1: Val=20241230-20250109
[2/5] Window 2: Val=20250129-20250208
...
```

## Calculating Expected Windows

Use this formula to estimate how many windows will be created:

```
windows = (total_days - train_days - validation_days) / step_days + 1
```

Example:
```
timerange: 120 days
train_days: 60
validation_days: 15
step_days: 30

windows = (120 - 60 - 15) / 30 + 1 = 45 / 30 + 1 = 2.5 → 2 windows
```

## Troubleshooting

### Problem: Evaluation is very slow

**Cause:** Too many windows

**Solution:** 
1. Increase `step_days` (e.g., 10 → 30)
2. Reduce `timerange` (e.g., 365 days → 120 days)
3. Set `max_windows` to limit (e.g., max_windows: 5)

### Problem: "Stopping: validation window would exceed timerange"

**Cause:** Not enough data for even one window

**Solution:**
1. Increase `timerange` length
2. Reduce `train_days` or `validation_days`
3. Check your data files exist for the timerange

### Problem: All windows failing

**Cause:** Insufficient trading activity or data issues

**Solution:**
1. Check data files are present and valid
2. Verify pairs are trading during timerange
3. Adjust `min_train_trades` to lower value

## Best Practices

1. **Start small:** Test with 1-2 windows first, then scale up
2. **Use max_windows:** Always set a safety limit (5-10 is reasonable)
3. **Monitor first run:** Watch the first individual to estimate total time
4. **Trade-off:** More windows = more robust, but diminishing returns after 5-10
5. **Consider caching:** Elite strategies benefit from caching across windows

## Example Configurations

See these files for complete examples:
- `genetic_algorithm/config/ga_config_walk_forward_example.yaml` - Recommended settings
- `genetic_algorithm/config/ga_config.yaml` - Default configuration

## Running Walk-Forward Evolution

```bash
# Use the example configuration
python genetic_algorithm/run_ga.py --config genetic_algorithm/config/ga_config_walk_forward_example.yaml

# Or create your own config and run
python genetic_algorithm/run_ga.py --config my_walk_forward_config.yaml
```

## Understanding the Output

```
Walk-Forward Optimization Enabled
  Number of windows: 3
  Average train days: 60.0
  Average validation days: 15.0
  First train start: 2024-11-20
  Last validate end: 2025-02-18
  Aggregation method: mean
```

This tells you:
- 3 windows will be used
- Each strategy evaluated 3 times
- Total timerange coverage
- How scores are combined (mean = average)

## Need Help?

- Check logs for warnings about window count
- Run with smaller parameters first
- Use `max_windows` to limit evaluation time
- Consider parallel evaluation for larger runs (future feature)
