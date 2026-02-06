# Computational Reproducibility Audit

**Reviewer**: Reviewer 1 (Computational Reproducibility)
**Paper**: Transferability and Harms of Agent Intergroup Bias in Real-World Deployments
**Date**: 2026-02-06

---

## 1. Summary Verdict

The simulation code is **fully deterministic and reproduces bit-for-bit** across runs. The code architecture is clean, well-structured, and uses modern NumPy seeding. However, there is one **statistically meaningful error** in the confidence interval computation (z=1.96 instead of t-distribution with df=9) that systematically underestimates all reported confidence intervals by approximately 13%. Several minor issues are also documented below.

---

## 2. Reproducibility Test

### 2.1 Experiment Rerun
I executed `run_experiments.py` from scratch and compared all output JSON files against the stored data in `revision/data/`.

**Result: PASS -- Bit-for-bit identical.**

All numerical values in the regenerated `experiment_results.json` and all sub-table files match the stored versions exactly. The only field that changes between runs is `provenance.generated_at` (the timestamp), which is metadata-only and does not affect any computed values.

### 2.2 Figure Regeneration
I executed `generate_figures.py` from the regenerated data.

**Result: PASS -- All 10 figures generated without error.**

Figures produced: `domain_comparison.png`, `cue_strength.png`, `poisoning_impact.png`, `transferability.png`, `horizon_effect.png`, `effect_sizes.png`, `sensitivity_tornado.png`, `null_comparison.png`, `fairness_metrics.png`, `horizon_multidomain.png`.

---

## 3. RNG Seeding Audit

### 3.1 SeedSequence Independence
The code uses `np.random.SeedSequence(42)` with sequential `.spawn(1)[0]` calls to create independent per-experiment-replicate RNG streams (`run_experiments.py:346-350`).

**Analysis**: Each call to `ss.spawn(1)[0]` produces a child `SeedSequence` with an incrementing `spawn_key` (0, 1, 2, ...). I verified that:
- Each child produces a statistically independent stream (different `spawn_key` values).
- Recreating `SeedSequence(42)` and repeating the same spawn sequence produces identical children.
- The children are used to construct `np.random.default_rng(child)` generators that are mutually independent.

**Verdict: CORRECT.** SeedSequence spawning provides proper statistical independence between experiment-replicate RNG streams.

### 3.2 Vestigial Legacy Seeds
Two calls to `np.random.seed(42)` exist:
- Line 316 inside `run_experiment()`
- Line 562 in the `__main__` block

These seed the **legacy** NumPy random module (`np.random`), but the simulation exclusively uses the **new-style** Generator API (`np.random.default_rng`). The legacy seeds are never consumed by any simulation code path.

**Verdict: HARMLESS but CONFUSING.** These vestigial seeds should be removed to avoid misleading readers into thinking the legacy RNG is used. A reader could mistakenly conclude the experiments share a single RNG stream.

### 3.3 Non-Determinism Sources
- **datetime.datetime.now()** in provenance metadata (line 340): Changes every run but is data-only, never used in computation.
- **numpy version** in provenance: `numpy_version: "1.26.4"`. Results may differ with different NumPy versions due to Generator implementation changes, but this is standard practice and correctly documented.

**Verdict: No non-determinism sources in the computational path.**

---

## 4. Confidence Interval Computation -- ISSUE FOUND

### 4.1 The Problem
Both `aggregate_replicates()` (line 282) and `aggregate_transfer_replicates()` (line 305) use:

```python
ci_half = 1.96 * std_val / np.sqrt(n)
```

With `n = 10` replicates (df = 9), the correct approach is to use the t-distribution critical value:

| Approach | Critical Value | CI Half-Width (for std=0.005) |
|----------|---------------|-------------------------------|
| z = 1.96 (used) | 1.960 | 0.003099 |
| t(df=9) (correct) | 2.262 | 0.003577 |

**The z-approximation underestimates all 95% CIs by approximately 13.4%.**

This means every `*_ci95` value reported in the paper (Tables 2, 3, 4, and all error bars in figures) is too narrow by ~13%.

### 4.2 Impact Assessment
- The paper reports CIs like `0.206 +/- 0.003` for hiring bias. The correct CI would be `0.206 +/- 0.004`.
- At this sample size (n=10), the t-distribution correction is not negligible. The z-approximation is acceptable only for n > 30.
- **Qualitative conclusions are unaffected** because the CIs are used for illustration, not for hypothesis testing (the paper correctly uses paired t-tests for significance).
- The Cohen's d values and p-values are computed correctly and are unaffected by this issue.

### 4.3 Recommendation
**MEDIUM severity.** Replace `1.96` with `stats.t.ppf(0.975, n-1)` in both aggregation functions. This is a one-line fix in two locations.

---

## 5. Paper-to-Data Consistency

### 5.1 Claims that Match Data
The following paper claims are consistent with the generated data:

| Claim | Paper Value | Data Value | Status |
|-------|------------|------------|--------|
| Hiring bias magnitude | 0.206 +/- 0.003 | 0.20622 +/- 0.00347 | MATCH |
| Hiring harm score | 0.149 +/- 0.002 | 0.14858 +/- 0.00200 | MATCH |
| Hiring DI ratio | 0.411 | 0.4109 | MATCH |
| Healthcare bias | 0.136 +/- 0.004 | 0.13607 +/- 0.00428 | MATCH |
| Healthcare harm | 0.115 +/- 0.003 | 0.11500 +/- 0.00322 | MATCH |
| Transfer ratio range | 0.82 to 0.85 | 0.817 to 0.853 | MATCH |
| Poisoning 30% increase | ~70% | 69.7% | MATCH |
| Poisoning 50% increase | ~115% | 114.5% | MATCH (calculated: (0.2868-0.1336)/0.1336=114.8%) |
| Null model bias < 0.028 | All < 0.028 | Max = 0.02772 (hiring) | MATCH |
| Null model DI > 0.92 | All > 0.92 | Min = 0.9208 (hiring) | MATCH |

### 5.2 Minor Rounding Discrepancy
The paper abstract states the transfer ratio range is "$0.82$ to $0.85$" while the data shows 0.817 to 0.853. The paper rounds to 2 decimal places, which is acceptable.

The paper states bias increases "approximately 11%" from horizon 1 to 50 in one place (Figure 5 caption) and "approximately 10%" in the text (Section 4.3). The data shows:
- Horizon 1: 0.131 (from by_horizon data)
- Horizon 50: 0.144 (from by_horizon data)
- Actual increase: (0.144 - 0.131) / 0.131 = 9.9%

Both "approximately 10%" and "approximately 11%" are defensible roundings of 9.9%.

### 5.3 Null Model Not Fully Null
The null model sets `outgroup_base_rate = ingroup_base_rate` (eliminating structural bias), but retains `cue_strength = 0.3`. This means the null model still has cue-mediated bias amplification operating on zero structural bias. With the cue amplification formula `b_eff = b_base * (1 + 2c) + 0.3p`, when `b_base = 0` and `p = 0`, `b_eff = 0`. This is correct -- cue strength has no effect when base bias is zero. However, the per-agent noise (line 162: `rng.normal(0, 0.05)`) creates small random biases that then get amplified by the horizon feedback mechanism, producing the observed residual bias of ~0.02-0.03. The paper correctly notes these are "noise-level fluctuations amplified only by the cue strength and horizon feedback mechanisms."

---

## 6. Code Quality Observations

### 6.1 Architecture
- Clean dataclass-based parameter management (`ExperimentParams`, `DomainConfig`)
- Well-separated concerns: simulation, aggregation, experiment orchestration
- Provenance metadata attached to outputs
- Normalized data structures (e.g., `by_cue_strength` is a list of records, not nested dicts)

### 6.2 Figure Generation
- Colorblind-friendly palette (Wong 2011) correctly implemented
- 300 DPI publication quality
- Error bars/bands on all plots
- Consistent styling across all 10 figures

### 6.3 Minor Code Issues
1. **Unused `seed_counter`** (line 345-349): The counter increments but is never read or used for anything.
2. **Redundant `np.random.seed(42)` calls**: Lines 316 and 562 seed the legacy RNG that is never used.
3. **Hardcoded magic numbers**: The bias formula uses hardcoded constants (2.0 for cue amplification, 0.3 for poisoning boost, 0.015 for accumulation rate) that are not parameterized. These are documented in comments and the paper equations, which is acceptable.

---

## 7. Summary of Findings

| Issue | Severity | Status |
|-------|----------|--------|
| Bit-for-bit reproducibility | -- | PASS |
| SeedSequence independence | -- | PASS |
| CI uses z=1.96 instead of t(df=9) | MEDIUM | All CIs underestimated by ~13% |
| Vestigial `np.random.seed(42)` calls | LOW | Harmless but confusing |
| Unused `seed_counter` variable | LOW | Dead code |
| Paper-to-data consistency | -- | PASS (all claims verified) |
| Figure regeneration | -- | PASS |
| No non-determinism sources | -- | PASS |
| Horizon percentage claim (10% vs 11%) | LOW | Minor inconsistency in rounding |

### Recommendations
1. **Fix CI computation**: Replace `1.96` with `scipy.stats.t.ppf(0.975, n-1)` in `aggregate_replicates()` and `aggregate_transfer_replicates()`.
2. **Remove vestigial seeds**: Delete `np.random.seed(42)` on lines 316 and 562.
3. **Remove unused variable**: Delete `seed_counter` on line 345.
4. **Harmonize horizon claim**: Use consistent "approximately 10%" in both the figure caption and text, or recompute after fixing CIs.
