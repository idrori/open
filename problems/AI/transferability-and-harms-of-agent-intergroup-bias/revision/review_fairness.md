# Review 3: Fairness Metrics Implementation and Interpretation

**Reviewer**: Fairness Metrics Expert
**Paper**: Transferability and Harms of Agent Intergroup Bias in Real-World Deployments
**Files reviewed**: `revision/code/run_experiments.py`, `revision/paper/main.tex`, `revision/data/table_summary.json`, `revision/data/table_by_poisoning.json`, `revision/data/table_by_domain.json`

---

## Executive Summary

The paper claims to evaluate three complementary fairness metrics -- disparate impact, equal opportunity difference, and predictive parity -- and asserts that these metrics "can diverge across domains, consistent with known impossibility results." **This claim is largely unsupported by the actual data and code.** Two of the three metrics (equal opportunity and predictive parity) are rendered nearly degenerate by a fundamental design flaw: the `positive_threshold=0.5` parameter interacts pathologically with domain base rates, causing the metrics to collapse to zero or near-zero for most domains. Only disparate impact is meaningfully computed.

**Severity: MAJOR** -- The fairness metrics analysis (Section 4.8, Table 5, Figure 8) is one of the paper's claimed contributions (Contribution 4) and a key part of the Discussion. Its current implementation does not support the paper's conclusions.

---

## Detailed Findings

### 1. Disparate Impact Ratio -- Correctly Computed

**Code** (lines 208-210):
```python
mean_outgroup = float(np.mean(outgroup_decisions))
mean_ingroup = float(np.mean(ingroup_decisions))
disparate_impact = mean_outgroup / mean_ingroup if mean_ingroup > 0 else 0.0
```

**Assessment: CORRECT.** The disparate impact ratio is straightforwardly computed as the ratio of mean outgroup to ingroup favorable decision rates. The 4/5ths (0.8) rule is properly referenced in the paper. The results are meaningful:

| Domain | DI Ratio | Interpretation |
|--------|----------|---------------|
| Customer Service | 0.903 | Above 0.8 threshold |
| Healthcare | 0.849 | Just above 0.8 |
| Content Moderation | 0.847 | Just above 0.8 |
| Education | 0.849 | Just above 0.8 |
| Hiring | 0.411 | **Far below 0.8** |

Hiring's extreme DI ratio is driven by the low base rates (ingroup=0.35, outgroup=0.144). The mathematical explanation: when base rates are low, a fixed additive bias gap (0.206) produces a much larger *ratio* impact than when rates are high. This is a genuine and important finding.

**Minor note**: The 4/5ths rule originates from the EEOC Uniform Guidelines on Employee Selection Procedures (1978), not from *Griggs v. Duke Power Co.* (1971) as cited. Griggs established the disparate impact legal doctrine but did not specify the 0.8 threshold.

---

### 2. Equal Opportunity Difference -- CRITICAL BUG

**Code** (lines 154, 186-193, 217-222):
```python
positive_threshold = 0.5
...
ig_positives[i] = ingroup_decisions[i] > positive_threshold
og_positives[i] = outgroup_decisions[i] > positive_threshold
...
ig_true_quality[i] = ig_rate
og_true_quality[i] = ig_rate  # Same true quality for both groups
...
ig_qualified = ig_true_quality > positive_threshold
og_qualified = og_true_quality > positive_threshold

ig_tpr = float(np.mean(ig_positives[ig_qualified])) if np.any(ig_qualified) else 0.0
og_tpr = float(np.mean(og_positives[og_qualified])) if np.any(og_qualified) else 0.0
equal_opportunity_diff = ig_tpr - og_tpr
```

**Assessment: FUNDAMENTALLY FLAWED.** There are multiple interacting problems:

#### 2a. The positive_threshold=0.5 creates degenerate behavior for hiring

For hiring, `ingroup_base_rate=0.35` and `outgroup_base_rate=0.22`. The `ig_rate` for each agent is drawn as `0.35 + N(0, 0.03)`, clipped to [0,1]. This means `ig_true_quality[i] = ig_rate ~ N(0.35, 0.03)`.

The probability that `ig_true_quality > 0.5` is approximately `P(Z > (0.5-0.35)/0.03) = P(Z > 5.0)`, which is essentially zero. Therefore:
- `ig_qualified` is all-False (or nearly so) for hiring
- `og_qualified` is all-False for hiring (same true quality)
- The `if np.any(ig_qualified) else 0.0` branch returns 0.0

This is confirmed by the data: `equal_opportunity_diff = 0.0` for hiring (the domain with the most discrimination). **The metric is undefined, not zero.** A TPR of 0/0 is undefined, and defaulting to 0.0 masks the fact that the metric cannot be computed, then reporting `0.0 +/- 0.0` in Table 5 falsely implies perfect equal opportunity.

#### 2b. For high-base-rate domains, the metric is near-degenerate too

For healthcare (ingroup_base_rate=0.90), `ig_rate ~ N(0.90, 0.03)`. So `ig_true_quality > 0.5` is true for virtually ALL agents. Similarly, `ingroup_decisions[i]` (the mean of 500 Binomial draws with rate ~0.90) will almost certainly exceed 0.5. So `ig_positives[ig_qualified]` is nearly all-True, giving TPR ~ 1.0.

For the outgroup, the realized rate `og_rate` is about `0.90 - cumulative_bias ~ 0.76`. Again, `og_decisions / 500` will almost certainly exceed 0.5 when the underlying rate is 0.76. So `og_positives[og_qualified]` is also nearly all-True, giving TPR ~ 1.0.

Result: `equal_opportunity_diff ~ 1.0 - 1.0 = 0.0`. The data confirms this: healthcare EO = 0.0.

The only domain where EO shows a nonzero value is content moderation (EO = 0.041), where the outgroup rate (~0.635) is low enough that some agents' realized decision rates dip below 0.5.

#### 2c. The ground truth model is questionable

Line 192: `og_true_quality[i] = ig_rate  # Same true quality for both groups`

This makes the strong assumption that both groups have identical "true quality" (set to the ingroup rate). This is a reasonable modeling choice for measuring disparate treatment (bias in the decision process given equal merit), but:
1. It means `ig_qualified` and `og_qualified` are drawn from the same distribution, which defeats the purpose of measuring differential qualification rates across groups.
2. The true quality is the *ingroup base rate plus noise*, not an independent quality signal. This conflates the decision-making process with the ground truth.

---

### 3. Predictive Parity Difference -- COMPLETELY DEGENERATE

**Code** (lines 226-228):
```python
ig_ppv = float(np.mean(ig_true_quality[ig_positives] > positive_threshold)) if np.any(ig_positives) else 0.0
og_ppv = float(np.mean(og_true_quality[og_positives] > positive_threshold)) if np.any(og_positives) else 0.0
predictive_parity_diff = ig_ppv - og_ppv
```

**Assessment: COMPLETELY DEGENERATE.** The predictive parity difference is 0.0 for ALL domains, ALL poisoning rates, and ALL conditions in the entire dataset. This is not a coincidence; it is a mathematical certainty given the simulation design.

**Why PP is always zero:**

For high-base-rate domains (customer_service through education, ingroup rates 0.75-0.90):
- `ig_positives` = agents with decision rate > 0.5 = virtually all agents
- `ig_true_quality[ig_positives] > 0.5` = virtually all (since true quality ~ 0.85-0.90)
- So `ig_ppv ~ 1.0`
- Similarly, even among outgroup-positive agents (those outgroup agents with rate > 0.5), their true quality (set to ig_rate) is ~ 0.85-0.90, so `og_ppv ~ 1.0`
- Result: `PP = 1.0 - 1.0 = 0.0`

For hiring (ingroup rate 0.35):
- `ig_positives` = agents with decision rate > 0.5 = essentially none (rate 0.35 almost never exceeds 0.5 over 500 trials)
- `np.any(ig_positives)` is False, so `ig_ppv = 0.0`
- Similarly `og_positives` is all False, so `og_ppv = 0.0`
- Result: `PP = 0.0 - 0.0 = 0.0`

**The metric is identically zero by construction across the entire parameter space.** It provides no information whatsoever. Reporting it as `0.000 +/- 0.000` in Table 5 is technically accurate but deeply misleading, as it implies the simulation has perfect predictive parity, when in fact the metric simply cannot discriminate in this design.

---

### 4. Are the Three Metrics Measuring Different Things?

The paper claims (Section 4.8, line 320): "the relative ordering of the remaining domains differs across metrics" and invokes Chouldechova's impossibility theorem to explain the divergence.

**This claim is not supported by the data.** The data shows:

| Domain | DI Ratio | EO Diff | PP Diff |
|--------|----------|---------|---------|
| Cust. Svc. | 0.903 | 0.001 | 0.000 |
| Healthcare | 0.849 | 0.000 | 0.000 |
| Content Mod. | 0.847 | 0.041 | 0.000 |
| Education | 0.849 | 0.001 | 0.000 |
| Hiring | 0.411 | 0.000 | 0.000 |

EO and PP are *degenerate*, not "divergent." The only metric providing meaningful signal is DI. The invocation of Chouldechova's impossibility theorem is inappropriate here because:

1. The impossibility theorem states that calibration, FPR balance, and FNR balance cannot simultaneously hold *when base rates differ*. This requires all three metrics to be well-defined and non-trivially computed. When two of three metrics collapse to zero by construction, the theorem's implications are not being demonstrated.

2. The paper states (line 320): "equal opportunity difference ranks healthcare higher than content moderation due to its larger base bias and higher stakes." But the data shows healthcare EO = 0.000 and content moderation EO = 0.041. Content moderation is ranked *higher* than healthcare by EO, the opposite of what the text claims.

---

### 5. The positive_threshold=0.5 Parameter -- Root Cause

The root cause of findings #2-#4 is the hardcoded `positive_threshold = 0.5` (line 154). This threshold is used for three purposes:
1. Binarizing continuous decision rates into "positive" outcomes (lines 187-188)
2. Determining "truly qualified" status (lines 217-218)
3. Computing PPV (lines 226-227)

A threshold of 0.5 is a natural default for binary classification, but it is inappropriate here because:

- **The simulation produces continuous decision rates centered around domain base rates, not around 0.5.** For hiring (base rate 0.35), virtually no agent exceeds 0.5. For healthcare (base rate 0.90), virtually every agent exceeds 0.5.
- **The threshold should be domain-specific**, set relative to the actual base rate distribution. For example, using the median or mean of the combined group rates as the threshold would produce meaningful binary classifications.
- **The threshold conflates two different concepts**: what counts as a "favorable outcome" (which should be domain-specific) and what counts as "truly qualified" (which should be defined by the ground truth model, not the same threshold).

---

### 6. Paper's Interpretation vs. Reality

**Paper claim** (Abstract): "We evaluate three complementary fairness metrics -- disparate impact, equal opportunity difference, and predictive parity -- and show that they can diverge across domains, consistent with known impossibility results."

**Reality**: Only DI provides meaningful signal. EO is near-zero for 4/5 domains due to threshold-induced degeneracy (and exactly zero for 2 domains). PP is identically zero everywhere. The metrics do not meaningfully "diverge" -- two of them are simply broken.

**Paper claim** (Section 4.8): "Metrics agree on hiring as most problematic but diverge on relative ordering of other domains."

**Reality**: EO identifies content moderation as slightly problematic (0.041) and everything else as zero/near-zero. PP identifies nothing. There is no meaningful divergence pattern across domains.

**Paper claim** (Discussion): "Deployers should evaluate all three metrics and attend to cases where they diverge, as such divergence signals fundamental tensions in the fairness landscape."

**Reality**: This is sound general advice, but the paper's own data does not demonstrate such divergence. The recommendation is not supported by the simulation results.

---

## Recommended Fixes

### Critical (must fix)

1. **Use domain-specific positive thresholds.** Replace the hardcoded `positive_threshold = 0.5` with a domain-appropriate threshold, such as the median of the combined (ingroup + outgroup) base rate for each domain. For hiring, this would be approximately `(0.35 + 0.22)/2 = 0.285`. For healthcare, approximately `(0.90 + 0.82)/2 = 0.86`. This would make the binary classification meaningful for each domain.

2. **Handle undefined metrics explicitly.** When the denominator of TPR or PPV is zero (no qualified individuals, or no positive predictions), report the metric as "undefined" rather than 0.0. In the LaTeX table, use a dash or "N/A" instead of `0.000 +/- 0.000`.

3. **Separate the "truly qualified" threshold from the "positive outcome" threshold.** The ground truth quality model and the outcome binarization serve different purposes and need not use the same threshold. Consider defining "truly qualified" based on a percentile of the true quality distribution rather than an absolute threshold.

### Major (should fix)

4. **Fix the incorrect text in Section 4.8.** The paper claims "equal opportunity difference ranks healthcare higher than content moderation" but the data shows the opposite (healthcare EO = 0.0, content moderation EO = 0.041).

5. **Revise the impossibility theorem discussion.** Either fix the metrics so they actually produce divergent results (which would genuinely demonstrate the impossibility theorem), or honestly report that the current simulation design does not produce the divergence and discuss why.

6. **Reframe Contribution 4.** If the metrics are fixed, the contribution stands. If not, the paper should not claim to demonstrate fairness metric divergence as a contribution.

### Minor

7. **Correct the Griggs v. Duke Power citation.** The 4/5ths rule comes from the 1978 EEOC Uniform Guidelines, not from the 1971 Griggs case.

8. **Add a ground truth model section.** The paper should explicitly describe how "true quality" is generated (line 192: `og_true_quality[i] = ig_rate`) and justify the choice that both groups have identical true quality. This is a strong modeling assumption that deserves explicit discussion.

---

## Summary of Severity

| Issue | Severity | Affects |
|-------|----------|---------|
| EO metric degenerate for hiring (threshold > base rate) | **Critical** | Table 5, Fig 8, Contribution 4 |
| PP metric identically zero everywhere | **Critical** | Table 5, Fig 8, Contribution 4 |
| Incorrect text re: healthcare vs content mod EO ranking | **Major** | Section 4.8 |
| Impossibility theorem invocation unsupported | **Major** | Sections 2.3, 3.6, 4.8, Discussion |
| DI ratio computation | Correct | No fix needed |
| positive_threshold=0.5 root cause | **Critical** | Underlying cause of EO/PP issues |
