# Statistical Methodology Review

**Reviewer 2: Statistical Methodologist**
**Paper: Transferability and Harms of Agent Intergroup Bias in Real-World Deployments**

---

## 1. Confidence Interval Computation: z=1.96 vs. t-Distribution

**Finding: ISSUE -- CIs are ~13% too narrow.**

The code (`run_experiments.py`, line 282) computes 95% CIs using:

```python
ci_half = 1.96 * std_val / np.sqrt(n)
```

With `n=10` replicates, the correct multiplier is the t-distribution critical value `t(9, 0.975) = 2.262`, not `z = 1.96`. The z-approximation is only appropriate for large samples (typically n >= 30).

**Magnitude of error:** The ratio `1.96 / 2.262 = 0.867`, meaning all reported CIs are approximately **13.3% too narrow**. For example:

| Metric | Reported CI | Corrected CI | Difference |
|--------|-------------|--------------|------------|
| Hiring bias | +/-0.003 | +/-0.004 | 33% wider |
| Healthcare harm | +/-0.003 | +/-0.004 | 33% wider |
| Customer svc transfer ratio | +/-0.064 | +/-0.073 | 14% wider |

The rounding in the paper masks some of the effect (e.g., hiring bias CI rounds to +/-0.003 in both cases), but for quantities reported to 3 decimal places, the error is material. Transfer ratios with their larger SDs are particularly affected.

**Impact on claims:** While the qualitative conclusions are unlikely to change (effects are large), reported CIs should be widened by ~13%. The same z=1.96 error appears in `aggregate_transfer_replicates()` at line 305.

**Recommendation:** Replace `1.96` with `stats.t.ppf(0.975, n-1)` in both `aggregate_replicates()` and `aggregate_transfer_replicates()`.

---

## 2. Statistical Power Analysis

**Finding: Power is adequate for the observed effect sizes, but marginally so for smaller effects.**

With `n=10` replicates, I compute post-hoc power for the observed effects:

Each replicate uses `n_agents=100` paired observations (ingroup vs. outgroup per agent), with a paired t-test. The within-replicate t-statistics are extremely large (t=13-27), so individual replicates always achieve significance (sig_fraction=1.0 across all conditions).

The between-replicate variability is what matters for CI precision. With n=10 replicates:
- For the main domain comparison (Cohen's d of aggregate means ~1.5-3.3), power exceeds 0.99.
- For the horizon effect (bias change from 0.131 to 0.144, a ~10% increase), the effect is modest relative to between-replicate noise (SD ~0.007-0.011). A paired test across horizon levels would have moderate power (~0.6-0.8) to detect this difference.
- For the null model, the expected effect is zero. The observed bias magnitudes of 0.022-0.028 with CIs of ~0.003 suggest that even the null model produces nonzero bias. This is surprising and important (see Section 5).

**Recommendation:** Report a prospective power analysis for the minimum effect size of interest, or justify n=10 replicates as sufficient for the claimed precision.

---

## 3. Multiple Comparisons

**Finding: ISSUE -- No correction for multiple testing.**

The paper reports p-values and significance across:
- 5 domains (Experiment A)
- 7 cue strengths x 3 domains = 21 conditions (Experiment B)
- 6 horizon lengths (Experiment C)
- 5 poisoning rates (Experiment D)
- 5 domains for transferability (Experiment E)
- 15 sensitivity conditions (Experiment F)
- 5 null model domains (Experiment G)

That is 57+ hypothesis tests with no correction for multiple comparisons (no Bonferroni, no FDR/BH, no family-wise error rate control). The paper reports all tests as significant (p < 0.05) but never applies any correction.

**Mitigating factors:** The p-values are astronomically small (typically 1e-24 to 1e-50), so even a conservative Bonferroni correction with m=57 (adjusted alpha = 0.05/57 = 0.00088) would not change any significance conclusion. However, this should be explicitly stated in the paper rather than left implicit.

**Exception -- the null model:** In the null model, some p-values are larger (e.g., customer_service p=0.00014, content_moderation p=1.2e-5). Under Bonferroni correction with m=5 (null model only), these remain significant, but this is worth noting because the null model is *supposed* to show no bias, yet all five domains are significantly different from zero. This deserves discussion (see Section 5).

**Recommendation:** State that no multiple comparison correction is applied, and justify by noting that even Bonferroni-corrected p-values remain significant. Alternatively, apply Benjamini-Hochberg FDR correction and report adjusted p-values.

---

## 4. Cohen's d Computation

**Finding: ISSUE -- Denominator choice is methodologically questionable.**

The code (line 205) computes Cohen's d as:

```python
pooled_std = np.sqrt((np.std(ingroup_decisions)**2 + np.std(outgroup_decisions)**2) / 2)
cohens_d = float(bias_magnitude / pooled_std) if pooled_std > 0 else 0.0
```

This uses `np.std()` which defaults to `ddof=0` (population SD, divides by n), not `ddof=1` (sample SD, divides by n-1). For n=100 agents, the difference is small (~0.5%), but it is a systematic bias that makes all Cohen's d values slightly inflated.

More importantly, the **denominator is the between-agent SD of decision rates**, not the within-agent SD or the SD of the paired differences. This is a design choice, but it should be clearly stated. The paper says Cohen's d measures "effect size of the ingroup-outgroup decision gap" (Table 2 caption), but what it actually measures is the mean bias divided by the pooled between-agent variability. This conflates two sources of variation:

1. **Agent-level heterogeneity** in base rates (some agents get higher rates than others due to noise)
2. **The bias signal** itself

A more standard approach for paired designs would be Cohen's d_z = mean(differences) / SD(differences), which is what you get from a paired t-test. The paper's t-statistics (available in the data) would allow computing d_z = t / sqrt(n_agents).

Cross-checking: For healthcare, t=18.05, n=100 agents, so d_z = 18.05/10 = 1.81, while the reported d = 2.27. The discrepancy arises because the pooled SD of rates (~0.06) is smaller than the SD of differences (~0.075), making the reported d larger than d_z. Neither is wrong per se, but the choice should be stated and interpreted.

**Recommendation:** Either (a) switch to Cohen's d_z for the paired design, or (b) explicitly state that d uses the pooled SD of group rates (not the SD of differences) and interpret accordingly.

---

## 5. Null Model Residual Bias

**Finding: CONCERN -- Null model shows nonzero, statistically significant bias.**

The null model sets `outgroup_base_rate = ingroup_base_rate` (zero structural bias). Yet the data shows:

| Domain | Null Bias | CI95 | p-value |
|--------|-----------|------|---------|
| Customer svc | 0.0217 | 0.0030 | 0.00014 |
| Healthcare | 0.0270 | 0.0032 | 1.4e-6 |
| Content mod | 0.0243 | 0.0022 | 1.2e-5 |
| Education | 0.0262 | 0.0037 | 1.1e-6 |
| Hiring | 0.0277 | 0.0029 | 1.1e-5 |

All five domains show bias magnitude of 0.022-0.028, significantly different from zero (all p < 0.001). The paper claims "bias magnitudes are near zero across all domains (mean < 0.028)," but this understates the issue. These values are **statistically significantly nonzero** and represent approximately 15-20% of the structural model's bias for customer service (structural: 0.083, null: 0.022).

**Root cause:** The simulation model at line 139 computes `amplified_bias = base_bias * (1 + cue_strength * 2.0)` with `cue_strength=0.3`. When `base_bias=0`, `amplified_bias=0`, but then `poisoning_boost=0` (no poisoning), and `effective_bias=0`. However, the horizon feedback loop (lines 166-170) starts from `agent_bias = effective_bias + agent_noise` where `agent_noise ~ N(0, 0.05)`. Positive noise values get compounded through the horizon feedback, creating a **positive bias** on average because the horizon model clips bias at 0 (line 171: `np.clip(cumulative_bias, 0, 0.5)`). This asymmetric clipping converts symmetric noise into a systematic positive bias.

This means the null model is not truly null -- it has a built-in positive bias floor due to the clipping/feedback interaction. The paper should discuss this artifact and potentially report bias magnitudes net of the null model baseline.

**Recommendation:** Either (a) report structural model bias as (structural - null) to remove the artifact, (b) remove the lower clip at 0 (allow negative bias), or (c) discuss this as a limitation and quantify its magnitude.

---

## 6. Sensitivity Analysis: One-at-a-Time Adequacy

**Finding: LIMITATION -- OAT sensitivity analysis misses interaction effects.**

The sensitivity analysis varies each parameter independently at +/-25% and +/-50% while holding others fixed (one-at-a-time / OAT). This approach:

- **Covers:** Main effects of each parameter on outputs.
- **Misses:** Interaction effects (e.g., does high stakes + high base bias produce disproportionately worse outcomes than the sum of their individual effects?). Given that harm is computed as `bias * stakes * harm_weight` (Eq. 3), there are obvious multiplicative interactions.
- **Misses:** Joint parameter uncertainty. In practice, all parameters are uncertain simultaneously. A domain might have both higher-than-expected stakes and higher-than-expected base bias.

For the harm score equation H = (r_in - r_out) * s * w, the OAT approach can only reveal that harm is linear in stakes and harm_weight (trivially true from the equation) and roughly linear in base_bias (true by construction since bias ~ base_bias * (1 + 2c)). The sensitivity analysis largely confirms what the equations already guarantee.

Additionally, the sensitivity analysis only examines one domain (healthcare_triage). The claim that "the ranking of hiring and healthcare as highest-risk domains is robust to parameter perturbation" (Section 5.8 and Discussion) is supported by the tornado diagram for healthcare, but there is no explicit cross-domain robustness check where, e.g., education's parameters are increased while healthcare's are decreased.

**Recommendation:** Either (a) add a global sensitivity analysis (e.g., Sobol indices) to capture interactions, or (b) add explicit cross-domain perturbation scenarios (e.g., swap healthcare and education parameters to test if ranking inverts), or (c) acknowledge the limitation of OAT more explicitly.

---

## 7. Cross-Check of Numerical Claims Against Data

### 7a. Domain comparison (Table 3 / Abstract)

| Claim (Paper) | Data (table_summary.json) | Match? |
|---|---|---|
| Hiring bias: 0.206 +/- 0.003 | 0.20622, CI=0.00347 | YES (rounds to 0.206 +/- 0.003) |
| Hiring harm: 0.149 +/- 0.002 | 0.14858, CI=0.00200 | YES |
| Hiring DI: 0.411 | 0.41087 | YES |
| Healthcare harm: 0.115 +/- 0.003 | 0.11500, CI=0.00322 | YES |
| Healthcare DI: 0.849 | 0.84897 | YES |
| Cust svc DI: 0.903 | 0.90294 | YES |
| Content mod DI: 0.847 | 0.84686 | YES |
| Education DI: 0.849 | 0.84859 | YES |
| Cust svc bias: 0.083 +/- 0.004 | 0.08256, CI=0.00381 | YES |
| Cust svc harm: 0.007 +/- 0.000 | 0.00740, CI=0.00029 | YES |
| Cust svc d: 1.56 | 1.5647 | YES |
| Healthcare d: 2.27 | 2.2687 | YES |
| Content mod bias: 0.115 +/- 0.007 | 0.11479, CI=0.00685 | YES |
| Content mod harm: 0.035 +/- 0.002 | 0.03463, CI=0.00167 | YES |
| Content mod d: 1.98 | 1.9771 | YES |
| Education bias: 0.133 +/- 0.004 | 0.13268, CI=0.00419 | YES |
| Education harm: 0.066 +/- 0.002 | 0.06574, CI=0.00168 | YES |
| Education d: 2.30 | 2.3043 | YES |
| Hiring d: 3.28 | 3.2830 | YES |
| Healthcare bias: 0.136 +/- 0.004 | 0.13607, CI=0.00428 | YES |

All Table 3 values match the data within rounding.

### 7b. Horizon effects (Section 4.3)

Paper claims: "mean bias increases from 0.131 +/- 0.004 at horizon 1 to 0.144 +/- 0.007 at horizon 50, representing an approximately 10% increase."

Data (table_by_horizon.json):
- Horizon 1: bias = 0.130952, CI = 0.003767 -> rounds to 0.131 +/- 0.004 (MATCH)
- Horizon 50: bias = 0.144018, CI = 0.006712 -> rounds to 0.144 +/- 0.007 (MATCH)
- Relative increase: (0.144018 - 0.130952) / 0.130952 = 9.98% -> "approximately 10%" (MATCH)

Note: The figure caption says "approximately 11%" while the body text says "approximately 10%." The data supports ~10%.

### 7c. Poisoning effects (Section 4.5)

Paper claims: "belief poisoning at 30% rate increases bias from 0.134 +/- 0.003 to 0.227 +/- 0.004, a relative increase of approximately 70%."

Data (table_by_poisoning.json):
- Rate 0.0: bias = 0.133554, CI = 0.003093 -> rounds to 0.134 +/- 0.003 (MATCH)
- Rate 0.3: bias = 0.226694, CI = 0.004065 -> rounds to 0.227 +/- 0.004 (MATCH)
- Relative increase: (0.226694 - 0.133554) / 0.133554 = 69.7% -> "approximately 70%" (MATCH)

Paper claims: "At 50% poisoning, bias reaches 0.287 +/- 0.005, representing a 115% relative increase."

Data: Rate 0.5: bias = 0.286822, CI = 0.004623 -> rounds to 0.287 +/- 0.005 (MATCH)
- Relative increase: (0.286822 - 0.133554) / 0.133554 = 114.8% -> "115%" (MATCH)

### 7d. Transfer ratios (Table 4 / Abstract)

Paper claims: "Transfer ratios range from 0.82 to 0.85."

Data (table_transferability.json):
- Customer svc: 0.838
- Healthcare: 0.853
- Content mod: 0.822
- Education: 0.829
- Hiring: 0.817

Range: [0.817, 0.853]. Paper says "0.82 to 0.85" which rounds the min down (0.817 -> 0.82) and the max up (0.853 -> 0.85). This is acceptable approximation but slightly imprecise.

Paper Table 4 values match the data:
| Domain | Paper Transfer | Data Transfer | Paper CI | Data CI |
|--------|----------------|---------------|----------|---------|
| Cust svc | 0.838 +/- 0.064 | 0.838, CI=0.064 | MATCH | MATCH |
| Healthcare | 0.853 +/- 0.046 | 0.853, CI=0.046 | MATCH | MATCH |
| Content mod | 0.822 +/- 0.037 | 0.822, CI=0.037 | MATCH | MATCH |
| Education | 0.829 +/- 0.041 | 0.829, CI=0.041 | MATCH | MATCH |
| Hiring | 0.817 +/- 0.014 | 0.817, CI=0.014 | MATCH | MATCH |

Paper Table 4 also reports "Harm Amplification":
| Domain | Paper | Data |
|--------|-------|------|
| Cust svc | 0.834 +/- 0.052 | 0.834, CI=0.052 | MATCH |
| Healthcare | 0.848 +/- 0.041 | 0.848, CI=0.041 | MATCH |
| Content mod | 0.827 +/- 0.040 | 0.827, CI=0.040 | MATCH |
| Education | 0.834 +/- 0.041 | 0.834, CI=0.041 | MATCH |
| Hiring | 0.819 +/- 0.012 | 0.819, CI=0.012 | MATCH |

### 7e. Null model (Section 4.7)

Paper claims: "bias magnitudes are near zero across all domains (mean < 0.028, with disparate impact ratios > 0.92)."

Data (table_null_model.json):
- Max bias: hiring at 0.0277 -> < 0.028 (MATCH)
- Min DI: hiring at 0.921 -> > 0.92 (MATCH)

### 7f. Fairness metrics (Table 6)

Paper Table 6 vs. table_by_domain.json (which contains EO and PP metrics from Experiment A):

| Domain | Paper EO Diff | Data EO Diff | Match? |
|--------|---------------|--------------|--------|
| Cust svc | 0.001 +/- 0.002 | 0.001, CI=0.00196 | YES |
| Healthcare | 0.000 +/- 0.000 | 0.0, CI=0.0 | YES |
| Content mod | 0.041 +/- 0.013 | 0.041, CI=0.01353 | YES |
| Education | 0.001 +/- 0.002 | 0.001, CI=0.00196 | YES |
| Hiring | 0.000 +/- 0.000 | 0.0, CI=0.0 | YES |

All PP differences are 0.000 in both paper and data. MATCH.

### 7g. Figure caption discrepancy

The caption of Figure 4 states "increasing approximately 11% from horizon 1 to 50" while the text in Section 4.3 says "approximately 10%." The data supports 9.98%. This is a minor inconsistency; "approximately 10%" is more accurate.

---

## 8. Additional Statistical Concerns

### 8a. Paired t-test appropriateness

The code uses `stats.ttest_rel` (paired t-test) at line 202, which is appropriate since each agent has both ingroup and outgroup decisions. However, the 100 agents within a single replicate share the same RNG stream, and each agent's decisions are generated from `rng.binomial(n_interactions, rate)`. The independence assumption is satisfied between agents (each gets its own noise draws), so this is correct.

### 8b. Aggregation strategy

Results are first computed per-replicate (each with 100 agents), then aggregated across 10 replicates. The CI is computed on the 10 replicate means. This is a valid approach (treating replicate as the unit of replication), but it means the CIs reflect between-replicate variability, not within-replicate uncertainty. For n=100 agents per replicate, within-replicate uncertainty is very small. The between-replicate approach is more conservative and appropriate.

### 8c. SeedSequence usage

The code uses `np.random.SeedSequence(42)` and spawns children via `ss.spawn(1)[0]` for each experiment-replicate combination. This ensures independent streams. However, the seeds are generated sequentially (not indexed by experiment-replicate), so the mapping from seed to experiment depends on execution order. This is fine for reproducibility (same execution order always yields same results) but fragile if experiments are reordered or added.

---

## 9. Summary of Findings

| Issue | Severity | Impact on Conclusions |
|-------|----------|-----------------------|
| z=1.96 instead of t(9,0.975)=2.262 | **Medium** | All CIs ~13% too narrow; qualitative conclusions unchanged |
| No multiple comparison correction | **Low** | All p-values survive even Bonferroni; should be stated |
| Cohen's d denominator choice | **Low-Medium** | Reported d values are valid but non-standard for paired design; should be documented |
| Null model residual bias | **Medium** | Structural artifact inflates null model bias by 0.022-0.028; undermines "near zero" claim |
| OAT sensitivity analysis | **Low** | Misses interactions but adequate for stated claims; should acknowledge limitation |
| Figure 4 caption vs text: 11% vs 10% | **Low** | Minor inconsistency; 10% is correct |
| All numerical claims vs data | **Pass** | All values match within rounding |

### Overall Assessment

The statistical methodology is generally sound for a simulation study. The numerical claims are faithfully reproduced from the underlying data. The two most substantive issues are (1) the use of z=1.96 instead of the t-distribution for CIs with n=10, which systematically narrows all CIs by ~13%, and (2) the null model showing statistically significant nonzero bias due to the asymmetric clipping in the horizon feedback loop, which somewhat undermines the paper's claim that the null model confirms effects arise from structural bias rather than simulation artifacts. Neither issue changes the qualitative conclusions, but both should be corrected for methodological rigor.
