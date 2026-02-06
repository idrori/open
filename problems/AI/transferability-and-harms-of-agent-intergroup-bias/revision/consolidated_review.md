# Consolidated Review: Transferability and Harms of Agent Intergroup Bias

## Overall Assessment

Six independent reviews examined the code, statistics, fairness metrics, domain calibration, model structure, and paper-only reproducibility. The simulation is fully deterministic and bit-for-bit reproducible (Reviewer 1). All core numerical claims (bias magnitudes, harm scores, DI ratios, transfer ratios, poisoning effects) match the underlying data within rounding (Reviewers 1, 2, 6). Qualitative conclusions are confirmed by independent reimplementation (Reviewer 6). However, several Critical and Major issues were identified that affect methodological rigor and the validity of specific claims.

---

## Critical Issues

### C1. Null model clip artifact creates systematic positive bias
**Reviewers**: 2, 5
**Location**: `run_experiments.py` line 171
**Problem**: `np.clip(cumulative_bias, 0, 0.5)` clips negative bias values to 0. In the null model (base_bias=0), agent noise N(0, 0.05) is symmetric, but clipping converts negative noise to 0 while passing positive noise through. This creates a mathematically predictable positive bias floor of ~0.02 (E[clip(N(0,0.05), 0, 0.5)] ~ 0.05*sqrt(2/pi) ~ 0.020). The horizon feedback then compounds this artifact. The paper reports null model bias < 0.028 as "near zero" but this is an artifact, not noise.
**Impact**: Undermines the null model validation claim (Section 4.7) that "observed effects arise from structural bias rather than artifacts."
**Action**: Change clip to `np.clip(cumulative_bias, -0.5, 0.5)` to allow negative bias in the null model. This makes the null model truly null while still preventing extreme values.

### C2. Fairness metrics (EO, PP) are degenerate due to positive_threshold=0.5
**Reviewer**: 3
**Location**: `run_experiments.py` line 154
**Problem**: Hardcoded `positive_threshold=0.5` interacts pathologically with domain base rates. For hiring (base rate 0.35), virtually no agent exceeds 0.5, so EO is 0/0 defaulting to 0.0. For healthcare (base rate 0.90), virtually all agents exceed 0.5, so TPR~1.0 for both groups and EO~0.0. PP is identically 0.0 for all domains across all conditions. Only DI provides meaningful signal.
**Impact**: Table 5, Figure 8, and Contribution 4 (fairness metric divergence) are not supported by the data. The impossibility theorem discussion is unwarranted.
**Action**: Use domain-specific thresholds: `threshold = 0.5 * (ingroup_base_rate + outgroup_base_rate)` so the binarization is meaningful relative to each domain's base rates.

### C3. CI computation uses z=1.96 instead of t-distribution
**Reviewers**: 1, 2
**Location**: `run_experiments.py` lines 282, 305
**Problem**: With n=10 replicates (df=9), the correct 95% CI multiplier is t(9,0.975)=2.262, not z=1.96. All CIs are systematically ~13% too narrow.
**Impact**: All reported CIs in Tables 2-4 and all error bars in figures are underestimated. Qualitative conclusions unchanged.
**Action**: Replace `1.96` with `stats.t.ppf(0.975, n-1)` in both `aggregate_replicates()` and `aggregate_transfer_replicates()`.

---

## Major Issues

### M1. Incorrect fairness metric text in Section 4.8
**Reviewer**: 3
**Location**: `main.tex` line 320
**Problem**: Paper claims "equal opportunity difference ranks healthcare higher than content moderation" but data shows healthcare EO=0.000 and content moderation EO=0.041 -- the opposite.
**Action**: Fix text after correcting the EO metric implementation (C2). The new data will require rewriting this claim.

### M2. Impossibility theorem invocation unsupported
**Reviewer**: 3
**Problem**: The paper invokes Chouldechova's impossibility theorem to explain metric "divergence," but two of three metrics are degenerate, not divergent. After fixing C2, this section should be rewritten based on actual metric behavior.
**Action**: Rewrite Section 4.8 discussion based on corrected fairness metrics data.

### M3. Calibration claims overstated
**Reviewer**: 4
**Problem**: Table 1 caption says "calibrated from domain-specific empirical literature" but only hiring has approximate calibration (relative ratio). Healthcare calibration is weak (no quantitative link). Three domains (customer service, content moderation, education) have no empirical calibration. Three calibration sources (Dee 2005, Sap et al. 2019, Gneezy & List 2004) are cited in code but missing from references.bib.
**Action**: Change "calibrated from" to "informed by" in Table 1 caption. Add missing references. Add honest discussion distinguishing grounded vs. assumed parameters.

### M4. Bertrand & Mullainathan gap size misrepresented
**Reviewer**: 4
**Problem**: The paper says B&M documented "a roughly 50% difference in callback rates." B&M found ~3.2 pp absolute gap (6.5% vs 9.7%), which is ~50% relative difference. The model's absolute gap of 13 pp (0.35-0.22) is ~4x larger than the real absolute gap. The model captures the relative ratio from Quillian (~36%), not absolute rates.
**Action**: Clarify in Section 3.3 that calibration targets the relative discrimination ratio from the Quillian meta-analysis, not absolute callback rates.

### M5. Harm metric is a weighted rate gap, not utility-theoretic harm
**Reviewer**: 5
**Problem**: H = (r_in - r_out) * s * w is called "harm" throughout, but it is a unitless weighted gap, not a welfare or loss function. Harm rankings are predetermined by parameter choices (s, w).
**Action**: Add clarifying sentence in the harm discussion and note in Limitations that harm rankings reflect parameter choices.

### M6. No multiple comparison correction stated
**Reviewer**: 2
**Problem**: 57+ hypothesis tests with no correction. P-values are tiny enough that Bonferroni would not change conclusions, but this should be explicitly stated.
**Action**: Add a sentence in Section 3.8 noting that no multiple comparison correction is applied and justifying by reference to the very small p-values.

### M7. Cohen's d denominator not documented
**Reviewers**: 2, 6
**Problem**: Code uses pooled SD of group rates (ddof=0), not SD of paired differences. This is non-standard for paired designs and not documented in the paper. Reproducibility reviewer could not reproduce d values.
**Action**: Add explicit statement in paper that d uses pooled SD of per-agent group rates, not paired differences.

---

## Minor Issues

### m1. Vestigial np.random.seed(42) calls
**Reviewer**: 1
**Location**: `run_experiments.py` lines 316, 562
**Action**: Remove both calls.

### m2. Unused seed_counter variable
**Reviewer**: 1
**Location**: `run_experiments.py` lines 345-349
**Action**: Remove seed_counter.

### m3. Horizon percentage inconsistency (10% vs 11%)
**Reviewers**: 1, 2, 6
**Location**: Figure 4 caption says "approximately 11%", text says "approximately 10%". Data shows 9.98%.
**Action**: Harmonize to "approximately 10%" everywhere.

### m4. Griggs v. Duke Power citation for 4/5ths rule
**Reviewer**: 3
**Problem**: 4/5ths rule comes from 1978 EEOC Uniform Guidelines, not Griggs (1971).
**Action**: Minor -- note in text or add EEOC citation.

### m5. Missing citations in references.bib
**Reviewer**: 4
**Problem**: Dee (2005), Sap et al. (2019), Gneezy & List (2004) cited in code but not in paper.
**Action**: Add these three references to references.bib and cite in Section 3.3.

### m6. OAT sensitivity analysis misses interactions
**Reviewers**: 2, 5
**Problem**: One-at-a-time sensitivity misses joint perturbation effects.
**Action**: Acknowledge in Limitations section.

### m7. Transferability ratio is within-model, not genuine transfer
**Reviewer**: 5
**Problem**: "Lab" and "deployment" are two parameter settings in the same model, not genuinely different environments. T < 1 is algebraically predetermined by cue strength ratio.
**Action**: Already partially addressed in Limitations ("Simplified transferability"). Could add more explicit acknowledgment.

### m8. Cohen's d uses ddof=0 (population SD)
**Reviewer**: 2
**Location**: `run_experiments.py` line 205
**Problem**: np.std defaults to ddof=0. For n=100, difference is ~0.5%.
**Action**: Minor, but switch to ddof=1 for correctness.

---

## Action Items Summary

### Code fixes (run_experiments.py)
1. **[C1]** Line 171: Change `np.clip(cumulative_bias, 0, 0.5)` to `np.clip(cumulative_bias, -0.5, 0.5)`
2. **[C2]** Line 154: Replace hardcoded `positive_threshold = 0.5` with domain-specific threshold `0.5 * (domain.ingroup_base_rate + domain.outgroup_base_rate)`
3. **[C3]** Lines 282, 305: Replace `1.96` with `stats.t.ppf(0.975, n-1)`
4. **[m1]** Remove `np.random.seed(42)` on lines 316 and 562
5. **[m2]** Remove `seed_counter` on line 345
6. **[m8]** Line 205: Use `ddof=1` in np.std calls for Cohen's d

### Paper fixes (main.tex)
1. **[C3]** Update CI description to mention t-distribution
2. **[M1/M2]** Rewrite Section 4.8 fairness discussion based on corrected metrics data
3. **[M3]** Change Table 1 caption "calibrated from" to "informed by"
4. **[M4]** Clarify Bertrand & Mullainathan interpretation (relative ratio, not absolute rates)
5. **[M5]** Add note that harm is a weighted rate gap proxy, not utility-theoretic harm
6. **[M6]** Add multiple comparison correction statement
7. **[M7]** Document Cohen's d computation method
8. **[m3]** Fix Figure 4 caption: "approximately 11%" -> "approximately 10%"
9. **[m5]** Add missing references (Dee, Sap, Gneezy)
10. **[m6]** Acknowledge OAT limitation in Limitations section
11. **[C1]** Update null model discussion to reflect corrected results
12. **[M5]** Strengthen Limitations section with harm metric and model structure caveats

### Post-fix
1. Rerun `run_experiments.py`
2. Rerun `generate_figures.py`
3. Update all numerical values in main.tex from new data
4. Recompile PDF
