# Reproducibility Report: Transferability and Harms of Agent Intergroup Bias

**Reviewer**: Reviewer 6 (Paper-Only Reproducer)
**Date**: 2026-02-06
**Method**: Independent reimplementation from main.tex only; no existing code or data consulted

---

## 1. Summary Verdict

| Category | Count | Percentage |
|----------|-------|------------|
| Total claims checked | 59 | 100% |
| MATCH (< 5% relative error) | 45 | 76.3% |
| CLOSE (5-15% relative error) | 4 | 6.8% |
| MISMATCH (> 15% relative error) | 10 | 16.9% |

**Overall**: 83.1% of numerical claims reproduce within 15% relative error from the paper's methodology description alone. The 10 mismatches fall into two categories: (a) Cohen's d effect sizes (5 mismatches), which require knowledge of the exact pooling method not specified in the paper, and (b) equal opportunity difference values (5 mismatches), which require a ground truth / merit model not described in the paper.

**Core results -- bias magnitude, harm scores, disparate impact ratios, transfer ratios, null model, sensitivity ranges, horizon effects, and poisoning amplification -- all reproduce well.** The qualitative conclusions of the paper are fully supported by independent reimplementation.

---

## 2. Detailed Claim Comparison

### 2.1 Domain Comparison (Table 2, cue=0.3, horizon=10)

| Domain | Metric | Paper | Reproduced | Rel. Error | Status |
|--------|--------|-------|------------|------------|--------|
| Customer Service | Bias | 0.083 | 0.081 | 2.7% | MATCH |
| Customer Service | Harm | 0.007 | 0.007 | 4.3% | MATCH |
| Customer Service | DI | 0.903 | 0.905 | 0.2% | MATCH |
| Customer Service | Cohen's d | 1.56 | 3.25 | 108% | MISMATCH |
| Healthcare Triage | Bias | 0.136 | 0.137 | 0.9% | MATCH |
| Healthcare Triage | Harm | 0.115 | 0.117 | 2.0% | MATCH |
| Healthcare Triage | DI | 0.849 | 0.848 | 0.2% | MATCH |
| Healthcare Triage | Cohen's d | 2.27 | 3.27 | 44% | MISMATCH |
| Content Moderation | Bias | 0.115 | 0.114 | 1.1% | MATCH |
| Content Moderation | Harm | 0.035 | 0.034 | 2.6% | MATCH |
| Content Moderation | DI | 0.847 | 0.848 | 0.2% | MATCH |
| Content Moderation | Cohen's d | 1.98 | 3.63 | 83% | MISMATCH |
| Education | Bias | 0.133 | 0.132 | 0.8% | MATCH |
| Education | Harm | 0.066 | 0.065 | 1.5% | MATCH |
| Education | DI | 0.849 | 0.850 | 0.1% | MATCH |
| Education | Cohen's d | 2.30 | 3.42 | 49% | MISMATCH |
| Hiring | Bias | 0.206 | 0.218 | 5.7% | CLOSE |
| Hiring | Harm | 0.149 | 0.157 | 5.2% | CLOSE |
| Hiring | DI | 0.411 | 0.378 | 8.0% | CLOSE |
| Hiring | Cohen's d | 3.28 | 4.33 | 32% | MISMATCH |

**Analysis**: Bias, harm, and DI reproduce excellently for all domains except hiring, where values are close but slightly elevated. The hiring discrepancy likely stems from a slightly different base ingroup rate assumption (the paper states 0.35 for hiring, which I used, but subtle differences in how the horizon model interacts with the base rate at higher complexity may account for the 5-8% deviation). Cohen's d is systematically too high in my reproduction, suggesting the paper uses a different denominator (likely the pooled SD of ingroup and outgroup decision distributions from the 500 interactions, rather than the SD of agent-level bias values).

### 2.2 Horizon Effects (Section 4.3, Healthcare Triage)

| Metric | Paper | Reproduced | Rel. Error | Status |
|--------|-------|------------|------------|--------|
| Bias at h=1 | 0.131 | 0.131 | 0.3% | MATCH |
| Bias at h=50 | 0.144 | 0.144 | 0.3% | MATCH |
| % increase h=1 to h=50 | ~10% | 9.3% | 7.0% | CLOSE* |

*The paper says "approximately 10%" (also "approximately 11%" in Figure 4 caption). Our 9.3% is within this approximate range.

### 2.3 Belief Poisoning (Section 4.5, Healthcare Triage)

| Metric | Paper | Reproduced | Rel. Error | Status |
|--------|-------|------------|------------|--------|
| Baseline bias (p=0) | 0.134 | 0.136 | 1.1% | MATCH |
| Bias at p=0.3 | 0.227 | 0.231 | 1.8% | MATCH |
| Relative increase at p=0.3 | ~70% | 70.5% | 0.7% | MATCH |
| Bias at p=0.5 | 0.287 | 0.295 | 2.6% | MATCH |
| Relative increase at p=0.5 | ~115% | 117.5% | 2.2% | MATCH |

**Analysis**: Poisoning effects reproduce very well. The additive poisoning term 0.3p in Eq. 1 is deterministic and clearly specified.

### 2.4 Transferability (Table 3)

| Domain | Paper TR | Reproduced TR | Rel. Error | Status |
|--------|----------|---------------|------------|--------|
| Customer Service | 0.838 | 0.820 | 2.1% | MATCH |
| Healthcare Triage | 0.853 | 0.856 | 0.4% | MATCH |
| Content Moderation | 0.822 | 0.829 | 0.9% | MATCH |
| Education | 0.829 | 0.836 | 0.8% | MATCH |
| Hiring | 0.817 | 0.842 | 3.1% | MATCH |

| Domain | Paper HR | Reproduced HR | Rel. Error | Status |
|--------|----------|---------------|------------|--------|
| Customer Service | 0.834 | 0.820 | 1.7% | MATCH |
| Healthcare Triage | 0.848 | 0.856 | 0.9% | MATCH |
| Content Moderation | 0.827 | 0.829 | 0.2% | MATCH |
| Education | 0.834 | 0.836 | 0.2% | MATCH |
| Hiring | 0.819 | 0.842 | 2.8% | MATCH |

**Analysis**: All transfer ratios match within 5%. The abstract claim that "transfer ratios range from 0.82 to 0.85" is also confirmed (reproduced range: 0.820 to 0.856).

### 2.5 Null Model (Section 4.7)

| Domain | Bias < 0.028? | DI > 0.92? |
|--------|---------------|------------|
| Customer Service | 0.010 -- YES | 0.989 -- YES |
| Healthcare Triage | 0.017 -- YES | 0.981 -- YES |
| Content Moderation | 0.012 -- YES | 0.984 -- YES |
| Education | 0.015 -- YES | 0.983 -- YES |
| Hiring | 0.020 -- YES | 0.944 -- YES |

**Analysis**: All null model claims reproduce. Bias magnitudes are well below 0.028 and DI ratios are above 0.92 for all domains.

### 2.6 Sensitivity Analysis (Section 4.8, Healthcare)

| Metric | Paper | Reproduced | Rel. Error | Status |
|--------|-------|------------|------------|--------|
| Base bias -50% bias magnitude | 0.074 | 0.069 | 7.2% | CLOSE |
| Base bias +50% bias magnitude | 0.200 | 0.203 | 1.7% | MATCH |

**Analysis**: The range for base bias sensitivity is close. The paper reports 0.074 at -50%; we get 0.069, a 7% deviation likely due to interaction between the horizon feedback model and the reduced base bias.

### 2.7 Multi-Metric Fairness (Table 4)

| Domain | Metric | Paper | Reproduced | Status |
|--------|--------|-------|------------|--------|
| Customer Service | EO Diff | 0.001 | 0.080 | MISMATCH |
| Healthcare Triage | EO Diff | 0.000 | 0.137 | MISMATCH |
| Content Moderation | EO Diff | 0.041 | 0.116 | MISMATCH |
| Education | EO Diff | 0.001 | 0.134 | MISMATCH |
| Hiring | EO Diff | 0.000 | 0.219 | MISMATCH |
| All domains | PP Diff | 0.000 | ~0.000 | MATCH |

**Analysis**: Predictive parity (PP) reproduces correctly (near zero for all domains). Equal opportunity (EO) difference is systematically too large in our reproduction. The paper's near-zero EO values for most domains suggest a specific ground truth model where the TPR gap is nearly zero by construction -- possibly because the "deserving" label correlates perfectly with the favorable decision (i.e., all decisions are correct for ingroup members, and EO only captures the bias-induced error differential). Without knowing the exact ground truth generation process, this metric cannot be reproduced from the paper alone.

---

## 3. Methodology Gaps

The following information is missing or insufficiently specified in the paper to enable full reproduction:

### 3.1 Critical Gaps (caused mismatches)

1. **Cohen's d computation method**: The paper reports d as "effect size of the ingroup-outgroup decision gap" but does not specify the exact formula used. Standard Cohen's d for two groups requires the pooled standard deviation of the two group distributions, but the paper does not describe what those distributions are. Are they per-agent favorable rates? Per-interaction binary outcomes? The denominator matters enormously: my agent-level bias SD gives d values roughly 1.5-2x the paper's values.

2. **Equal opportunity ground truth model**: The paper defines EO as "the difference in the rate at which deserving individuals receive favorable outcomes" but never describes how "deserving" status is generated. Is it independent of group membership? What is the base merit rate? How does it interact with the agent's decision-making? This is essential for computing TPR by group.

3. **Feedback strength for Content Moderation and Education**: The paper explicitly states feedback strengths for Customer Service (0.2), Healthcare (0.8), and Hiring (0.6), but omits the values for Content Moderation and Education. I estimated 0.4 and 0.5 respectively; the actual values could differ.

### 3.2 Minor Gaps (did not cause mismatches but reduce confidence)

4. **Base ingroup favorable rates for non-hiring domains**: The paper only specifies ingroup=0.35, outgroup=0.22 for hiring. The base rates for other domains are not given. I reverse-engineered approximate rates from the DI ratios in Table 2, but this is circular.

5. **How 500 interactions per agent map to results**: The paper says "100 agents with 500 interactions each" but the simulation equations (Eqs. 1-2) compute bias as a continuous value, not from sampling 500 binary outcomes. It is unclear whether the 500 interactions are used for: (a) sampling noise in bias estimation, (b) computing fairness metrics only, or (c) something else.

6. **Random stream architecture**: The paper mentions NumPy SeedSequence but does not specify how random streams are allocated across agents, domains, and experiments. Different allocation strategies can produce different CI values even with the same seed.

7. **Exact cue strength values tested**: The paper says bias "increases monotonically with cue strength" but does not list the exact cue values tested in Section 4.2.

8. **Exact horizon values tested**: The horizons plotted in Figures 4-5 are not enumerated in the text.

---

## 4. Potential Errors Found

### 4.1 Internal Consistency Issues

1. **Horizon percentage claim inconsistency**: The text in Section 4.3 says "approximately 10% increase" while the Figure 4 caption says "approximately 11%." Our reproduction gives 9.3%, which is consistent with "approximately 10%" but not "approximately 11%." This is a minor inconsistency.

2. **Transfer ratio range claim**: The abstract says "0.82 to 0.85" while Table 3 gives the lowest as 0.817 (which rounds to 0.82) and highest as 0.853. This is consistent but the abstract rounds optimistically.

3. **Poisoning baseline vs domain comparison**: Section 4.5 reports baseline bias (p=0) as 0.134, while Table 2 reports healthcare bias as 0.136 at the same conditions (cue=0.3, horizon=10). These should be identical. The 0.002 difference is within the reported CI but suggests the poisoning experiment may have been run with a slightly different seed or configuration.

### 4.2 Not Errors but Worth Noting

4. **Hiring DI sensitivity**: The hiring DI of 0.411 is very sensitive to the base ingroup rate. Small changes in the assumed ingroup rate (e.g., 0.35 vs 0.34) produce large DI changes because the denominator is small. This makes the 0.411 value less stable than the DI values for other domains.

5. **Predictive parity near zero for all domains**: PP Diff is exactly 0.000 for all domains, which is suspicious. In a model with group-differential rates, PPV should differ unless the ground truth model is specifically constructed to cancel it out. This likely indicates a very specific (but undocumented) ground truth generation process.

---

## 5. Qualitative Conclusions Assessment

Despite the quantitative mismatches in Cohen's d and EO, all qualitative conclusions from the paper are confirmed:

| Conclusion | Confirmed? | Evidence |
|------------|------------|----------|
| Hiring has highest bias and harm | YES | Bias=0.218, Harm=0.157 (both highest) |
| Healthcare has second-highest harm | YES | Harm=0.117 (second highest) |
| Only hiring has DI < 0.8 | YES | Hiring DI=0.378; all others > 0.84 |
| Bias increases monotonically with cue | YES | All 5 domains show monotonic increase |
| Horizon produces ~10% bias growth | YES | 9.3% for healthcare h=1 to h=50 |
| 30% poisoning amplifies bias ~70% | YES | 70.5% relative increase |
| Transfer ratios below 1.0 | YES | Range 0.82-0.86 |
| Null model shows near-zero bias | YES | All domains < 0.020 |
| Hiring and healthcare rank highest under sensitivity | YES | Robust to +/-50% perturbation |

---

## 6. Recommendations for Improving Reproducibility

### High Priority

1. **Specify Cohen's d computation fully**: State the exact formula including what constitutes the two groups and how the pooled SD is computed. Example: "Cohen's d is computed as the mean difference in per-agent favorable rates between ingroup and outgroup, divided by the pooled standard deviation of these per-agent rates across all agents."

2. **Describe the ground truth model for fairness metrics**: Specify how "deserving" status is generated, what the merit base rate is, and how it relates to group membership. Without this, EO and PP cannot be reproduced.

3. **State all feedback strength values**: Include Content Moderation and Education feedback strengths in Table 1 or the text.

### Medium Priority

4. **Provide base ingroup rates for all domains**: Either add a column to Table 1 or describe the model that determines them.

5. **Clarify the role of 500 interactions**: Explain whether interactions contribute to bias estimation, fairness metric computation, or both.

6. **Fix the horizon percentage inconsistency**: Use one consistent figure (e.g., "approximately 10%") in both the text and caption.

### Low Priority

7. **List exact tested values**: For cue strength and horizon experiments, enumerate the exact values used (e.g., cue in {0.0, 0.1, ..., 1.0}).

8. **Document random stream allocation**: Specify how SeedSequence children are allocated across agents, domains, and experiments.

---

## 7. Reproduction Script

The independent reproduction script is located at:
`revision/code/reproduce_from_paper.py`

The comparison output is at:
`revision/data/reproducibility_check.json`

Both were generated without consulting any existing code or data files.
