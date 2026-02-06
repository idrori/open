# Review 4: Domain Calibration Check

## Summary Verdict

The paper's calibration claims are **partially honest but significantly overstated**. The hiring domain has the strongest empirical grounding but the specific parameter values (ingroup=0.35, outgroup=0.22) do not directly correspond to the cited literature. Healthcare calibration is only loosely connected to the cited source. Three of five domains (customer service, content moderation, education) lack any empirical calibration and rely on assumed parameters. The paper's language suggests tighter calibration than actually exists.

---

## 1. Hiring: Bertrand & Mullainathan (2004) and Quillian et al. (2017)

### What the paper claims
The paper states (Section 3.3): "Our base bias of 0.13 for hiring (ingroup rate 0.35, outgroup rate 0.22) reflects approximately the magnitude observed in these audit studies." The code comments say: "Quillian et al. (2017): ~35% callback gap; ingroup=0.35, outgroup=0.22."

### What the cited papers actually found
- **Bertrand & Mullainathan (2004)**: White-sounding names received ~9.65% callback rates; African-American-sounding names received ~6.45% callback rates. This is a **3.2 percentage point gap** (or ~50% relative difference). The absolute callback rates are ~6.5% and ~9.7%, NOT 22% and 35%.
- **Quillian et al. (2017)**: Meta-analysis finding that whites receive on average **36% more callbacks** than African Americans across 28 field experiments spanning 1989-2015.

### Assessment

**MAJOR ISSUE: The code comment "~35% callback gap" conflates two different things.** Quillian et al. found a 36% *relative* difference in callback rates. The model uses 0.35 and 0.22 as *absolute* rates, which happen to produce a 37% relative gap ((0.35-0.22)/0.35 = 0.37). This is numerically coincidental alignment with the Quillian meta-analysis ratio, but the absolute values (35% and 22% callback rates) are **3-4x higher than actual callback rates** in the Bertrand & Mullainathan study (~6.5% and ~9.7%).

The base bias of 0.13 (the absolute gap) is also much larger than the real-world gap of ~3.2 percentage points found by Bertrand & Mullainathan.

**However**, the paper does say "approximately the magnitude," and one could argue the model is capturing the *relative* discrimination ratio rather than the absolute callback rates. The paper is somewhat honest about this being a simplification but does not acknowledge the 3-4x inflation of absolute rates.

**Verdict**: Partially grounded. The relative ratio roughly matches Quillian et al., but the absolute values are significantly inflated. The paper should explicitly state it is calibrating to the relative discrimination ratio, not absolute callback rates.

---

## 2. Healthcare: Hoffman et al. (2016) and Obermeyer et al. (2019)

### What the paper claims
The paper states that healthcare parameters are "calibrated from the pain assessment disparity literature: Hoffman et al. (2016) found that a substantial fraction of medical trainees held false beliefs about biological differences between racial groups, leading to biased treatment recommendations."

### Model parameters
- Healthcare: ingroup_base_rate=0.90, outgroup_base_rate=0.82 (gap of 0.08)
- stakes=0.95, harm_weight=0.90

### What the cited papers actually found
- **Hoffman et al. (2016)**: Found that ~50% of white medical students/residents endorsed false beliefs about biological differences between races, and those who endorsed these beliefs rated Black patients' pain as lower and made less accurate treatment recommendations. The study reports *qualitative* bias in subjective assessments, not a specific numerical rate gap of 8%.
- **Obermeyer et al. (2019)**: Found that a healthcare algorithm using cost as a proxy for health needs systematically underestimated Black patients' needs. At a given risk score, Black patients had ~26% more chronic illnesses than White patients. Remedying the bias would increase the fraction of Black patients receiving additional help from 17.7% to 46.5%. This paper is cited in the Related Work section but NOT used for calibration.

### Assessment

**MAJOR ISSUE: No quantitative calibration link exists.** Hoffman et al. (2016) does not report a "0.08 favorable decision rate gap." The paper documents that racial bias exists in pain assessment, but the specific number 0.08 is not derived from any cited source. Similarly, Obermeyer et al. (2019) deals with cost-based algorithmic bias (a very different mechanism than the agent decision-making modeled here) and its specific findings (e.g., 17.7% vs 46.5%) do not map onto the 0.90/0.82 rate structure.

The stakes (0.95) and harm weight (0.90) for healthcare are intuitively reasonable (healthcare decisions can be life-threatening) but are not derived from any cited source.

**Verdict**: Very weak calibration. The cited literature establishes that healthcare bias *exists* but does not support the specific parameter values chosen. The paper should be more explicit that these are assumed rather than empirically derived.

---

## 3. Content Moderation: Buolamwini & Gebru (2018) / Sap et al. (2019)

### Model parameters
- Content moderation: ingroup_base_rate=0.75, outgroup_base_rate=0.68 (gap of 0.07)
- stakes=0.60, harm_weight=0.50

### What the cited papers actually found
- **Buolamwini & Gebru (2018)**: Studied gender classification in facial recognition, finding error rates up to 34.7% for darker-skinned females vs. 0.8% for lighter-skinned males. This is about **facial recognition accuracy**, not content moderation decisions.
- **Sap et al. (2019)** (cited in code but NOT in references.bib): Found that AAE tweets are up to **2x more likely** to be flagged as offensive. This is closer to content moderation but suggests a much larger relative gap than the 0.75/0.68 rates used.

### Assessment

**MAJOR ISSUE: Domain mismatch.** Buolamwini & Gebru is about facial recognition accuracy disparities, not content moderation. It is cited in the Related Work to establish that AI systems have bias, but it does not calibrate the content moderation parameters. Sap et al. (2019), which is actually relevant to content moderation, is cited in the code comments but is NOT included in `references.bib` and NOT cited in the paper text.

The 2x flagging rate difference from Sap et al. would imply a much larger gap than the 0.07 used (e.g., if ingroup rate is 0.75, 2x disparity would mean outgroup ~0.375, not 0.68).

**Verdict**: No empirical calibration. The cited literature is from a different domain (facial recognition vs. content moderation). The relevant paper (Sap et al.) is missing from the bibliography.

---

## 4. Education: Dee (2005)

### Model parameters
- Education: ingroup_base_rate=0.88, outgroup_base_rate=0.80 (gap of 0.08)
- stakes=0.70, harm_weight=0.70

### What the cited paper found
- **Dee (2005)** (cited in code, NOT in references.bib): Found that teachers are 33% more likely to report a student as inattentive when they don't share a race/ethnicity, and 22% more likely to report a student as rarely completing homework.

### Assessment

Dee (2005) documents teacher expectation bias but measures it as *odds ratios* for subjective evaluations, not as favorable decision rates. The 0.88/0.80 gap (9% relative difference) does not directly correspond to the 22-33% odds ratios reported by Dee. Additionally, Dee (2005) is cited only in the code comments and is **NOT in references.bib or the paper text**.

**Verdict**: Weak calibration with missing citation. The literature supports the existence of education bias but not the specific values.

---

## 5. Customer Service: Gneezy & List (2004)

### Model parameters
- Customer service: ingroup_base_rate=0.85, outgroup_base_rate=0.80 (gap of 0.05)
- stakes=0.30, harm_weight=0.30

### What the cited literature found
- **Gneezy, List, & Livingston (2006/2012)** (cited in code as "Gneezy & List (2004)"): Their field experiments studied discrimination in marketplace settings like auto repair shops (disabled customers received 30% higher price quotes) and baseball card markets. These are *marketplace* discrimination studies, not customer service quality studies per se.

### Assessment

The citation in the code ("Gneezy & List (2004): small service quality differentials") is vague and does not precisely map to a published study with that exact finding. The Gneezy-List research program covers many discrimination experiments, but "small service quality differentials" is not a direct quotation or precise finding. This citation is also NOT in references.bib or the paper text.

**Verdict**: No empirical calibration. The cited source is vague and not properly referenced.

---

## 6. Feedback Strength Parameters

### Model values
- healthcare_triage: 0.8
- hiring: 0.6
- education: 0.5
- content_moderation: 0.3
- customer_service: 0.2

### Assessment

These parameters have **NO empirical support** whatsoever. The code provides intuitive justifications (e.g., "early triage decisions strongly affect later ones") but no citations. The paper text offers narrative explanations but does not claim empirical grounding for feedback_strength.

The paper is relatively honest about this -- the horizon model explanation in Section 3.4 provides intuitive reasoning rather than claiming empirical calibration. However, these parameters significantly affect results through the bias accumulation mechanism.

**Verdict**: Entirely assumed. Reasonably justified by domain logic but not empirically grounded.

---

## 7. Cross-Cutting Issues

### Missing citations in references.bib
The code cites **three papers** (Dee 2005, Sap et al. 2019, Gneezy & List 2004) that are NOT included in references.bib and therefore not cited in the paper. These are the calibration sources for 3 of 5 domains (education, content moderation, customer service). This means the paper gives readers NO way to verify the calibration claims for these domains.

### Mismatch between paper claims and actual calibration
Table 1's caption states: "Parameters are calibrated from domain-specific empirical literature (Section 3.3)." However, Section 3.3 only discusses calibration for hiring and healthcare (and even those loosely). The other three domains receive no calibration discussion whatsoever. This caption is misleading.

### The "roughly 50% difference" claim
The paper states (Section 3.3): "Bertrand and Mullainathan documented a roughly 50% difference in callback rates between resumes with White-sounding and African-American-sounding names." This is correct (the relative difference is ~50%), but may mislead readers into thinking the absolute gap is much larger than the actual ~3.2 percentage points. The model's absolute gap of 13 percentage points (0.35 - 0.22) is ~4x larger than the real-world absolute gap.

### Sensitivity analysis partially addresses concerns
The sensitivity analysis (varying parameters +/- 50%) does help mitigate the impact of imprecise calibration. The finding that hiring and healthcare remain the top-risk domains across perturbations is reassuring. However, this does not excuse claiming "calibrated" parameters when they are largely assumed.

---

## Summary Table

| Domain | Cited Source | In references.bib? | Empirical Grounding | Rating |
|--------|-------------|--------------------|--------------------|--------|
| Hiring | Bertrand & Mullainathan (2004), Quillian et al. (2017) | Yes | Relative ratio loosely matches; absolute rates inflated 3-4x | Moderate |
| Healthcare | Hoffman et al. (2016) | Yes | Existence of bias confirmed; no quantitative link to 0.08 gap | Weak |
| Content Mod. | Buolamwini & Gebru (2018) / Sap et al. (2019) | Buolamwini yes; Sap no | Wrong domain (facial recognition vs. content mod.) | Very Weak |
| Education | Dee (2005) | No | Existence of bias confirmed; no quantitative link | Weak |
| Customer Svc. | Gneezy & List (2004) | No | Vague reference to unspecified study | Very Weak |
| feedback_strength | None | N/A | No empirical basis | None |

---

## Recommendations

1. **Add missing references** (Dee 2005, Sap et al. 2019, Gneezy & List) to references.bib and cite them in the calibration section.
2. **Revise Table 1 caption** to say "Parameters are informed by..." rather than "calibrated from..." since only hiring has approximate calibration.
3. **Explicitly state** in Section 3.3 which domain parameters are empirically grounded vs. assumed, rather than implying all five are calibrated.
4. **Clarify the Bertrand & Mullainathan interpretation**: state that the model captures the relative discrimination ratio (~36% from Quillian meta-analysis), not the absolute callback rates (which are ~6-10%, not 22-35%).
5. **Acknowledge** that content moderation parameters are assumed, as the cited Buolamwini & Gebru paper studies a different modality (facial recognition).
6. **Add empirical justification or explicit "assumed" labeling** for feedback_strength parameters.
