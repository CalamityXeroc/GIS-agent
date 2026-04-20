# [Title: Make It Clear and Specific]

**Authors:** Author Name¹, Author Name², Author Name¹
**Affiliations:** ¹Institution 1, ²Institution 2
**Contact:** corresponding.author@email.com

---

## Abstract

<!-- 150-300 words, self-contained summary -->
[**Background (1-2 sentences):**] [Why this problem matters]

[**Problem (1 sentence):**] [Specific problem addressed]

[**Solution (1-2 sentences):**] [Your key insight and approach]

[**Results (2-3 sentences):**] [Main findings with specific numbers]

[**Impact (1 sentence):**] [Significance and implications]

---

## 1. Introduction

<!-- Get to your contribution quickly (1-1.5 pages) -->

[**Hook paragraph:**] [Why should anyone care about this problem?]

[**Context paragraphs:**] [What's the broader landscape? What have others done?]

[**Gap paragraph:**] [What's missing? Why existing solutions fall short?]

[**Key insight:**] [What's your core idea that makes this work?]

[**Contributions paragraph:**]
This paper makes the following contributions:

• **[Contribution 1]:** [Novel technique/algorithm/framework] that [what it achieves] by [key mechanism]

• **[Contribution 2]:** [Theoretical analysis/proof] showing [property] with [bound/complexity] (Section X, Theorem Y)

• **[Contribution 3]:** [Empirical evaluation] demonstrating [performance gain] on [benchmarks] compared to [baselines] (Section X)

• **[Contribution 4]:** [Open-source release] of [code/data/model] at [URL] for reproducibility

---

## 2. Background and Problem Formulation

<!-- Define the problem formally (0.5-1 page) -->

### 2.1 Problem Definition

[Formal problem statement]

**Given:** [Input description]

**Goal:** [Objective, possibly with formal notation]

**Constraints:** [Any constraints or requirements]

### 2.2 Notation and Preliminaries

**Table 1:** Notation

| Symbol | Meaning |
|--------|---------|
| $n$    | [Description] |
| $G = (V, E)$ | [Description] |
| $\mathcal{D}$ | [Description] |

[Any background concepts needed to understand your approach]

---

## 3. Approach

<!-- Core technical content (2-3 pages) -->

### 3.1 Overview

[High-level intuition in plain language]

**Figure 1:** System architecture / Overview diagram
```
[Diagram showing overall approach]
```

### 3.2 [Component/Algorithm 1]

[Detailed description of first major component]

**Algorithm 1:** [Algorithm Name]
```
Input: [inputs]
Output: [outputs]
1: procedure NAME(parameters)
2:     initialize [variables]
3:     while [condition] do
4:         [step 1]
5:         [step 2]
6:     end while
7:     return [result]
```

[Explanation of algorithm logic and why it works]

### 3.3 [Component/Algorithm 2]

[Detailed description of second major component]

### 3.4 [Integration/Full System]

[How components work together]

### 3.5 Theoretical Analysis (if applicable)

**Theorem 1** (Complexity): [Statement]

*Proof sketch:* [Key ideas of the proof, or defer to appendix]

**Theorem 2** (Correctness): [Statement]

*Proof:* [Proof or reference to appendix]

---

## 4. Experimental Evaluation

<!-- Demonstrate that it works (2-3 pages) -->

### 4.1 Experimental Setup

**Datasets:**
We evaluate on [N] benchmark datasets:

- **Dataset A** [citation]: [Size, characteristics, why it's relevant]
- **Dataset B** [citation]: [Size, characteristics, why it's relevant]
- **Dataset C** [citation]: [Size, characteristics, why it's relevant]

**Baselines:**
We compare against [N] state-of-the-art methods:

- **Method 1** [citation]: [Brief description, year]
- **Method 2** [citation]: [Brief description, year]
- **Method 3** [citation]: [Brief description, year]

**Metrics:**
- **[Metric 1]:** [Why appropriate for this problem]
- **[Metric 2]:** [Why appropriate for this problem]
- **[Metric 3]:** [Runtime / scalability]

**Implementation Details:**
- Platform: [Python 3.10, PyTorch 2.0, etc.]
- Hardware: [GPU model, CPU specs]
- Hyperparameters: [Learning rate α = X, batch size = Y, etc.]
- Code: Available at [URL]

### 4.2 Main Results

**Table 2:** Performance comparison on [Dataset/Task]

| Method | Metric 1 ↑ | Metric 2 ↑ | Runtime ↓ | Year |
|--------|-----------|-----------|----------|------|
| Baseline 1 | XX.X% | XX.X% | XXs | 2023 |
| Baseline 2 | XX.X% | XX.X% | XXs | 2024 |
| **Ours** | **XX.X%** | **XX.X%** | **XXs** | 2025 |

[Analysis of results]
Our method achieves [performance] on [dataset], outperforming the best baseline ([method]) by [X%]. Notably, we achieve this while being [Y×] faster.

**Figure 2:** [Performance visualization - accuracy vs. time, scaling curves, etc.]
```
[Graph or chart showing key results]
```

### 4.3 Ablation Studies

[Systematically remove components to show what contributes]

**Table 3:** Ablation study on [Dataset]

| Configuration | Metric 1 | Metric 2 | Δ from Full |
|---------------|----------|----------|-------------|
| Full model | XX.X% | XX.X% | - |
| w/o [Component A] | XX.X% | XX.X% | -X.X% |
| w/o [Component B] | XX.X% | XX.X% | -X.X% |
| w/o [Both] | XX.X% | XX.X% | -X.X% |

[Interpretation]
These results show that [component A] contributes [X%] to overall performance, while [component B] contributes [Y%]. Both components are necessary for optimal performance.

### 4.4 Sensitivity Analysis

[How do results vary with hyperparameters, dataset characteristics, etc.?]

**Figure 3:** Sensitivity to [parameter]
```
[Graph showing performance vs. parameter value]
```

### 4.5 Qualitative Analysis (optional)

[Examples, case studies, error analysis]

---

## 5. Related Work

<!-- Position your work (can also be Section 2) (1-1.5 pages) -->

### 5.1 [Category 1 of Related Work]

[Thematic grouping of related work]

Work in this area includes [citations]. These approaches focus on [approach]. [Your work comparison].

### 5.2 [Category 2 of Related Work]

[Another thematic grouping]

### 5.3 [Category 3 of Related Work]

[Third grouping]

**Summary:** Unlike [prior work] which [limitation], our work [key difference] enabling [advantage].

---

## 6. Discussion (optional, can merge with Results or Conclusion)

<!-- Broader context, implications, limitations -->

**Implications:** [What do these results mean for the field?]

**Limitations:** This work has several limitations:
- [Limitation 1]: [Impact and potential mitigation]
- [Limitation 2]: [Impact and potential mitigation]

**Future Directions:** [Promising next steps]

---

## 7. Conclusion

<!-- Concise summary (0.5 page) -->

[**Problem restatement:**] This paper addressed [problem].

[**Approach summary:**] We proposed [approach] which [key innovation].

[**Results summary:**] Experiments on [datasets] demonstrated [main finding], achieving [performance metric].

[**Impact statement:**] This work [broader significance].

[**Future outlook:**] Future work will explore [direction].

---

## Acknowledgments

We thank [people/institutions] for [contribution]. This research was supported by [funding agency, grant number].

---

## References

<!-- Numbered style for CS conferences (IEEE, ACM) -->
[1] Author 1, Author 2. "Title." *Conference/Journal*, Year.

[2] Author 3. "Title." *Venue*, Year.

[...]

---

## Appendix A: Additional Proofs

[Full proofs of theorems if not included in main text]

## Appendix B: Additional Experiments

[Supplementary experimental results]

## Appendix C: Implementation Details

[Additional implementation details for reproducibility]

---

## Author Checklist (Remove before submission)

**Content:**
- [ ] Clear problem statement in intro
- [ ] Contributions list is specific and measurable
- [ ] Algorithm/approach is clearly explained
- [ ] Sufficient detail for reproducibility
- [ ] Experimental setup is thorough
- [ ] All baselines are state-of-the-art
- [ ] Ablation studies justify design choices
- [ ] Limitations are acknowledged

**Writing:**
- [ ] Abstract is self-contained
- [ ] All figures/tables referenced in text
- [ ] All claims have evidence
- [ ] Technical terms defined on first use
- [ ] Consistent notation throughout
- [ ] No grammatical errors

**Formatting:**
- [ ] Follows venue template
- [ ] Within page limit
- [ ] References formatted correctly
- [ ] Figures are readable
- [ ] Code/data URL included
- [ ] Supplementary material uploaded

**Ethics:**
- [ ] Proper attribution of prior work
- [ ] No overclaiming
- [ ] Negative results reported
- [ ] Reproducibility information included
- [ ] Broader impacts discussed (if required)
