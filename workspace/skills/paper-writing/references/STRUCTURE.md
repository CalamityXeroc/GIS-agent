# Paper Structure Templates

This document provides detailed structural templates for different types of academic papers. Use these as starting points and adapt to your specific venue requirements.

## Table of Contents

1. [Research Article (IMRAD)](#research-article-imrad)
2. [Computer Science Conference Paper](#computer-science-conference-paper)
3. [Literature Review / Survey Paper](#literature-review--survey-paper)
4. [Position Paper / Perspective](#position-paper--perspective)
5. [Technical Report](#technical-report)
6. [Thesis Chapter](#thesis-chapter)
7. [Short Paper / Extended Abstract](#short-paper--extended-abstract)

---

## Research Article (IMRAD)

**Standard format for experimental sciences, engineering, and empirical computer science**

### Structure

```
Title
Authors and Affiliations
Abstract (150-250 words)
Keywords (4-6 terms)

1. Introduction (1-2 pages)
   1.1 Background and Context
   1.2 Problem Statement
   1.3 Research Questions/Hypotheses
   1.4 Contributions
   1.5 Paper Organization

2. Related Work (1-2 pages)
   2.1 [Thematic Area 1]
   2.2 [Thematic Area 2]
   2.3 [Thematic Area 3]
   2.4 Positioning This Work

3. Methods (2-3 pages)
   3.1 Overview
   3.2 Experimental Design
   3.3 Materials/System Description
   3.4 Procedure
   3.5 Data Collection
   3.6 Analysis Methods

4. Results (2-3 pages)
   4.1 [Finding 1]
   4.2 [Finding 2]
   4.3 [Finding 3]
   4.4 Summary

5. Discussion (2-3 pages)
   5.1 Interpretation of Results
   5.2 Comparison with Previous Work
   5.3 Implications
   5.4 Limitations
   5.5 Future Work

6. Conclusion (0.5-1 page)

Acknowledgments
References
Appendices (optional)
```

### Content Guidelines

**Abstract Template**
```
[Background - 1-2 sentences]: What's the broad problem area?
[Gap - 1 sentence]: What's missing or problematic?
[Objective - 1 sentence]: What did you investigate?
[Methods - 1-2 sentences]: How did you do it?
[Results - 2-3 sentences]: What did you find?
[Conclusion - 1 sentence]: What does it mean?
```

**Introduction Flow**
```
Paragraph 1: Hook and broad context
    "Machine learning models are increasingly deployed in critical systems..."

Paragraph 2-3: Narrow the context, establish the problem
    "However, existing approaches to model validation have three limitations..."

Paragraph 4: Research gap
    "Despite extensive work on X and Y, the problem of Z remains unsolved..."

Paragraph 5: Your contribution
    "This paper addresses this gap by presenting [your approach]. Specifically,
    we make three contributions: (1)..., (2)..., (3)..."

Paragraph 6 (optional): Paper organization
    "The remainder of this paper is organized as follows..."
```

**Methods Section Checklist**
- [ ] Sufficient detail for replication
- [ ] Justification for design choices
- [ ] Clear description of variables/parameters
- [ ] Ethical approvals mentioned (if applicable)
- [ ] Statistical power analysis (if applicable)
- [ ] Diagrams of experimental setup
- [ ] Description of controls and baselines

**Results Section Checklist**
- [ ] Objective presentation (no interpretation)
- [ ] Clear connection to research questions
- [ ] Statistical significance reported
- [ ] Effect sizes provided
- [ ] Tables and figures with clear captions
- [ ] Both positive and negative results
- [ ] Raw data or summary statistics

**Discussion Section Structure**
```
Opening: Restate main findings briefly

Body paragraphs (each addresses one finding):
- State the finding
- Interpret it in context
- Compare with related work
- Explain implications

Limitations paragraph:
- Honest acknowledgment of weaknesses
- Why they don't invalidate findings
- How future work can address them

Future work paragraph:
- Specific next steps
- Open questions
- Broader directions
```

---

## Computer Science Conference Paper

**Typical for ACM, IEEE, AAAI, NeurIPS, ICML, etc.**

### Structure

```
Title
Authors and Affiliations
Abstract (150-300 words)

1. Introduction (1-1.5 pages)
   Problem motivation
   Limitations of existing approaches
   Key insight / core contribution
   Main results summary
   Contributions list
   Paper roadmap (optional)

2. Background and Related Work (1-2 pages)
   2.1 Problem Formulation
   2.2 Related Approaches
       2.2.1 [Category 1]
       2.2.2 [Category 2]
   2.3 Key Differences

3. Approach (2-3 pages)
   3.1 Overview
   3.2 [Component 1]
   3.3 [Component 2]
   3.4 Algorithm/Architecture
   3.5 Theoretical Analysis (if applicable)
       3.5.1 Complexity
       3.5.2 Correctness
       3.5.3 Convergence (if applicable)

4. Experimental Evaluation (2-3 pages)
   4.1 Experimental Setup
       4.1.1 Datasets
       4.1.2 Baselines
       4.1.3 Metrics
       4.1.4 Implementation Details
   4.2 Results
       4.2.1 Main Results
       4.2.2 Ablation Studies
       4.2.3 Sensitivity Analysis
   4.3 Discussion

5. Related Work (0.5-1 page, alternative placement)
   [If not in Section 2]

6. Conclusion and Future Work (0.5 page)

References
Appendix (supplementary material)
```

### Content Guidelines

**Contributions Format**
```
We make the following contributions:

• We propose [novel technique X], which addresses [problem Y] by [key insight Z].

• We provide theoretical analysis showing [property A] and prove [bound B]
  (Theorem 1).

• We demonstrate empirically that our approach outperforms state-of-the-art
  methods by [X%] on [benchmark Y], while requiring [Z×] less computation
  (Section 4).

• We release our implementation and datasets at [URL] to support reproducibility.
```

**Algorithm Presentation**
```
Present in three levels:

1. High-level intuition (text):
   "Our algorithm works by first preprocessing the data to identify key patterns,
   then iteratively refining predictions based on confidence scores..."

2. Visual representation (diagram):
   [Include system architecture or flowchart]

3. Formal specification (pseudocode):
   Algorithm 1: [Name]
   Input: ...
   Output: ...
   1: procedure NAME(parameters)
   2:     initialize ...
   3:     for each iteration do
   4:         ...
   5:     return result
```

**Experimental Section Template**
```
4.1 Experimental Setup

Datasets: We evaluate on three benchmark datasets:
• Dataset A: [brief description, size, characteristics]
• Dataset B: [brief description, size, characteristics]
• Dataset C: [brief description, size, characteristics]

Baselines: We compare against five state-of-the-art methods:
• Method 1 [Citation]: [one-line description]
• Method 2 [Citation]: [one-line description]
...

Metrics: We report:
• Accuracy: [definition and why it's appropriate]
• F1 Score: [definition and why it's appropriate]
• Runtime: [wall-clock time on specified hardware]

Implementation: We implemented our approach in Python using PyTorch.
Experiments ran on [hardware specs]. We used [optimizer] with learning
rate [X], batch size [Y], and trained for [Z] epochs. We report mean
and standard deviation over 5 runs with different random seeds.

4.2 Main Results

Table 1 shows results on all datasets. Our method achieves state-of-the-art
performance on X and Y, outperforming the best baseline by [N%] on average.
On dataset Z, we achieve comparable performance to [Method] while being
[M×] faster.

4.3 Ablation Studies

To understand which components contribute most to performance, we conduct
ablation studies (Table 2):

• Removing component A reduces accuracy by X%
• Removing component B reduces accuracy by Y%
• Combining both reduces accuracy by Z%

This demonstrates that both components are necessary...
```

---

## Literature Review / Survey Paper

**For synthesizing and analyzing existing research in a field**

### Structure

```
Title
Authors and Affiliations
Abstract (200-300 words)

1. Introduction (1-2 pages)
   1.1 Motivation and Scope
   1.2 Research Questions
   1.3 Methodology
   1.4 Contributions
   1.5 Organization

2. Background (1 page)
   2.1 Key Concepts and Terminology
   2.2 Historical Context
   2.3 Problem Definition

3. Methodology (1 page)
   3.1 Literature Search Strategy
   3.2 Inclusion/Exclusion Criteria
   3.3 Classification Framework
   3.4 Analysis Approach

4. [Thematic Section 1] (2-4 pages)
   4.1 [Subcategory A]
   4.2 [Subcategory B]
   4.3 Analysis and Comparison
   4.4 Open Problems

5. [Thematic Section 2] (2-4 pages)
   [Same structure as Section 4]

6. [Thematic Section 3] (2-4 pages)
   [Same structure as Section 4]

7. Cross-Cutting Analysis (1-2 pages)
   7.1 Common Themes
   7.2 Trade-offs
   7.3 Gaps in Literature
   7.4 Emerging Trends

8. Future Directions (1-2 pages)
   8.1 Open Problems
   8.2 Promising Approaches
   8.3 Research Opportunities

9. Conclusion (0.5-1 page)

References (extensive)
```

### Content Guidelines

**Introduction Template**
```
Paragraph 1: Why this topic matters
    "Topic X has received significant attention due to [importance]..."

Paragraph 2: Scope and boundaries
    "This survey covers work from [time period] on [specific aspects].
    We focus on [included topics] but do not cover [excluded topics]..."

Paragraph 3: Methodology
    "We identified papers through [search strategy], resulting in [N] papers
    that met our criteria..."

Paragraph 4: Organization and contributions
    "We organize the literature into [N] categories and make the following
    contributions: (1)..., (2)..., (3)..."
```

**Thematic Section Template**
```
4. [Category Name]

Opening paragraph: Define this category and explain its importance

4.1 [Subcategory A]

Work in this area focuses on [objective]. Key approaches include:

• Approach Type 1 [Citations]: These methods [description]. Representative
  work includes [specific papers with brief descriptions].

  Strengths: [bullet points]
  Limitations: [bullet points]

• Approach Type 2 [Citations]: These methods [description]...

  Strengths: [bullet points]
  Limitations: [bullet points]

4.2 [Subcategory B]
[Same structure]

4.3 Comparison and Analysis

Table X compares approaches along [dimensions]. We observe:

• Finding 1: [analysis]
• Finding 2: [analysis]
• Finding 3: [analysis]

The main trade-off in this category is between [X] and [Y]...
```

**Comparison Table Template**
```
Table 1: Comparison of [Category] Approaches

| Approach | Metrics | Scalability | Complexity | Year | Limitations |
|----------|---------|-------------|------------|------|-------------|
| Method A | 94.2%   | O(n log n)  | Medium     | 2024 | Requires labeled data |
| Method B | 91.5%   | O(n²)       | High       | 2023 | Not interpretable |
| Method C | 96.1%   | O(n)        | Low        | 2024 | Limited to domain X |
```

**Future Directions Template**
```
Based on our analysis, we identify the following promising research directions:

1. [Direction 1]: Current work has shown [progress], but significant gaps
   remain in [area]. Addressing this would enable [benefit].

2. [Direction 2]: The trade-off between [X] and [Y] is fundamental. New
   approaches that [strategy] could potentially achieve both.

3. [Direction 3]: Emerging trends in [area] suggest that [observation].
   Applying these ideas to [context] may yield [outcome].
```

---

## Position Paper / Perspective

**For presenting a viewpoint, critique, or vision for a field**

### Structure

```
Title (often provocative or question-based)
Authors and Affiliations
Abstract (150-200 words)

1. Introduction (0.5-1 page)
   The claim/position stated clearly
   Why it matters
   Paper roadmap

2. Background and Context (1 page)
   2.1 Current State of Affairs
   2.2 Why This Is Problematic
   2.3 Historical Perspective (optional)

3. The Case For [Your Position] (2-4 pages)
   3.1 Argument 1
       Evidence and reasoning
   3.2 Argument 2
       Evidence and reasoning
   3.3 Argument 3
       Evidence and reasoning

4. Addressing Counterarguments (1-2 pages)
   4.1 Objection 1
       Why it doesn't hold
   4.2 Objection 2
       Why it doesn't hold
   4.3 Objection 3
       Why it doesn't hold

5. Implications and Recommendations (1-2 pages)
   5.1 For Researchers
   5.2 For Practitioners
   5.3 For the Field

6. Conclusion (0.5 page)

References
```

### Content Guidelines

**Position Statement**
```
Clear and bold:
✅ "We argue that the field's focus on benchmark performance has led to
   fundamental misunderstandings about model capabilities."

Not vague:
❌ "There are some issues with how we evaluate models."
```

**Argumentation Structure**
```
For each main argument:

1. State the claim clearly
   "First, we argue that [claim]."

2. Provide evidence
   "Evidence for this comes from three sources. First, [evidence 1].
   Second, [evidence 2]. Third, [evidence 3]."

3. Explain the reasoning
   "These observations suggest that [interpretation]. This is significant
   because [implications]."

4. Connect to main thesis
   "This supports our overall position that [main claim]."
```

**Handling Counterarguments**
```
1. State the objection fairly
   "One might argue that [counterargument]."

2. Acknowledge validity (if any)
   "This concern is reasonable because [acknowledgment]."

3. Provide response
   "However, this objection doesn't undermine our position for three reasons.
   First, [response 1]. Second, [response 2]. Third, [response 3]."

4. Conclude
   "Thus, while [counterargument] raises important considerations, it doesn't
   invalidate [your position]."
```

---

## Technical Report

**For documenting systems, experiments, or detailed analyses**

### Structure

```
Title
Authors, Affiliations, Date
Version Number
Abstract / Executive Summary (1 page)

Table of Contents

1. Introduction (1-2 pages)
   1.1 Purpose and Scope
   1.2 Intended Audience
   1.3 Document Organization
   1.4 Related Documents

2. Background (1-2 pages)
   2.1 Context
   2.2 Requirements
   2.3 Constraints
   2.4 Assumptions

3. [Main Content Sections] (varies)
   [Organized by logical structure, not necessarily IMRAD]

   For system description:
   3. System Overview
   4. Architecture
   5. Components
   6. Interfaces
   7. Implementation

   For experiment report:
   3. Methodology
   4. Experimental Setup
   5. Results
   6. Analysis
   7. Discussion

8. Conclusions and Recommendations (1-2 pages)
   8.1 Summary of Findings
   8.2 Recommendations
   8.3 Future Work

References

Appendices
   A. Detailed Specifications
   B. Code Listings
   C. Data Tables
   D. Glossary
```

### Content Guidelines

**Executive Summary Template**
```
[Purpose]: This technical report documents [what].

[Background]: The work was conducted to [why], addressing [problem].

[Approach]: We [method summary in 2-3 sentences].

[Key Findings/Results]:
• Finding 1
• Finding 2
• Finding 3

[Recommendations]:
• Recommendation 1
• Recommendation 2

[Conclusions]: [One sentence bottom line]
```

**Technical Content**
- More detail than a paper (full equations, all parameters, complete specs)
- Can include negative results and dead ends
- Appendices for extensive details
- More figures and diagrams
- Step-by-step procedures
- Complete data tables

---

## Thesis Chapter

**For dissertation or thesis chapters**

### Structure

```
Chapter N: [Title]

N.1 Introduction (2-4 pages)
    N.1.1 Chapter Overview
    N.1.2 Connection to Thesis
    N.1.3 Chapter Contributions
    N.1.4 Organization

N.2 Background (2-4 pages)
    N.2.1 Concepts
    N.2.2 Prior Work
    N.2.3 Motivation

N.3 [Core Content Section 1] (4-8 pages)
    N.3.1 ...
    N.3.2 ...

N.4 [Core Content Section 2] (4-8 pages)
    N.4.1 ...
    N.4.2 ...

N.5 [Core Content Section 3] (4-8 pages)
    N.5.1 ...
    N.5.2 ...

N.6 Discussion (2-4 pages)
    N.6.1 Interpretation
    N.6.2 Limitations
    N.6.3 Implications

N.7 Summary (1-2 pages)
    N.7.1 Chapter Contributions
    N.7.2 Connection to Next Chapter

References (if separate per chapter)
```

### Content Guidelines

**Chapter Introduction**
```
Paragraph 1: What this chapter is about
    "This chapter presents [topic]. Specifically, we [objective]."

Paragraph 2: Why it matters for the thesis
    "This work addresses the second research question posed in Chapter 1:
    [question]. It builds on the [previous chapter topic] presented in
    Chapter [N-1] by..."

Paragraph 3: Specific contributions
    "The main contributions of this chapter are: (1)..., (2)..., (3)..."

Paragraph 4: Organization
    "The remainder of this chapter is organized as follows. Section N.2..."
```

**Chapter Summary**
```
Section N.7.1 Chapter Contributions

This chapter presented [topic]. The main contributions were:

• Contribution 1 (Section N.3): [one-sentence summary]
• Contribution 2 (Section N.4): [one-sentence summary]
• Contribution 3 (Section N.5): [one-sentence summary]

These contributions demonstrate [how they address research question].

Section N.7.2 Connection to Next Chapter

Having established [current chapter achievement], the next chapter
addresses [next topic]. Specifically, Chapter [N+1] will [preview]...
```

---

## Short Paper / Extended Abstract

**For workshops, poster sessions, or work-in-progress venues**

### Structure

```
Title
Authors and Affiliations
Abstract (100-150 words)

1. Introduction and Motivation (0.5-0.75 pages)
   Problem, gap, contribution (very concise)

2. Approach (0.75-1 page)
   Key idea and method overview
   [Optional: Small diagram]

3. Preliminary Results (0.75-1 page)
   [Table or figure with key results]
   Brief analysis

4. Related Work (0.25-0.5 pages)
   Brief positioning vs. most relevant work

5. Conclusion and Future Work (0.25 pages)

References (brief, ~10-15 papers)
```

### Content Guidelines

**Extreme Conciseness**
- Every sentence must count
- Focus on the key idea and main result
- Omit details available in full version
- Use figures to convey information efficiently

**What to Include**
- Clear problem statement
- Core novelty
- One compelling result
- Pointer to full version (if available)

**What to Omit**
- Extensive related work
- Detailed methodology
- Comprehensive experiments
- Lengthy discussion

**Example Introduction (for 2-page paper)**
```
Machine learning models deployed in production require robustness to
distribution shift. However, existing robustness metrics [citation]
fail to capture [specific issue].

This paper presents [method], a novel approach that addresses this gap
by [key insight]. Unlike prior work [citations] that [limitation], our
method [advantage].

We demonstrate on three datasets that [method] achieves [X]% improvement
in [metric] while maintaining [Y] computational efficiency. This suggests
[implication], opening new directions for [application].
```

---

## General Adaptation Guidelines

### Adjusting for Page Limits

**From 8 pages to 6 pages:**
- Combine Related Work into Introduction or move to end
- Reduce number of experimental conditions
- Move details to appendix
- Tighten writing throughout

**From 8 pages to 12 pages:**
- Expand Related Work with more comprehensive coverage
- Add more experimental conditions or datasets
- Include additional ablation studies
- Expand discussion and implications
- Add more detailed background

### Adjusting for Venue Culture

**Conference vs. Journal**
- Conference: More preliminary, novel ideas, faster pace
- Journal: More thorough, complete, polished

**Theoretical vs. Empirical Venue**
- Theoretical: More proofs, formal analysis, complexity
- Empirical: More experiments, datasets, empirical validation

**Academic vs. Industrial**
- Academic: Emphasize novelty and theory
- Industrial: Emphasize practical impact and deployment

---

## Quick Reference: Section Purposes

| Section | Primary Purpose | Secondary Purpose |
|---------|----------------|-------------------|
| Abstract | Summarize entire paper | Attract readers |
| Introduction | Motivate and frame problem | State contributions |
| Related Work | Position your work | Show knowledge of field |
| Method | Explain your approach | Enable replication |
| Experiments | Describe setup | Establish rigor |
| Results | Present findings | Support claims |
| Discussion | Interpret results | Acknowledge limitations |
| Conclusion | Summarize contributions | Point to future |

---

Use these templates as starting points, but always adapt to your specific content, venue requirements, and disciplinary conventions. The best structure serves your content and helps readers understand your contribution.
