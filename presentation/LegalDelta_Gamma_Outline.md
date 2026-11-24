# LegalDelta: Multilingual Legal Meaning Preservation

**Team members**
- [Member 1 Name] — Enrolment No. [__________]
- [Member 2 Name] — Enrolment No. [__________]
- [Member 3 Name] — Enrolment No. [__________]

**Project repository (code, PPT and paper):**
https://github.com/ak8nksha21/multilingual-legal-meaning-preservation

[Course / Institution / Guide name] · [Date]

---

# The Problem: Simplification Can Change the Law

- Legal text is full of obligations, permissions, prohibitions, conditions, exceptions, dates and numbers
- Simplifying it makes it easier to read, but one changed word can change what it means legally
  - "may disclose" → "may not disclose" reverses a permission
  - Dropping "within 30 days" removes a deadline
- Simplification is often automated or crowdsourced, so someone has to check that the meaning survived
- Few tools do this, and fewer still handle Indian languages such as Hindi

---

# Problem Statement

Given an **original legal statement** and its **simplified version**, predict a **Legal Meaning Preservation (LMP) score from 1 to 10** that matches human judgement.

- Input: (Original, Simplified) text pair
- Output: continuous LMP score (higher = more meaning preserved)
- Languages: English and Hindi
- Domains: rental agreements, Terms of Service / privacy policies
- Research question: does a multilingual transformer trained on pairs beat generic text-similarity methods?

---

# Objectives

1. Build a system that predicts LMP scores for original–simplified legal pairs
2. Support multilingual legal text (English + Hindi)
3. Compare it with generic semantic-similarity baselines
4. Evaluate generalisation across languages and legal domains
5. Stress-test it on negation, paraphrase and unrelated text
6. Document limitations honestly

*Inspired by JUDGEBERT (Beauchemin et al., EMNLP 2025). This is an adaptation, not a reproduction.*

---

# Dataset

- **357 human-scored original–simplified pairs**
- Languages: English 205, Hindi 152
- Domains: ToS/Privacy 227 (Google, Netflix, …), Rental 130
- Human LMP labels on a 1–10 scale

| Score | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|
| Pairs | 4 | 5 | 17 | 34 | 71 | 102 | 124 |

**Key observation:** the labels are heavily skewed toward high scores (63% are 9 or 10), so meaning-loss examples are rare.

---

# Experimental Design: Source-Disjoint Split

| Split | Pairs | Purpose |
|---|---:|---|
| Train | 228 | Model fitting + target normalisation statistics |
| Validation | 50 | Checkpoint selection (best RMSE) |
| Test | 79 | Final evaluation, run once |

- Pairs are split **by source document**, not at random, so clauses from one contract never leak across splits
- Targets are standardised using training-set mean (8.59) and std (1.38) only
- The test set was evaluated only once, after model selection

---

# Methodology Overview

Three approaches, from simple to learned:

1. **TF-IDF cosine + linear calibration**: lexical overlap mapped to the 1–10 scale
2. **Multilingual MiniLM cosine + calibration**: sentence-embedding similarity mapped to the 1–10 scale
3. **LegalDelta (proposed)**: XLM-RoBERTa that reads both texts *jointly* and regresses the score

Why joint encoding? Similarity methods embed each sentence separately. A cross-encoder lets attention compare tokens such as *may* and *may not* directly across the two texts.

---

# LegalDelta Architecture

Original + Simplified
→ Paired tokenisation `<s> original </s></s> simplified </s>` (max 256 tokens)
→ **XLM-RoBERTa-base encoder** (multilingual, 100 languages)
→ Pooled [CLS] representation
→ Dropout (0.2) → Dense 256 → GELU → Dropout
→ Linear output → de-standardise → **LMP score (1–10)**

*(Visual suggestion: left-to-right block diagram)*

---

# Training Objective: Regression + Ranking

**Total loss = Smooth L1 (regression) + 0.25 × Pairwise ranking loss**

- **Smooth L1:** keeps predictions close to the human scores and is robust to outliers
- **Pairwise ranking:** for every pair of examples in a batch with different gold scores, softplus(−sign(Δy) · Δŷ) penalises getting their order wrong
- Why: plain regression on skewed labels collapses toward the mean. The ranking term pushes the model to *separate* good and bad simplifications

---

# Implementation Details

| Parameter | Value |
|---|---|
| Encoder | xlm-roberta-base |
| Max sequence length | 256 |
| Batch size | 8 |
| Epochs | 6 (best = epoch 6) |
| Optimiser | AdamW, lr 1e-5, weight decay 0.01 |
| Dropout | 0.2 |
| Ranking weight | 0.25 |
| Mixed precision | AMP (GPU) |
| Seed | 42 |

Stack: Python, PyTorch, Hugging Face Transformers, scikit-learn, SciPy, pandas
Checkpoint: about 1.06 GB, stored outside Git

---

# Training Progress

| Epoch | Val RMSE | Val Pearson | Val Spearman | Prediction std |
|---|---:|---:|---:|---:|
| 1 | 1.142 | 0.370 | 0.510 | 0.02 |
| 2 | 1.041 | 0.512 | 0.557 | 0.24 |
| 3 | 0.932 | 0.621 | 0.655 | 0.48 |
| 4 | 0.819 | 0.699 | 0.728 | 0.75 |
| 5 | 0.782 | 0.738 | 0.726 | 0.85 |
| 6 | **0.753** | **0.757** | **0.740** | 0.84 |

**Insight:** at epoch 1 the model predicts almost the same score for everything (std 0.02). Its predictions spread out as it learns, which is the effect the ranking loss is meant to encourage.

*(Visual suggestion: line chart of Val RMSE and Pearson by epoch)*

---

# Results: Held-Out Test Set (79 pairs)

| Model | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|
| TF-IDF + calibration | 1.032 | 0.683 | 0.614 | **0.677** |
| MiniLM + calibration | 1.153 | 0.924 | 0.450 | 0.565 |
| **LegalDelta (XLM-R)** | **0.906** | **0.660** | **0.728** | 0.649 |

- LegalDelta has the lowest error and the highest linear correlation
- TF-IDF still ranks pairs slightly better (Spearman)
- Validation result for the selected checkpoint: RMSE 0.753, Pearson 0.757

*(Visual suggestion: grouped bar chart)*

---

# Results by Language and Domain

| Subset | N | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| English | 41 | 1.026 | 0.782 | 0.744 | 0.643 |
| Hindi | 38 | 0.756 | 0.528 | 0.667 | 0.650 |
| Rental | 48 | 0.908 | 0.604 | 0.780 | 0.724 |
| ToS / Privacy | 31 | 0.904 | 0.747 | 0.640 | 0.556 |

- The model works in both languages with a single multilingual encoder
- Hindi's lower error partly reflects its higher mean human score (9.24 vs 8.61 for English)
- Subsets are small, so read these as indicative

---

# Cross-Domain Stress Test

Train on one domain, test on the other (exploratory baseline experiment):

| Train → Test | RMSE | Pearson |
|---|---:|---:|
| Rental → ToS/Privacy | 3.67 | 0.03 |
| ToS/Privacy → Rental | 1.76 | 0.31 |

**Insight:** performance collapses on an unseen legal domain. Training data has to cover the domains the model will be used on.

---

# Sanity Checks: Where the Model Fails

| Case | Example | Expected | Predicted |
|---|---|---|---:|
| Identical | "The agreement starts on 1 January." | Very high | 9.21 |
| Paraphrase | "must notify" → "is required to notify" | Very high | 9.23 |
| **Negation** | "may disclose" → "may **not** disclose" | Low | **9.46** |
| **Unrelated** | Rent clause vs. photosynthesis | Very low | **8.40** |

**Key finding:** the model has largely learned "these texts look alike", not "these texts mean the same legally". Good aggregate metrics hide this.

---

# Error Analysis: Highest-Error Cases

- Worst case: en_rental_057 (human score 4, predicted 7.27) omitted the security-deposit itemisation condition
- Recurring error types in the top 15:
  - Omitted conditions or purposes
  - Omitted scope or authority
  - Omitted retention periods and deadlines
  - Permission or terminology changes
- Direction: the model mostly **overestimates** preservation. It misses what was *dropped*
- Cause: few low-score training examples (only 26 pairs score 6 or below)

---

# Limitations

1. Small dataset (357 pairs) for fine-tuning a large transformer
2. Skewed labels, with few examples of meaning loss
3. Does not reliably detect negation or contradiction
4. Scores unrelated text too high
5. Only 2 languages and 2 legal domains
6. No inter-annotator agreement or professional legal review of the labels
7. Output is a research score, **not legal advice**

---

# Future Work

- Larger, balanced dataset with targeted contrastive pairs: negation, obligation↔permission, deadlines, numbers, exceptions, scope
- Contradiction-aware training (NLI-style auxiliary task, hard negatives)
- Calibration and confidence estimates, and a validated preserved/not-preserved threshold
- More Indian languages and legal domains
- Explainability: highlight which obligation or condition changed

---

# Conclusion

- **LegalDelta**, a multilingual XLM-R cross-encoder trained with regression and ranking losses, predicts legal meaning preservation in English and Hindi
- It beats TF-IDF and MiniLM on RMSE, MAE and Pearson on a leak-free, source-disjoint test set
- Sanity checks show it still confuses surface similarity with legal equivalence
- It is a promising prototype that needs contradiction-sensitive data before high-stakes use

**Repository:** https://github.com/ak8nksha21/multilingual-legal-meaning-preservation

---

# References

1. Beauchemin, D., Albert-Rochette, M., Khoury, R., & Déziel, P. (2025). *JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences.* EMNLP 2025. https://aclanthology.org/2025.emnlp-main.5.pdf
2. Conneau, A. et al. (2020). *Unsupervised Cross-lingual Representation Learning at Scale (XLM-R).* ACL 2020.
3. Wang, W. et al. (2020). *MiniLM: Deep Self-Attention Distillation for Task-Agnostic Compression of Pre-Trained Transformers.* NeurIPS 2020.

# Thank You · Questions?
