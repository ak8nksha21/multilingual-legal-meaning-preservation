# Explainable Multilingual Legal Meaning Preservation

## Project Overview

This project investigates whether simplified legal text preserves the meaning of its original version in English and Hindi.

It compares two semantic similarity baselines with a JUDGEBERT-inspired XLM-RoBERTa regression model. The model predicts a Legal Meaning Preservation (LMP) score on a 1–10 scale.

**Important:** This is a JUDGEBERT-inspired approach, not an exact reproduction of the original JUDGEBERT implementation.

## Objectives

- Estimate meaning preservation between original and simplified legal sentences.
- Compare TF-IDF, Sentence Transformers (MiniLM), and XLM-RoBERTa regression.
- Evaluate generalization across legal domains.
- Test identical and unrelated sentence pairs.
- Analyze legally important changes such as omitted conditions, exceptions, scope, and retention periods.
- Produce scores, provisional preservation decisions, and qualitative explanations.

## Dataset

- Total sentence pairs: 357
- Languages: English and Hindi
- Domains: rental agreements and terms/privacy policies
- Target: supplied LMP ratings on a 1–10 scale

The dataset contains a high proportion of ratings between 8 and 10. Results should therefore be interpreted in light of label imbalance and dataset size.

Human-rating provenance and annotation agreement should be confirmed before making claims about independently validated human judgments.

## Methods

### Baselines

1. **TF-IDF:** Text representations with a linear regression calibration fitted on training data.
2. **MiniLM:** Sentence embeddings with a linear regression calibration fitted on training data.
3. **XLM-RoBERTa:** A transformer regression model trained to predict LMP ratings.

The clean baseline comparisons use the same reconstructed train/validation/test split. TF-IDF fitting and baseline calibration are restricted to training data.

### Evaluation Metrics

- Pearson correlation
- Spearman rank correlation
- Root Mean Squared Error (RMSE)
- Mean Absolute Error (MAE)

### Generalization Experiment

Models were evaluated on a domain excluded from their training and validation data:
- Train on rental agreements; test on terms/privacy policies.
- Train on terms/privacy policies; test on rental agreements.

### Qualitative and Sanity Checks

The project includes identical-pair and unrelated-pair checks, plus 15 manually reviewed cases involving potentially legally meaningful changes. These qualitative assessments are preliminary and are not legal advice.

## Results

### Clean Test-Set Comparison

| Model | Pearson | Spearman | RMSE | MAE |
|---|---:|---:|---:|---:|
| TF-IDF | 0.5976 | 0.7100 | 1.1634 | 0.7924 |
| MiniLM | 0.5875 | 0.6574 | 1.1058 | 0.8477 |
| XLM-RoBERTa | 0.5784 | 0.4195 | 1.2325 | 0.9754 |

TF-IDF achieved the highest Pearson and Spearman correlations and the lowest MAE. MiniLM achieved the lowest RMSE. XLM-RoBERTa did not outperform these baselines on the reported test metrics.

These are descriptive comparisons; no statistical significance testing was performed.

### Cross-Domain Generalization

| Training domain | Test domain | RMSE | MAE | Pearson | Spearman |
|---|---|---:|---:|---:|---:|
| Rental | Terms/privacy | 3.6670 | 3.4596 | 0.0281 | -0.0017 |
| Terms/privacy | Rental | 1.7615 | 1.5398 | 0.3100 | 0.3012 |

The results indicate weak cross-domain generalization, especially when training on rental agreements and testing on terms/privacy policies.

### Sanity-Check Finding

The XLM-RoBERTa model assigned high scores to some unrelated sentence pairs. This is an important limitation: a high predicted LMP score should not be treated as proof that legal meaning is preserved.

## Preservation Decisions

The prediction output uses a provisional threshold of 8.0:
- Score >= 8: `Preserved`
- Score < 8: `Review: possible meaning loss`

This threshold has not been independently validated. Decisions should be reviewed by a qualified human, especially where obligations, exceptions, conditions, dates, amounts, or legal rights may have changed.

## Repository Structure

```text
judgebert_lmp_submission/
├── README.md
├── requirements.txt
├── requirements.txt
├── notebooks/
└── results/
    ├── lmp_model_comparison_clean.csv
    ├── held_out_domain_results.csv
    ├── lmp_sanity_check_results.csv
    ├── lmp_qualitative_case_analysis.csv
    ├── lmp_split_audit.csv
    ├── tfidf_clean_test_predictions.csv
    ├── minilm_clean_test_predictions.csv
    ├── xlmr_test_predictions_audit.csv
    ├── lmp_prediction_results.csv
    ├── generalization_by_domain.csv
    └── generalization_by_language.csv
```

## Reproducibility

The `notebooks/` directory contains:
- `LMP_Baseline_Experiments.ipynb`: the main experiment notebook copied from the working Colab notebook.
- `LMP_results_analysis.ipynb`: a reconstructed notebook for inspecting saved results.

The saved results are included for inspection. Verify notebook execution from a fresh runtime before claiming full reproducibility.

Before claiming full reproducibility, add the complete notebook, pin the software dependencies, document the dataset source and preprocessing, and verify that the experiments can be rerun from a clean runtime.

## Limitations

- Small dataset with an imbalanced LMP label distribution.
- Performance does not establish legal correctness.
- XLM-RoBERTa produced suspiciously high scores for some unrelated pairs.
- Cross-domain generalization was weak.
- The preservation threshold is provisional.
- Qualitative reviews are preliminary and should be independently checked.
- Results are not a substitute for legal review.

## References

- GRAAL Research, JUDGEBERT: https://github.com/GRAAL-Research/JUDGEBERT
- XLM-RoBERTa: https://huggingface.co/FacebookAI/xlm-roberta-base
- Sentence Transformers: https://www.sbert.net/

## Disclaimer

This is an academic research prototype for evaluating legal meaning preservation. It is not a legal advice system and should not be used to make legal decisions without qualified human review.
