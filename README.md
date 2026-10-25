# LegalDelta: Multilingual Legal Meaning Preservation

**A multilingual NLP system for evaluating whether simplified legal text preserves the meaning of the original statement.**

## 1. Project Overview

Legal documents often contain complex language involving rights, obligations, permissions, prohibitions, conditions, exceptions, dates, and numerical requirements. Simplifying such text can improve readability, but even a small wording change can alter its legal meaning.

**LegalDelta** is a multilingual legal meaning preservation system designed to assess whether a simplified legal statement retains the meaning of its original counterpart.

The project investigates whether multilingual transformer models can predict legal meaning preservation scores more effectively than generic text-similarity methods.

The current implementation supports English and Hindi and evaluates text pairs drawn from rental agreements and Terms of Service/privacy-policy documents.

### Objectives

- Develop a system that predicts legal meaning preservation (LMP) scores for original and simplified text pairs.
- Support multilingual legal text in English and Hindi.
- Compare generic semantic-similarity baselines with a transformer-based regression model.
- Evaluate generalization across languages and legal domains.
- Analyze difficult examples involving changes in meaning, including negation, obligations, conditions, and unrelated text.
- Identify limitations of automated legal meaning assessment.

## 2. Proposed Model: LegalDelta

LegalDelta is the **proposed system**, while XLM-RoBERTa is the transformer architecture used in the current implementation.

The system receives an original legal statement and a simplified statement and predicts a continuous Legal Meaning Preservation score on a 1–10 scale. Higher scores indicate greater preservation of the original meaning.

The current implementation focuses on score prediction. A score should not be interpreted as a legal determination or a guarantee that every legally significant change has been detected.

## 3. Implemented Architecture

The current LegalDelta model uses `xlm-roberta-base` as a multilingual paired-text regression model.

### Architecture pipeline

Original legal statement + simplified statement  
→ Paired-text tokenization  
→ XLM-RoBERTa encoder  
→ Pooled contextual representation  
→ Dropout  
→ Fully connected layer (256 units)  
→ GELU activation  
→ Dropout  
→ Linear regression output  
→ Predicted LMP score

### Main components

**1. Paired-text input**

The original and simplified statements are tokenized together, allowing the model to learn relationships between the two texts.

**2. Multilingual encoder**

XLM-RoBERTa generates contextual representations of the paired text. Its multilingual pretraining makes it suitable for experiments involving English and Hindi.

**3. Regression head**

The pooled representation passes through a small neural-network head that produces a continuous score.

**4. Combined training objective**

Training combines:

- **Smooth L1 regression loss:** Encourages predictions to match the human-assigned LMP scores.
- **Pairwise ranking loss:** Encourages the model to rank examples consistently with their relative preservation scores.

The regression targets are standardized using training-set statistics. Predictions are converted back to the original score scale for evaluation.

### Training configuration

| Parameter | Value |
|---|---|
| Base encoder | `xlm-roberta-base` |
| Maximum sequence length | 256 tokens |
| Batch size | 8 |
| Training epochs | 6 |
| Optimizer | AdamW |
| Learning rate | `1e-5` |
| Weight decay | 0.01 |
| Dropout | 0.2 |
| Regression loss | Smooth L1 |
| Pairwise ranking loss weight | 0.25 |
| Random seed | 42 |
| Model selection | Best validation RMSE |

The best checkpoint is selected using validation performance. The held-out test set is reserved for final evaluation.

## 4. Dataset and Experimental Design

The project uses a dataset of 357 original–simplified legal text pairs with human-assigned LMP scores.

The dataset includes:

- **Languages:** English and Hindi.
- **Legal domains:** Rental agreements and Terms of Service/privacy policies.
- **Target:** LMP score on a 1–10 scale.

### Source-disjoint data split

| Split | Number of pairs |
|---|---:|
| Training | 228 |
| Validation | 50 |
| Test | 79 |
| **Total** | **357** |

The data is divided by source document rather than randomly splitting individual pairs. This reduces the risk of the same source appearing in multiple splits.

Target normalization statistics are calculated from the training set only. The validation set is used for model selection, while the test set is used for the final reported evaluation.

## 5. Baselines

Two baseline approaches are used to compare the transformer-based model with more general text-similarity methods.

**TF-IDF cosine similarity with linear calibration**

Represents each text using TF-IDF features, computes cosine similarity, and uses a linear calibration model to estimate the LMP score.

**Multilingual MiniLM with calibration**

Uses multilingual sentence embeddings to represent the text and calibrates the similarity score to the LMP scale.

**LegalDelta improved model**

Uses the XLM-RoBERTa paired-text regression architecture trained with regression and pairwise ranking objectives.

All methods are evaluated using held-out test data.

## 6. Results

### Validation performance

The selected LegalDelta checkpoint achieved the following validation metrics:

| Metric | Validation result |
|---|---:|
| RMSE | 0.7526 |
| MAE | 0.5760 |
| Pearson correlation | 0.7573 |
| Spearman correlation | 0.7399 |

### Held-out test performance

| Metric | Test result |
|---|---:|
| RMSE | 0.9062 |
| MAE | 0.6597 |
| Pearson correlation | 0.7284 |
| Spearman correlation | 0.6490 |

Lower RMSE and MAE indicate smaller prediction errors. Higher Pearson and Spearman correlations indicate stronger linear and rank-order associations with the human-assigned scores, respectively.

### Baseline comparison

| Model | RMSE ↓ | MAE ↓ | Pearson ↑ | Spearman ↑ |
|---|---:|---:|---:|---:|
| TF-IDF + linear calibration | 1.0315 | 0.6830 | 0.6143 | **0.6766** |
| Multilingual MiniLM + calibration | 1.1530 | 0.9236 | 0.4502 | 0.5648 |
| LegalDelta improved model | **0.9062** | **0.6597** | **0.7284** | 0.6490 |

The improved model achieved the lowest test RMSE and MAE and the highest Pearson correlation among these three methods. TF-IDF achieved a higher test Spearman correlation, so the transformer model did not outperform every baseline on every metric.

These results support further investigation of the transformer-based approach but do not establish that it reliably detects every type of legal meaning change.

## 7. Language and Domain Analysis

### Performance by language

| Language | Test pairs | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| English | 41 | 1.0259 | 0.7823 | 0.7435 | 0.6425 |
| Hindi | 38 | 0.7561 | 0.5275 | 0.6670 | 0.6498 |

### Performance by legal domain

| Domain | Test pairs | RMSE | MAE | Pearson | Spearman |
|---|---:|---:|---:|---:|---:|
| Rental agreements | 48 | 0.9078 | 0.6037 | 0.7802 | 0.7244 |
| Terms of Service/privacy | 31 | 0.9038 | 0.7465 | 0.6399 | 0.5560 |

The results show differences across languages and legal domains. Because the evaluation subsets are relatively small, these differences should be interpreted cautiously rather than as definitive evidence of generalization.

## 8. Qualitative Sanity Checks

The project includes qualitative checks involving identical statements, minor wording changes, negation reversals, and unrelated statements.

The improved model produced approximately:

| Example type | Predicted score |
|---|---:|
| Identical statements | 9.21 |
| Minor wording change | 9.23 |
| Negation reversal | 9.46 |
| Unrelated statements | 8.40 |

These results expose an important weakness: the model assigns high scores even to some meaning-changing or unrelated pairs.

These are diagnostic examples, not formal aggregate metrics. They indicate that good overall regression metrics do not guarantee reliable detection of legally important contradictions or semantic mismatches.

## 9. Repository Structure

The project contains training code, reports, and saved evaluation artifacts.

```text
judgebert_lmp_submission/
├── README.md
├── notebooks/
│   └── LegalDelta_Improved_Model_Training.py
├── reports/
│   ├── LegalDelta_Final_Results_Report.md
│   └── LegalDelta_Final_Results_Report.docx
└── results/
    └── improved_candidate/
        ├── test_metrics.json
        ├── test_predictions.csv
        ├── language_metrics.json
        ├── domain_metrics.json
        ├── sanity_checks.json
        └── ...
```

The exact files under `results/improved_candidate/` depend on the artifacts included in the submitted repository. The full model checkpoint is stored separately in Google Drive rather than committed to Git.

## 10. Setup and Reproducibility

### Requirements

The training script uses Python and the following main libraries:

- PyTorch
- Transformers
- NumPy
- pandas
- scikit-learn
- SciPy

Install the dependencies in a suitable Python environment:

```bash
python -m pip install torch transformers numpy pandas scikit-learn scipy
```

Use a compatible PyTorch installation for your hardware. GPU acceleration is recommended for training the transformer model.

### Dataset preparation

The training script expects the source-disjoint training and validation CSV files with these columns:

- `Original`
- `Simplified`
- `LMP score`
- `Source`

The training and validation files used in the reported experiment were:

```text
train_source_grouped.csv
validation_source_grouped.csv
```

Place them in the location configured by the script, or update the `BASE` path to match your environment. The held-out test file and associated evaluation artifacts are required to reproduce the final test analysis.

**Note:** The dataset files are not automatically supplied by the installation command. Make sure the required data and the correct checkpoint are available before running inference or evaluation.

### Running the training script

From the project root, run:

```bash
python notebooks/LegalDelta_Improved_Model_Training.py
```

The script must be configured with the appropriate dataset paths and output directory for your environment. Training from the base encoder creates a new model; it does not automatically load the previously saved improved checkpoint.

The reported results correspond to the recorded experiment. Re-running training may produce slightly different results depending on the software versions, hardware, and runtime environment.

## 11. Model Checkpoint and Access

The trained improved-model checkpoint is stored in Google Drive:

```text
MyDrive/LegalDelta_LMP/LegalDelta_improved_candidate/
└── best_model_state.pt
```

The checkpoint is approximately 1.06 GB and is intentionally excluded from the normal Git repository to avoid storing a large binary file in source control.

The associated tokenizer, encoder configuration, metadata, and evaluation artifacts are stored alongside the model artifacts in the Google Drive output directory and/or the repository results folder.

To let a reviewer load the exact trained model, provide access to the checkpoint through an approved Google Drive share link or the institution's shared storage. The reviewer must have permission to access the file.

**Important:** The Drive path above is a location, not a public download link. A shareable link must be created and tested separately before it is included in the final submission.

## 12. Limitations

The current implementation has several limitations:

1. **Small dataset:** Fine-tuning a large multilingual transformer on 357 pairs limits the strength of generalization claims.
2. **Score imbalance:** The distribution of human-assigned scores can influence predictions and make low-preservation examples difficult to learn.
3. **Negation and contradiction:** Qualitative checks show that the model may assign high scores to text pairs with important meaning changes.
4. **Unrelated text:** The model does not consistently assign sufficiently low scores to completely unrelated statements.
5. **Language and domain coverage:** English, Hindi, rental agreements, and Terms of Service/privacy policies do not represent all legal languages and document types.
6. **No legal guarantee:** A predicted LMP score is an experimental model output, not legal advice or a verified judgment of equivalence.
7. **No guaranteed explanations:** The current implementation predicts a score; it does not reliably provide a legally validated explanation of each change.

## 13. Future Work

Future development of LegalDelta may include:

- Creating a balanced, larger dataset with more diverse legal meaning changes.
- Adding targeted challenge sets for negation, obligations, permissions, exceptions, numbers, dates, entities, and scope changes.
- Improving the treatment of unrelated text and contradictory statements.
- Evaluating calibration and confidence estimates.
- Expanding language and legal-domain coverage.
- Investigating interpretable outputs that highlight potentially altered legal obligations.
- Developing a preservation decision alongside the numerical score, after defining and validating decision thresholds.

These are future directions and should not be interpreted as capabilities already implemented.

## 14. Conclusion

LegalDelta investigates multilingual legal meaning preservation using human-scored text pairs, generic similarity baselines, and a transformer-based regression model.

The current XLM-RoBERTa implementation improves test RMSE, MAE, and Pearson correlation over the evaluated baselines, although TF-IDF performs better on Spearman correlation. Qualitative checks also reveal important weaknesses in identifying meaning-changing and unrelated pairs.

The project therefore demonstrates a promising experimental approach while highlighting the need for stronger legal-specific evaluation before the system can be relied upon for high-stakes use.

## 15. Reference

Beauchemin, D., Albert-Rochette, M., Khoury, R., & Déziel, P. (2025). *JUDGEBERT: Assessing Legal Meaning Preservation Between Sentences*. Proceedings of EMNLP 2025.

Paper: https://aclanthology.org/2025.emnlp-main.5.pdf

Reference implementation: https://github.com/GRAAL-Research/JUDGEBERT