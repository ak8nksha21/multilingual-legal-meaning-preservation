# LegalDelta: Legal Meaning Preservation (LMP)

## 1. Objective

This project predicts how well a simplified legal sentence preserves
the meaning of its original version. It uses English and Hindi examples
from rental and terms-of-service/privacy contexts.

The proposed XLM-R paired-text regression model is inspired by
JUDGEBERT; it is not an exact reproduction of JUDGEBERT.

## 2. Dataset and evaluation

- Training examples: 228
- Validation examples: 50
- Held-out test examples: 79
- Languages: English and Hindi
- Split: source-disjoint train, validation, and test sets
- Human target: LMP score on a 1–10 scale

The reported test results are from saved artifacts. No new test
predictions or training runs were performed to create this report.

## 3. Overall model comparison

| Model                                    |   Pearson |   Spearman |   RMSE |    MAE |
|:-----------------------------------------|----------:|-----------:|-------:|-------:|
| TF-IDF cosine + linear calibration       |    0.6143 |     0.6766 | 1.0315 | 0.683  |
| Multilingual MiniLM cosine + calibration |    0.4502 |     0.5648 | 1.153  | 0.9236 |
| XLM-R paired-text regression             |    0.7284 |     0.649  | 0.9062 | 0.6597 |

The proposed XLM-R model improves on the recorded TF-IDF baseline in
RMSE, MAE, and Pearson correlation, but not Spearman correlation.
It outperforms the recorded MiniLM baseline on these four metrics.
The baseline figures are previously recorded results, not rerun here.

## 4. Language-wise results

| Language   |   Test examples |   RMSE |    MAE |   Pearson |   Spearman |   Mean human score |   Mean predicted score |
|:-----------|----------------:|-------:|-------:|----------:|-----------:|-------------------:|-----------------------:|
| en         |              41 | 1.0259 | 0.7823 |    0.7435 |     0.6425 |             8.6098 |                 8.9917 |
| hi         |              38 | 0.7561 | 0.5275 |    0.667  |     0.6498 |             9.2368 |                 9.2673 |

These results are descriptive. The subsets are small and their score
distributions differ, so they do not establish that the model is better
at understanding one language.

## 5. Domain-wise results

| Domain      |   Test examples |   RMSE |    MAE |   Pearson |   Spearman |   Mean human score |   Mean predicted score |
|:------------|----------------:|-------:|-------:|----------:|-----------:|-------------------:|-----------------------:|
| rental      |              48 | 0.9078 | 0.6037 |    0.7802 |     0.7244 |             8.9583 |                 9.0786 |
| tos_privacy |              31 | 0.9038 | 0.7465 |    0.6399 |     0.556  |             8.8387 |                 9.195  |

This analysis covers only the represented domains and has limited
sample sizes. It is an initial generalization check, not proof of
generalization across all legal domains.

## 6. Sanity checks

| case                | original                                                 | simplified                                                           | expected_behavior   |   predicted_LMP_score |
|:--------------------|:---------------------------------------------------------|:---------------------------------------------------------------------|:--------------------|----------------------:|
| Identical           | The agreement starts on 1 January.                       | The agreement starts on 1 January.                                   | Very high           |                 9.207 |
| Minor paraphrase    | The user must notify the company within 30 days.         | The user is required to notify the company within 30 days.           | Very high           |                 9.233 |
| Negation reversal   | The user may disclose confidential information.          | The user may not disclose confidential information.                  | Lower               |                 9.457 |
| Unrelated sentences | The tenant must pay rent by the fifth day of each month. | Photosynthesis converts light energy into chemical energy in plants. | Very low            |                 8.401 |

The model scores identical and mildly paraphrased pairs highly, as
expected. However, it assigns a high score to a pair where negation
reverses the legal meaning and to an unrelated pair. These are
qualitative probes, not formal benchmark metrics with human gold labels.

## 7. Difficult-case analysis

The following 15 examples were selected for high absolute error.
They are useful for diagnosing errors but are not a representative
random sample.

| ID            | Language   | Domain      |   LMP score |   Predicted LMP score |   Absolute error |
|:--------------|:-----------|:------------|------------:|----------------------:|-----------------:|
| en_rental_057 | en         | rental      |           4 |                7.2687 |           3.2687 |
| hi_rental_047 | hi         | rental      |           6 |                8.3213 |           2.3213 |
| en_tos_091    | en         | tos_privacy |           6 |                8.1577 |           2.1577 |
| hi_rental_044 | hi         | rental      |          10 |                7.9679 |           2.0321 |
| en_rental_050 | en         | rental      |           6 |                7.8933 |           1.8933 |
| en_tos_083    | en         | tos_privacy |           8 |                9.8049 |           1.8049 |
| en_tos_096    | en         | tos_privacy |           7 |                8.7843 |           1.7843 |
| hi_rental_051 | hi         | rental      |           8 |                9.763  |           1.763  |
| en_tos_099    | en         | tos_privacy |           8 |                9.6619 |           1.6619 |
| en_tos_081    | en         | tos_privacy |           8 |                9.4069 |           1.4069 |
| en_rental_052 | en         | rental      |           6 |                7.2998 |           1.2998 |
| hi_rental_065 | hi         | rental      |          10 |                8.7004 |           1.2996 |
| en_rental_053 | en         | rental      |          10 |                8.7958 |           1.2042 |
| en_tos_090    | en         | tos_privacy |           8 |                9.1671 |           1.1671 |
| hi_rental_040 | hi         | rental      |           7 |                8.1038 |           1.1038 |

Review the full original and simplified clauses when describing each
case. Potential error categories include omitted conditions, deadlines,
scope restrictions, permission changes, and over- or underestimation.
These classifications should not be treated as formal legal opinions.

## 8. Limitations

1. The dataset is modest in size for multilingual legal regression.
2. The available artifacts do not establish professional legal review
   or inter-annotator agreement for human scores.
3. The model does not reliably detect negation reversal or unrelated
   text in the sanity checks.
4. Language and domain breakdowns have limited sample sizes.
5. Historical baseline metrics should be identified as previously
   recorded results.
6. The model should not be used for autonomous legal decisions.

## 9. Conclusion

The XLM-R paired-text model is a promising prototype for legal meaning
preservation scoring in English and Hindi. It improves on the recorded
TF-IDF baseline in several error and correlation metrics, but not in
Spearman correlation. Sanity checks reveal important weaknesses in
distinguishing legal meaning preservation from surface similarity.

## 10. Future work

- Add human-annotated contrastive pairs for negation, obligations,
  permissions, deadlines, exceptions, and scope.
- Expand the dataset and include additional unseen legal sources.
- Establish independent human review and annotation agreement.
- Investigate contradiction-sensitive or ordinal objectives under a
  properly separated evaluation protocol.
