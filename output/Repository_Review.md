# Repository review for the updated MinneMUDAC presentation

Reviewed September 30, 2026. Audience: project judges and Big Brothers Big Sisters stakeholders.

The repository supports an exploratory workflow for extracting events from mentorship support notes, experimenting with match-duration prediction, and presenting patterns in Power BI. Saved artifacts substantiate the workflow concept. They do not establish production readiness, validated event extraction accuracy, causal effects of risk factors, or improvement from interventions.

## Scope and evidence

Reviewed the README, both standalone Python pipelines, principal modeling and extraction notebooks, dataset and findings documents, spreadsheet schemas and aggregates, all four existing PPTX files, and dashboard screenshots embedded in those presentations. Confirmed the PBIX file exists, but did not execute or refresh it. No live API calls, external transfer of support notes, new model training, or code changes were performed.

Two small synthetic checks reproduced the JSON normalization failure and missing error ledger entries without importing the API client or executing network calls. Saved notebook scores remain historical outputs and were not reproduced by retraining.

## Findings requiring attention

1. **Exposed API credential.** [grok_prompt_processor.py](/Users/Apple/Documents/MinneMUDAC-2025/Grok%20Prompt%20Pipeline/grok_prompt_processor.py:23) supplies a literal API key as an environment fallback. Revoke/rotate that credential, remove the fallback, and assess exposure in repository history. Its value is omitted from this review and presentation. Whether the credential remains active was not tested.

2. **The standalone scripts do not resolve their default inputs.** [advanced_ml_training.py](/Users/Apple/Documents/MinneMUDAC-2025/ML%20Pipeline/advanced_ml_training.py:39) and [grok_prompt_processor.py](/Users/Apple/Documents/MinneMUDAC-2025/Grok%20Prompt%20Pipeline/grok_prompt_processor.py:654) point to absent workbooks under `MUDAC/External Data`. The normal modeling notebook references `ML Completed Subset.xlsx`, which is also absent. The similarly named tracked workbooks cannot be substituted without verifying cohort and feature provenance.

3. **JSON response flattening fails for the pipeline's list-valued responses.** [grok_prompt_processor.py](/Users/Apple/Documents/MinneMUDAC-2025/Grok%20Prompt%20Pipeline/grok_prompt_processor.py:591) passes a Series of lists to `pd.json_normalize`. A synthetic valid one-item response list produced `ValueError: 'events' column not found in normalized data`. Expand response records deliberately and preserve Match ID and note identity through any one-to-many transformation.

4. **Failed queries do not populate the caller's error ledger.** [grok_prompt_processor.py](/Users/Apple/Documents/MinneMUDAC-2025/Grok%20Prompt%20Pipeline/grok_prompt_processor.py:493) rebinds `error_df` locally. In a synthetic failed query, the returned result had one failed response and zero error rows. Return the error record, or collect errors in the calling function, then persist them with the batch result.

5. **Preprocessing uses information outside the training partition.** [advanced_ml_training.py](/Users/Apple/Documents/MinneMUDAC-2025/ML%20Pipeline/advanced_ml_training.py:600) fits imputation and scaling before the sequence split. The normal notebook imputes before its split, and `minnie_mudac.ipynb` scales before cross-validation. Fit all learned transformations inside each training fold and use a separate evaluation set for hyperparameter selection. The LSTM notebook selects hyperparameters using one fold, then includes that fold in its reported five-fold averages.

6. **Prediction needs an explicit observation cutoff.** The normal notebook aggregates whole available histories, including note counts, maximum elapsed time, and closure-discussion labels. Those can reflect information unavailable at an early intervention point. The advanced pipeline lacks a strict list of fields available at prediction time. Benchmark on truncated histories matching the challenge or the intended operating setting. Exclude post-cutoff notes and closure-only fields. Active matches are ongoing, so elapsed length should not automatically serve as a final-duration target.

7. **Tree models read padding for shorter sequences.** [advanced_ml_training.py](/Users/Apple/Documents/MinneMUDAC-2025/ML%20Pipeline/advanced_ml_training.py:636) takes the final padded time position after sequences have been padded at the end. Use each sequence's recorded length to select its last observed row. The attention layer also lacks a mask excluding padded positions, and the stacker trains on in-sample base predictions. Use masked attention and out-of-fold base predictions for stacking.

8. **Extraction categories and data dates need review.** The saved flattened workbook contains 34 event fields, rather than the README's “35+.” Two categories contain no positive observations: `Volunteer: Lost contact with child/agency` and `Volunteer: Time constraint`. This can indicate a genuine absence or inconsistent extraction labels and needs inspection before interpretation. Of 25,050 rows, 511 dates do not parse with `pd.to_datetime(errors='coerce')`. The parsed date range extends to September 23, 2025. Do not assume all records match the dashboard's historical filters or treat a date parsed successfully as verified source chronology.

## Data summarized in the presentation

| Source | Verified scope |
|---|---|
| `MUDAC/Novice.xlsx` | 3,275 matches: 2,486 closed, 774 active, 15 pending closure |
| `MUDAC/Training_with_Life_Events.xlsx` | 39,345 rows, consistent with the data source brief's original training count |
| `MUDAC/Test_Truncated.xlsx` | 2,566 rows for 300 matches |
| `MUDAC/Apr3TrainRun1LLMFlattened1.xlsx` | 25,050 extracted rows for 3,275 matches, 34 event columns |
| `MUDAC/ML_Training_Test_Dataset.xlsx` and `Completed Test Dataset.xlsx` | Each contains 1,705 processed rows for 300 matches. Filename alone does not prove training provenance. |

Closed-match duration has a median of 15.7 months and an interquartile range of 8.7–28.2 months. These are descriptive summaries, not program outcome estimates.

The most frequent extracted event labels are COVID impact (3,758 rows), closure discussed (2,237), family lost contact with the volunteer (1,552), family moved (1,201), family time constraints (994), and match type changed (866). Counts include repeat observations of the same match, and multiple event types can appear on a row. They do not measure distinct affected matches or causal effects on closure.

## Model results and their limits

The normal notebook records Random Forest test RMSE of 2.8893 months and cross-validation RMSE of 2.7296 months. The tuned LSTM notebook records five-fold mean RMSE of 6.57 months, MAE of 3.94 months, and R² of 0.8900. Different input provenance and evaluation limitations prevent a direct ranking of those experiments. No persisted evaluation report for the standalone advanced ensemble script was found.

The README reports Team U37's Round 1 score as 60.29/80 and the all-team mean as 57.2. No original score sheet was found. The presentation identifies these as README-reported results and omits the unsupported approximate percentile assertion.

## Recommended pilot

Repair the extraction and model evaluation issues first. Have coordinators annotate a sample to measure event precision and recall, including ambiguous and missed events. Then evaluate a limited staff-reviewed pilot using agreed thresholds for review time, alert burden, cost per reviewed match, and subgroup performance. Track actual billed token usage and retries. Do not claim savings, reliable early warnings, or reduced closures until measured.

Automated calling, speech transcription, MatchForce integration, and live alerts are proposed extensions in the existing slides. The reviewed repository does not implement those integrations. The updated presentation identifies them as proposed.

## Presentation source treatment

The updated deck uses aggregate facts, editable charts and tables, a synthetic note example, and an existing dashboard screenshot without participant records. Speaker notes identify the repository sources and qualify historical scores. Existing presentations remain unchanged. The original dashboard screenshot is a saved illustration and may show counts or filters that differ from the workbook summaries.
