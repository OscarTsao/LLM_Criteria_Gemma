# Model Card: ReDSM5 Sentence-Level Classification

## Intended Use
- **Primary goal:** Research on automated detection of DSM-5 Major Depressive Episode criteria in Reddit-derived text.
- **Users:** ML practitioners and clinical researchers familiar with privacy, ethics, and annotation constraints.
- **Inputs:** Sentence-level text excerpts with optional post metadata.
- **Outputs:** Multi-label probabilities for ten DSM-5 symptom categories.

## Limitations
- Not validated for clinical decision-making or patient triage.
- Reddit language differs from clinical notes; domain shift is expected.
- Class imbalance remains; per-class thresholds require recalibration on new cohorts.
- No robustness guarantees against adversarial or toxic prompts.

## Safety Considerations
- Never surface predictions to end users without human review.
- Abstain when maximum class probability < recommended thresholds; escalate uncertain cases.
- Strip personal health information (PHI) from logs, OOF exports, and shared artefacts.
- Retrain or recalibrate when data distribution shifts.

## Metrics
- Primary: Macro-AUPRC over post-level aggregation.
- Secondary: Macro-F1 with global (0.5) and per-class thresholds.
- Calibration: Expected Calibration Error (ECE) plus coverage-risk curves.

## Data Privacy
- ReDSM5 annotations contain sensitive mental health content; ensure compliance with data use agreements.
- OOF artefacts include post identifiers—store securely and anonymise before sharing.

## Maintenance
- Re-run Optuna sweeps when adding new encoders or altering preprocessing.
- Refresh thresholds & calibration when migrating to production stacks.
- Track configuration, seeds, and environment versions in experiment logs.
