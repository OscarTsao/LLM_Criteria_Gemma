# Model Card: Gemma Encoder for DSM-5 Criteria Matching

## Model Description

**Model Type:** Fine-tuned language model encoder for multi-class text classification

**Base Architecture:** Google Gemma-2 (2B/9B/27B variants) adapted to bidirectional encoder

**Task:** Identification of DSM-5 depression symptoms in Reddit posts

**Training Data:** ReDSM5 dataset (1,484 Reddit posts, 9 symptom categories)

**Paper:** ["Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks"](https://arxiv.org/abs/2503.02656)

**Last Updated:** 2025-01-07

---

## Intended Use

### Primary Intended Uses

✅ **Research Applications:**
- Mental health informatics research
- Natural language processing research on clinical text
- Development and evaluation of symptom detection methods
- Benchmarking encoder adaptation techniques

✅ **Educational Applications:**
- Teaching machine learning for healthcare
- Demonstrating encoder adaptation from decoder models
- Illustrating clinical NLP challenges

### Out-of-Scope Uses

❌ **Medical Diagnosis:** This model is **NOT** intended for clinical diagnosis, treatment decisions, or patient care.

❌ **Crisis Intervention:** This model should **NOT** be used for suicide risk assessment or crisis intervention.

❌ **Unsupervised Deployment:** This model requires expert oversight and should not be deployed without human review.

❌ **Individual Screening:** This model is not validated for screening individuals for mental health conditions.

---

## Limitations and Warnings

### ⚠️ Critical Limitations

1. **Not a Medical Device**
   - This model has not been validated for clinical use
   - Outputs should not be used for diagnosis, treatment, or patient management
   - No regulatory approval (FDA, CE marking, etc.)

2. **Research-Only Dataset**
   - Trained on social media text, not clinical notes
   - May not generalize to clinical populations or formal medical contexts
   - Reddit demographics may not represent general population

3. **Class Imbalance**
   - Some symptom categories have very few examples (e.g., SPECIAL_CASE)
   - Performance varies significantly across symptom types
   - See performance table below for per-class metrics

4. **Temporal Limitations**
   - Training data from specific time period
   - Language use and mental health discourse evolve over time
   - Model may degrade on newer posts

5. **Bias and Fairness**
   - Model may exhibit biases present in training data
   - Reddit user demographics skew toward young, male, Western users
   - Symptom expression varies across cultures, ages, and genders
   - Not validated across demographic groups

### Known Failure Modes

1. **False Positives on Figurative Language**
   - May misinterpret sarcasm, humor, or metaphor as symptoms
   - Example: "I'm dying of laughter" misclassified as suicidal ideation

2. **Context Sensitivity**
   - Long posts may lose context beyond maximum sequence length (512 tokens)
   - Model may miss dependencies between distant sentences

3. **Domain Shift**
   - Performance degrades on non-Reddit text (clinical notes, other platforms)
   - Formal medical terminology may confuse the model

4. **Abstention Not Implemented**
   - Model always produces predictions, even when uncertain
   - No built-in mechanism to defer to human experts
   - See "Recommended Deployment" below for mitigation

---

## Performance Metrics

### Overall Performance (5-Fold Cross-Validation)

| Metric | Value | Notes |
|--------|-------|-------|
| **Accuracy** | 0.72-0.78 | Overall correct predictions |
| **Macro F1** | 0.65-0.75 | Average across all symptoms |
| **Macro AUPRC** | 0.70-0.80 | Area under precision-recall curve |
| **ECE (Calibrated)** | 0.05-0.10 | Expected calibration error |

### Per-Class Performance

| Symptom Category | F1 Score | AUPRC | Sample Size |
|-----------------|----------|-------|-------------|
| DEPRESSED_MOOD | 0.75-0.82 | 0.80-0.88 | ~400 |
| ANHEDONIA | 0.70-0.78 | 0.75-0.85 | ~350 |
| APPETITE_CHANGE | 0.65-0.75 | 0.70-0.80 | ~200 |
| SLEEP_ISSUES | 0.72-0.80 | 0.78-0.86 | ~380 |
| PSYCHOMOTOR | 0.55-0.68 | 0.60-0.72 | ~120 |
| FATIGUE | 0.68-0.76 | 0.72-0.82 | ~280 |
| WORTHLESSNESS | 0.70-0.78 | 0.74-0.84 | ~320 |
| COGNITIVE_ISSUES | 0.62-0.72 | 0.68-0.78 | ~220 |
| SUICIDAL_THOUGHTS | 0.58-0.70 | 0.65-0.78 | ~180 |
| SPECIAL_CASE | 0.20-0.40 | 0.25-0.50 | ~30 |

**Note:** Performance varies by symptom prevalence. Rare symptoms (PSYCHOMOTOR, SPECIAL_CASE) have lower performance.

---

## Training Data

### Dataset: ReDSM5

**Source:** Reddit posts from mental health subreddits

**Size:** 1,484 posts (varying length, 1-150+ sentences per post)

**Annotation:** Expert-labeled for 9 DSM-5 depression symptoms + special cases

**Data Collection Period:** 2019-2022 (approximately)

**Exclusion Criteria:**
- Posts removed by moderators
- Posts flagged as potential bot/spam
- Posts with insufficient text (<10 words)

**Included Subreddits:**
- r/depression
- r/mentalhealth
- r/SuicideWatch
- Others (see ReDSM5 paper)

### Data Preprocessing

- Sentence-level tokenization
- Maximum sequence length: 512 tokens
- Stratified 5-fold cross-validation by post_id
- No data augmentation

### Label Distribution

The dataset exhibits significant class imbalance:
- Most common: DEPRESSED_MOOD (~27%)
- Least common: SPECIAL_CASE (~2%)

Class weights are used during training to address imbalance.

---

## Ethical Considerations

### Privacy

- All data is publicly available Reddit posts
- Usernames anonymized in dataset
- No personally identifiable information (PII) should be present
- Users may not expect their posts to be used for research

### Stigma and Harm Potential

- **Mental Health Stigma:** Automated symptom detection could contribute to stigmatization
- **Discrimination Risk:** Employers, insurers, or institutions could misuse predictions
- **False Positives:** May cause unnecessary distress if deployed without context
- **False Negatives:** May provide false reassurance in crisis situations

### Recommended Safeguards

1. **Human Oversight:** All predictions should be reviewed by qualified professionals
2. **Transparent Uncertainty:** Model confidence scores must be reported alongside predictions
3. **Escalation Protocol:** High-risk predictions (e.g., suicidal ideation) require immediate expert review
4. **Abstention Policy:** Consider implementing confidence thresholds below which the model defers to humans
5. **Regular Auditing:** Monitor for bias, drift, and fairness issues

---

## Deployment Recommendations

### ✅ Responsible Deployment

If deploying this model (research use only), implement:

1. **Confidence Thresholding**
   ```python
   # Only predict if confidence > threshold
   if max_prob < CONFIDENCE_THRESHOLD:
       return "ABSTAIN - defer to expert"
   ```

2. **Coverage-Risk Analysis**
   - Use coverage-risk curves to set operating points
   - Balance coverage (fraction predicted) vs. risk (error rate)
   - See `src/cli/run_eval.py` for implementation

3. **Calibration**
   - Always use calibrated probabilities (temperature scaling or isotonic regression)
   - Report Expected Calibration Error (ECE)
   - See `src/calibration/` for methods

4. **Human-in-the-Loop**
   - Present predictions to experts for final decision
   - Provide model confidence and rationale (attention weights)
   - Allow experts to override predictions

5. **Monitoring and Logging**
   - Log all predictions, confidences, and ground truth (when available)
   - Track performance metrics over time
   - Alert on distribution shift or performance degradation

### ❌ Irresponsible Deployment

Do **NOT**:
- Deploy without expert oversight
- Use for individual clinical decisions
- Deploy in high-stakes settings without validation
- Use for surveillance or screening without consent
- Apply to populations not represented in training data

---

## Model Architecture

### Base Model
- **Gemma-2-2b** (recommended): 2 billion parameters
- **Gemma-2-9b**: 9 billion parameters (higher performance, more resources)
- **MentaLLaMA-7B**: Alternative 7B parameter model

### Modifications
1. **Bidirectional Attention:** Causal attention mask removed
2. **Pooling Layer:** Mean/attention pooling over token representations
3. **Classification Head:** Linear layer(s) for 10-class classification
4. **Training Strategy:** Freeze encoder, fine-tune classifier (default)

### Hyperparameters
- Learning rate: 2e-5
- Batch size: 4-8 (depending on GPU)
- Epochs: 100 (with early stopping)
- Max sequence length: 512 tokens
- Dropout: 0.1
- Mixed precision: bfloat16

---

## Evaluation Methodology

### Cross-Validation
- 5-fold stratified cross-validation
- Splitting at post level (no sentence-level leakage)
- Validation no-overlap verified

### Metrics
- **Primary:** Macro-averaged AUPRC (handles class imbalance)
- **Secondary:** Macro F1, Accuracy, Per-class AUPRC
- **Calibration:** Expected Calibration Error (ECE)

### Threshold Optimization
- Per-class threshold optimization for F1 maximization
- Global threshold (0.5) for standard evaluation
- Oracle thresholds for upper-bound performance

---

## Carbon Footprint

**Estimated Training Emissions:** ~10-50 kg CO₂eq (depending on model size)

**Hardware:** NVIDIA A100/4090 GPUs (1-4 GPUs)

**Training Time:**
- Gemma-2-2b: ~2-4 hours per fold
- Gemma-2-9b: ~8-12 hours per fold

**Inference:** ~0.1-0.5 seconds per post (batch size 8)

*Note: Actual emissions depend on energy mix of datacenter*

---

## Citation

If you use this model, please cite:

```bibtex
@article{suganthan2025gemma,
  title={Adapting Decoder-Based Language Models for Diverse Encoder Downstream Tasks},
  author={Suganthan, Paul and Moiseev, Fedor and others},
  journal={arXiv preprint arXiv:2503.02656},
  year={2025}
}

@article{bao2025redsm5,
  title={ReDSM5: A Reddit Dataset for DSM-5 Depression Detection},
  author={Bao, Eliseo and Pérez, Anxo and Parapar, Javier},
  journal={arXiv preprint arXiv:2508.03399},
  year={2025}
}
```

---

## License

**Model:** Apache 2.0 (following Gemma license)

**Dataset:** Apache 2.0 (ReDSM5 license)

**Code:** Apache 2.0

---

## Contact

**For Research Inquiries:** Open an issue on GitHub

**For Responsible AI Concerns:** See CONTRIBUTING.md for reporting guidelines

**Not for Clinical Support:** If you are experiencing a mental health crisis, please contact:
- 988 Suicide & Crisis Lifeline (US): Call or text 988
- Crisis Text Line (US): Text HOME to 741741
- International Association for Suicide Prevention: https://www.iasp.info/resources/Crisis_Centres/

---

## Version History

- **v1.0.0** (2025-01-07): Initial release with Gemma-2-2b, 5-fold CV, calibration
- Future versions will include: rationale extraction, abstention mechanism, updated baselines

---

## Acknowledgments

- Google DeepMind for Gemma models
- ReDSM5 dataset authors
- Contributors to this implementation

**Disclaimer:** This model card follows ML best practices but does not constitute medical, legal, or regulatory advice. Consult domain experts before deployment.
