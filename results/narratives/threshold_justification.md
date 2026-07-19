### Clinical Justification for the Decision Threshold

The sklearn default classification threshold (0.50) is a mathematical convention,
not a clinical decision. For PCOS screening we adopt a lower threshold
(τ = 0.21) selected on the **validation set** to maximise recall subject
to a precision floor of 0.65. Three considerations motivate this choice.

**1. Asymmetric costs.** Missed PCOS diagnoses carry well-documented long-term
harms: sub-fertility, endometrial hyperplasia, type-2 diabetes (2–4× baseline
risk), and cardiovascular morbidity (Teede et al., 2018). A false positive, in
contrast, is resolved by a single confirmatory visit applying the full Rotterdam
work-up.

**2. Precedent from established screening programmes.** Mammography and
cervical cytology are calibrated toward high sensitivity for exactly this
reason — downstream diagnostic pathways absorb false positives cheaply, whereas
false negatives leave disease undetected until symptoms escalate.

**3. Empirical impact on the held-out test set.** Moving from τ = 0.50 to
τ = 0.21 changes the confusion matrix from (9 FN, 2 FP) to
(1 FN, 20 FP): a 89% reduction in missed cases at the cost of
18 additional referrals for confirmatory testing.

The threshold was tuned exclusively on validation-set predictions; the test set
was held out and evaluated once.
