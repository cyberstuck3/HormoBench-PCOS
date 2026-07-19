"""Clinical narratives — threshold justification and SHAP-to-literature mapping."""


def threshold_justification(threshold, fn_default, fp_default, fn_new, fp_new):
    reduction_pct = int(round(100 * (fn_default - fn_new) / max(fn_default, 1)))
    return f"""### Clinical Justification for the Decision Threshold

The sklearn default classification threshold (0.50) is a mathematical convention,
not a clinical decision. For PCOS screening we adopt a lower threshold
(τ = {threshold:.2f}) selected on the **validation set** to maximise recall subject
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
τ = {threshold:.2f} changes the confusion matrix from ({fn_default} FN, {fp_default} FP) to
({fn_new} FN, {fp_new} FP): a {reduction_pct}% reduction in missed cases at the cost of
{fp_new - fp_default} additional referrals for confirmatory testing.

The threshold was tuned exclusively on validation-set predictions; the test set
was held out and evaluated once.
"""


SHAP_INTRO = """### SHAP Feature Interpretation Linked to PCOS Literature

The top predictors identified by SHAP recapitulate the diagnostic constructs
codified by the Rotterdam ESHRE/ASRM PCOS Consensus Workshop (2004) and
refined by the International Evidence-Based Guideline (Teede et al., 2018).
Concordance between a data-driven model and consensus clinical criteria is
evidence that the model has learned physiologically meaningful signal rather
than dataset-specific artefacts.
"""

SHAP_LIT = {
    'LH(mIU/mL)':
        "Luteinizing hormone is characteristically elevated in PCOS, with an "
        "inverted LH:FSH ratio > 1. Elevated LH drives ovarian theca-cell "
        "androgen production (Rosenfield & Ehrmann, 2016).",
    'Follicle No. (R)':
        "Antral follicle count ≥ 12 per ovary is a Rotterdam ultrasound "
        "criterion (Rotterdam ESHRE/ASRM, 2004).",
    'Follicle No. (L)':
        "Bilateral antral follicle count is the ultrasound cornerstone of the "
        "polycystic ovarian morphology criterion (Dewailly et al., 2014).",
    'Weight gain(Y/N)':
        "Weight gain reflects the insulin-resistance–adiposity axis central to "
        "PCOS pathophysiology; 50–70% of women with PCOS meet criteria for "
        "metabolic syndrome (Diamanti-Kandarakis & Dunaif, 2012).",
    'hair growth(Y/N)':
        "Hirsutism is the clinical expression of Rotterdam criterion #2 "
        "(hyperandrogenism), typically quantified by the Ferriman–Gallwey score "
        "(Teede et al., 2018).",
    'Skin darkening (Y/N)':
        "Acanthosis nigricans is a cutaneous marker of insulin resistance and "
        "is more prevalent in PCOS cohorts (Diamanti-Kandarakis & Dunaif, 2012).",
    'Cycle(R/I)':
        "Oligo- or anovulation is Rotterdam criterion #1 and the most common "
        "presenting symptom (Rotterdam ESHRE/ASRM, 2004).",
    'Fast food (Y/N)':
        "Diet quality modulates insulin resistance and androgen levels; dietary "
        "change is the first-line lifestyle intervention (Teede et al., 2018).",
    'Pimples(Y/N)':
        "Acne is a dermatological manifestation of hyperandrogenism "
        "(Rotterdam criterion #2).",
    'AMH(ng/mL)':
        "Anti-Müllerian hormone reflects the enlarged antral follicle pool and "
        "is 2–3× higher in PCOS; increasingly proposed as a Rotterdam substitute "
        "for follicle counting (Dewailly et al., 2014).",
    'Hair loss(Y/N)':
        "Androgenic alopecia is a further expression of hyperandrogenism "
        "(Teede et al., 2018).",
    'Age (yrs)':
        "PCOS is a reproductive-age condition; symptom profile shifts across "
        "adolescence, reproductive years, and peri-menopause.",
    'Cycle length(days)':
        "Cycle length > 35 days operationalises the oligo-ovulation criterion "
        "(Rotterdam ESHRE/ASRM, 2004).",
    'PRG(ng/mL)':
        "Mid-luteal progesterone is a proxy for ovulation; low levels support "
        "the anovulation criterion.",
    'Marraige Status (Yrs)':
        "Not a direct PCOS biomarker; likely a proxy for age or fertility-"
        "seeking behaviour in this specific cohort. Flagged for caution and "
        "future ablation study.",
}


def shap_interpretation_md(top_features):
    """top_features: list of (feature_name, mean_abs_shap)."""
    lines = [SHAP_INTRO, "",
             r"| Rank | Feature | Mean \|SHAP\| | Literature basis |",
             "|-----:|---------|--------------:|-------------------|"]
    for i, (feat, val) in enumerate(top_features, 1):
        basis = SHAP_LIT.get(feat, "— (feature not in prior-literature mapping)")
        lines.append(f"| {i} | {feat} | {val:.3f} | {basis} |")

    lines.append("")
    lines.append("**References**")
    lines.append("")
    lines.append(
        "- Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group (2004). "
        "*Fertility and Sterility*, 81(1), 19–25."
    )
    lines.append(
        "- Teede HJ et al. (2018). Recommendations from the international "
        "evidence-based guideline for the assessment and management of PCOS. "
        "*Human Reproduction*, 33(9), 1602–1618."
    )
    lines.append(
        "- Dewailly D et al. (2014). Definition and significance of polycystic "
        "ovarian morphology. *Human Reproduction Update*, 20(3), 334–352."
    )
    lines.append(
        "- Rosenfield RL, Ehrmann DA (2016). The pathogenesis of PCOS revisited. "
        "*Endocrine Reviews*, 37(5), 467–520."
    )
    lines.append(
        "- Diamanti-Kandarakis E, Dunaif A (2012). Insulin resistance and PCOS "
        "revisited. *Endocrine Reviews*, 33(6), 981–1030."
    )
    return '\n'.join(lines) + '\n'
