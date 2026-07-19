### SHAP Feature Interpretation Linked to PCOS Literature

The top predictors identified by SHAP recapitulate the diagnostic constructs
codified by the Rotterdam ESHRE/ASRM PCOS Consensus Workshop (2004) and
refined by the International Evidence-Based Guideline (Teede et al., 2018).
Concordance between a data-driven model and consensus clinical criteria is
evidence that the model has learned physiologically meaningful signal rather
than dataset-specific artefacts.


| Rank | Feature | Mean \|SHAP\| | Literature basis |
|-----:|---------|--------------:|-------------------|
| 1 | LH(mIU/mL) | 0.656 | Luteinizing hormone is characteristically elevated in PCOS, with an inverted LH:FSH ratio > 1. Elevated LH drives ovarian theca-cell androgen production (Rosenfield & Ehrmann, 2016). |
| 2 | Follicle No. (R) | 0.302 | Antral follicle count ≥ 12 per ovary is a Rotterdam ultrasound criterion (Rotterdam ESHRE/ASRM, 2004). |
| 3 | Follicle No. (L) | 0.266 | Bilateral antral follicle count is the ultrasound cornerstone of the polycystic ovarian morphology criterion (Dewailly et al., 2014). |
| 4 | Weight gain(Y/N) | 0.228 | Weight gain reflects the insulin-resistance–adiposity axis central to PCOS pathophysiology; 50–70% of women with PCOS meet criteria for metabolic syndrome (Diamanti-Kandarakis & Dunaif, 2012). |
| 5 | hair growth(Y/N) | 0.219 | Hirsutism is the clinical expression of Rotterdam criterion #2 (hyperandrogenism), typically quantified by the Ferriman–Gallwey score (Teede et al., 2018). |
| 6 | Skin darkening (Y/N) | 0.197 | Acanthosis nigricans is a cutaneous marker of insulin resistance and is more prevalent in PCOS cohorts (Diamanti-Kandarakis & Dunaif, 2012). |
| 7 | Cycle(R/I) | 0.177 | Oligo- or anovulation is Rotterdam criterion #1 and the most common presenting symptom (Rotterdam ESHRE/ASRM, 2004). |
| 8 | Fast food (Y/N) | 0.162 | Diet quality modulates insulin resistance and androgen levels; dietary change is the first-line lifestyle intervention (Teede et al., 2018). |
| 9 | Pimples(Y/N) | 0.157 | Acne is a dermatological manifestation of hyperandrogenism (Rotterdam criterion #2). |
| 10 | AMH(ng/mL) | 0.100 | Anti-Müllerian hormone reflects the enlarged antral follicle pool and is 2–3× higher in PCOS; increasingly proposed as a Rotterdam substitute for follicle counting (Dewailly et al., 2014). |
| 11 | Hair loss(Y/N) | 0.066 | Androgenic alopecia is a further expression of hyperandrogenism (Teede et al., 2018). |
| 12 | Age (yrs) | 0.058 | PCOS is a reproductive-age condition; symptom profile shifts across adolescence, reproductive years, and peri-menopause. |
| 13 | Cycle length(days) | 0.055 | Cycle length > 35 days operationalises the oligo-ovulation criterion (Rotterdam ESHRE/ASRM, 2004). |
| 14 | Marraige Status (Yrs) | 0.055 | Not a direct PCOS biomarker; likely a proxy for age or fertility-seeking behaviour in this specific cohort. Flagged for caution and future ablation study. |
| 15 | PRG(ng/mL) | 0.052 | Mid-luteal progesterone is a proxy for ovulation; low levels support the anovulation criterion. |

**References**

- Rotterdam ESHRE/ASRM-Sponsored PCOS Consensus Workshop Group (2004). *Fertility and Sterility*, 81(1), 19–25.
- Teede HJ et al. (2018). Recommendations from the international evidence-based guideline for the assessment and management of PCOS. *Human Reproduction*, 33(9), 1602–1618.
- Dewailly D et al. (2014). Definition and significance of polycystic ovarian morphology. *Human Reproduction Update*, 20(3), 334–352.
- Rosenfield RL, Ehrmann DA (2016). The pathogenesis of PCOS revisited. *Endocrine Reviews*, 37(5), 467–520.
- Diamanti-Kandarakis E, Dunaif A (2012). Insulin resistance and PCOS revisited. *Endocrine Reviews*, 33(6), 981–1030.
