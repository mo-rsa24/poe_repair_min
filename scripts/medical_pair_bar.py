"""The pass bar for plans/10-does-roentgen-draw-two-findings, fixed before any image is generated.

P: both-rate of the plain pair prompt on the rare pair (Cardiomegaly + Pneumothorax)
C: both-rate of the plain pair prompt on the control pair (Cardiomegaly + Effusion)
K: both-rate of the composed adapters on the rare pair
Both-rate: fraction of the N_IMAGES_PER_PROMPT fixed-seed images in which the detector
calls both findings drawn, a call being a score above the value giving
CALL_SPECIFICITY on real NIH images without that finding.
"""

N_IMAGES_PER_PROMPT = 64
CALL_SPECIFICITY = 0.90
NULL_P_OVER_C = 0.5  # null if P >= this x C
SUPPORT_K_MINUS_P = 0.20  # support needs K >= P + this
SUPPORT_K_OVER_C = 0.8  # and K >= this x C
SUPPORT_SINGLE_KEEP = 0.8  # each single-finding rate in composed images >= this x its alone rate
MIN_DETECTOR_AUROC = 0.75  # below this on real NIH images for either finding: inconclusive


def verdict_plain(p, c, auroc_a, auroc_b):
    """Decide whether the problem exists, before any adapter is trained."""
    if min(auroc_a, auroc_b) < MIN_DETECTOR_AUROC:
        return "inconclusive"
    return "null" if p >= NULL_P_OVER_C * c else "problem-shown"


def verdict_composed(p, c, k, single_keep_a, single_keep_b):
    """Decide whether composition recovers the pair, after the adapters are trained."""
    if p >= NULL_P_OVER_C * c:
        return "null"
    ok = (
        k >= p + SUPPORT_K_MINUS_P
        and k >= SUPPORT_K_OVER_C * c
        and min(single_keep_a, single_keep_b) >= SUPPORT_SINGLE_KEEP
    )
    return "support" if ok else "not-supported"
