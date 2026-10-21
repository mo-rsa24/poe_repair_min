# Where the proposal disagrees with itself

The Composability Gap PhD proposal (v2) checked against itself, before any outside reading. Each
entry quotes both places. Section numbers are the proposal's own. None of these is resolved here:
each is raised again in the walk at the piece where it changes the reading.

## Disagreements

**1. What causes the failure**

The Abstract says "A primary reason for this failure is the mathematical operator used to combine
conditions". §2.1.2 says "failures in compositional diffusion arise from entangled representations
rather than the composition rule itself", and H2 in §4.3 says the gap "is primarily a
representation problem". Chapter 7 names a third thing: "the mismatch between logical composition
and the semantically correct joint concept".

**2. Whether the single-prompt image is the correct answer**

§2.2.4 says "Single text-to-image diffusion models often fail when a prompt requires several
factors to be combined correctly". §5.1.4 then uses the single-prompt latent as the reference the
gap is measured from, and Chapter 7 calls it "the semantically correct joint concept". H1 in §4.3
reads a large unreachable gap as logical composition going where "no single prompt can reproduce",
while §5.1.1 calls the same Group 4 outcomes "the largest and most systematic failures". Phase 1
measures how far apart the two results are. Nothing in it measures which one is right.

**3. H2's quantity is not what Phase 2 measures**

H2 in §4.3 predicts "a smaller composability gap", and §6.2 asks Phase 2 to "reduce the gap between
semantic and logical composition". §5.2.3 measures DCI and the joint recovery accuracy of one
withheld orientation–scale pair. No single-prompt-versus-composed distance is computed in Phase 2.

**4. The group ordering**

H1 in §4.3 gives four rising levels: "smallest", "small-to-moderate", "large", "largest". §5.1.4
gives three: `J_G1 ≈ J_G2 < J_G3 < J_G4`.

**5. Which end of the trajectory is step 0**

§2.2 has `x_T` as pure noise and `x_0` as the image. §5.1.4 has `d_0` at the start of denoising
and `d_T` (also written `d_50`) at the finished image. Since every run in a pair-seed record starts
from "the same initial latent state" (§5.1.2), `d_0` is zero by construction.

**6. What Phase 3 actually combines**

§5.3.2, Eq. 5.4 combines noise predictions conditioned on the disease labels `c1` and `c2`, over
`u` given `z_c`. The disease latents `z_d1` and `z_d2` do not appear in it. §5.3.3 says
"composition refers to combining the two disease-specific latent factors over a shared anatomical
scaffold `z_c`".

**7. How many things Phase 3 is judged on**

§5.3 lists four: "pathology presence, spatial plausibility, radiological fidelity, and held-out
co-morbid composition". §5.3.3 lists three: "pathology presence, co-morbidity fidelity, and spatial
plausibility", and defines a metric for the first two only. §5.3.4 and §6.2 both require the output
to be "spatially plausible".

**8. How rare the joint case is**

§5.3.1 calls the joint presentation "relatively rare". The same paragraph counts 858 joint images
against 2,300 with cardiomegaly and 1,981 with pleural thickening, and the note on Figure 5.2 puts
that at 37.3% of cardiomegaly images and 43.3% of pleural-thickening images.

**9. Figure 5.4 against its caption**

The caption gives `n = 2,300` and `n = 1,981` and places pleural thickening "along the peripheral
lower pleural margins". The panel titles give 5,427 and 4,842 annotations, and the pleural
thickening density sits near the top of the frame, with its centroid marked at `(0.45, 0.36)` on an
axis where 0 is the top. The counts differ because one is images and the other is boxes. The
position is read off the rendered figure and should be confirmed against the plot's data.

**10. The second disease**

§2.1.2 ends on "cardiomegaly and pleural effusion". §5.3.1 onward uses cardiomegaly and pleural
thickening.

**11. Two sets of hypotheses**

Table 1.1 has a column headed "Hypotheses" under a caption that says "core assumptions", with two
items per phase. §4.3 has H1, H2 and H3. The Table 1.1 item "Bayes-style score composition succeeds
only when the model already contains a usable approximation to the desired joint configuration"
appears in neither H1 nor the Phase 1 evaluation.

## Things asked for with no way stated to check them

**No number on any pass rule.** §5.1.4 asks for a gap that is "measurable" and "emerges earlier and
more strongly"; §5.2.4 for performance that "improves most clearly in Regime D"; §5.3.4 for samples
that "recover both target pathologies". None says how large, how much earlier, or what rate.

**The group-level divergence has no estimator.** §5.1.4 defines the Jeffreys divergence between
pooled latent distributions. §5.1.2 gives 6 pairs by 24 seeds, so 144 records per group, and a
Stable Diffusion latent has thousands of dimensions. How a density is fitted to 144 points is not
said.

**Phase 3 has no comparison run.** H3 holds "provided the model has learned sufficiently distinct
pathology-specific representations". §5.3.2 trains one partitioned model. With no unpartitioned
model beside it, a good result cannot be credited to the partition.

**No rule assigns a pair to a group.** §4.2 says pairs are "grouped by co-occurrence support,
feature-space disentanglement, or semantic conflict", and §5.1.1 describes the four groups. Neither
lists the pairs or says how co-occurrence or feature overlap is measured. If the pairs are sorted
by judgement of how hard they look, the predicted ordering of the gap is partly built in.

**"Output-level similarity" has no metric.** §5.1.3 names it as an indicator. §2.2.3 introduces
CLIP and BLIP-VQA as tools for this, and §5.1.4 uses neither.

**Phase 2's diffusion model is not described.** §5.2 says each regime is judged "under the same
downstream conditional diffusion architecture". How a regime's latent enters that model, and
whether the withheld pair is generated by adding two conditional scores or by one joint condition,
is not stated.
