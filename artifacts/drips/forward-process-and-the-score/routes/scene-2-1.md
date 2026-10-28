to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: scene-cadence, walk mode, 2026-10-21

scene: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9 (Dissolving the Dog, version 1792566915-9476)
covers: piece 2 rung 1 (the cosine boundary values C_k by hand), with the learner's by-hand build-up and the addendum's every-step items
last piece covered: piece 2 rung 1

frames (timetable tab, now 10 frames; links are named and stay put if frames are added later):
- the plane and the quarter circle: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-plane (new); still: artifacts/results/how-the-forward-process-dissolves-an-image/the-point-on-the-quarter-circle.png, first cell
- polar coordinates, legs cos θ and sin θ, cos² + sin² = 1 as one share bar, with a prediction on the y leg: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-polar (new); still: the-point-on-the-quarter-circle.png, second cell
- degrees and radians, the arc 0.8072 highlighted and laid flat beside π/2 = 1.5708: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-radians (new); still: the-point-on-the-quarter-circle.png, third cell
- the position ruler, 101 posts at (k + 0.8)/100.8 with a δ handle, δ = 0 as the broken run: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-posts (new); still: the-point-on-the-quarter-circle.png, fourth cell
- the quarter turn, cos² every 15° and its drops, with a prediction on which turn drops most: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-quarter-turn (new); still: the-point-on-the-quarter-circle.png, fifth and sixth cells
- one boundary term by term (k/T, + δ, ÷ (1 + δ), × π/2, cos, the square, ÷ cos²θ₀), presets k = 0, 1, 51, 99, 100 and δ = 0, fraction bars with the "Divide by cos²θ₀" switch: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#timetable-one-boundary (repaired; the old #timetable-2 link still opens it); still: artifacts/results/how-the-forward-process-dissolves-an-image/one-boundary-term-by-term.png

what the new frames show, on the walk's numbers:
- post 51 at 46.25° = 0.8072 rad sits at (0.6915, 0.7224), within 0.0001 of √ᾱ₅₀ = 0.6916 and √(1 − ᾱ₅₀) = 0.7223; 0.4782 + 0.5218 = 1.
- δ = 0.008 moves post 0 from 0° to 0.714°, and ÷ (1 + δ) keeps post 100 on 90°. With δ = 0, post 1 sits at 0.9°, C₁ = 0.99975, and the first step's noise share is 0.000247 (typical miss 0.0157), against 0.000631 and 0.0251.
- cos² at 0, 15, …, 90° = 1, 0.933, 0.75, 0.5, 0.25, 0.067, 0, with drops 0.067, 0.183, 0.25, 0.25, 0.183, 0.067.
- k = 0: 0.714°, cos 0.99992, cos² 0.99984, C₀ = 1. k = 1: 1.607°, cos² 0.99921, C₁ = 0.99937 = ᾱ₀. k = 51: 46.25°, cos 0.69151, cos² 0.47819, C₅₁ = 0.47826 = ᾱ₅₀. k = 100: 90°, C₁₀₀ = 0.
- With the switch off, C₀ reads 0.99984 and every C_k is 0.0155% low, while β₅₀ = 0.031546 both ways (largest difference over all 100 βs: 1.1 × 10⁻¹⁶).
- The square step reads "cos θ is the signal multiplier, cos²θ its variance share, because share = multiplier²". It does not say squaring keeps C_k non-negative.

answer key: scripts/scene_stills/circle_frame_key.py, output in circle_frame_key.out; every dispatch and addendum number reproduced.
checked: a browser walk of all 10 timetable frames and every preset at 1440 px light and 390 px dark, with 46 number and link checks, 0 failures, no console errors, no KaTeX errors, no sideways scroll. Not done: the Codex mathematics review.

for the walk to know:
- The other session's version (1792566272-bc54) inserted a new Derive 7 ("hold x₀ fixed: its draws form a Gaussian"), so the density frame that routes/scene-1-1-to-1-4.md calls #derive-7 is now #derive-8. That return file, the results README's density paragraph and scripts/scene_stills/capture_density_frame.mjs still say #derive-7.
- Leaving Derive 7 by Next mid-animation threw "Cannot set properties of null" in that version. This publish cancels the running animation when a frame is built, and the derive walk now runs clean.
