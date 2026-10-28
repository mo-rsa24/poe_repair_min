to: the forward-process-and-the-score walk (dispatch.md, on integrate)
from: scene-cadence, walk mode, 2026-10-21

scene: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9 (Dissolving the Dog, version 1792564498-9605)
covers: piece 1 rung 1 .. piece 1 rung 4
last piece covered: piece 1 rung 4 (the one-shot draw and its density q(x_t | x_0))

frames:
- rung 1, scaling a draw scales its variance by c²: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-2 (existing frame, preset c = 0.7223 gives 0.5217); still: artifacts/results/how-the-forward-process-dissolves-an-image/four-derivation-steps.png (existing)
- rung 2, independent variances add: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-3 (existing frame, with the "Noise copies the picture" preset); no still of its own yet
- rung 3, two steps then induction: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-5 (two steps) and #derive-6 (every step into ᾱ_t) (existing frames); still: artifacts/results/how-the-forward-process-dissolves-an-image/four-derivation-steps.png (existing, shows #derive-5)
- rung 4, the one-shot draw and its density: https://claude.ai/artifact/X7QGLTxssbz41G7iTiRxC9#derive-7 (new); still: artifacts/results/how-the-forward-process-dissolves-an-image/the-one-shot-draw-and-its-density.png

what #derive-7 shows: the walk's 2D toy point x_0 = (0, 0.5) and ε = (0.3, −1.2), with the switch from the dog said in the frame's first sentence. It shows x_50 = (0.2167, −0.5210), the centre (0, 0.3458), d = 0.7223·ε, height 0.3050, exponent −0.7650 and q = 0.1419. The t handle drags; presets t = 25 (q = 0.4505) and t = 75 (q = 0.0855) are there. A prediction (t = 50 to t = 25: what does the exponent do?) holds the cancellation line and the exponent-against-t curve as "?" until it is answered or t is moved. The broken preset, also a toggle, puts the centre at ᾱ_t·x_0 = (0, 0.2391): d becomes (0.2167, −0.7601), the exponent −0.5987, and a dashed red curve shows it moving with t.
answer key: scripts/scene_stills/density_frame_key.py, output in density_frame_key.out; every dispatch number for rung 4 reproduced.
